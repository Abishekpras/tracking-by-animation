import os.path as path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import modules.submodules as smd
import modules.utils as utils
from scipy.spatial import cKDTree

class TrackerArray(nn.Module):
    
    def __init__(self, o):
        super(TrackerArray, self).__init__()
        self.o = o
        self.ctr = -1
        self.ntm = NTM(o)

    def forward(self, h_o_prev, y_e_prev, C_o_seq, path, phase):
        """
        h_o_prev: N * O * dim_h_o
        y_e_prev: N * O * dim_y_e
        C_o_seq:  N * T * C2_1 * C2_2
        '"""
        o = self.o
        if o.new_track:
            self.mem = torch.zeros(o.N, o.O, 2).cuda()
            o.new_track = False
            self.ctr += 1

        # Split the input
        C_o = torch.unbind(C_o_seq, dim=1) # N * C2_1 * C2_2
        # Iterate over time
        h_o, y_e, y_l, y_p, Y_s, Y_a = {}, {}, {}, {}, {}, {}
        for t in range(0, o.T):
            self.ntm.t = t
            self.ntm.ntm_cell.t = t
            pos = None if path is None else path[:, t].cuda()
            self.mem, h_o[t], y_e[t], y_l[t], y_p[t], Y_s[t], Y_a[t] = self.ntm(h_o_prev, y_e_prev, C_o[t], pos, phase, self.mem) # N * O * dim_xx
            h_o_prev, y_e_prev = h_o[t], y_e[t]
            torch.save(self.mem[0], './plot_data/digits_memory_{}_{}.pt'.format(self.ctr, t))
            torch.save(y_e_prev[0], './plot_data/y_confidence_{}_{}.pt'.format(self.ctr, t))

        # Merge the states
        h_o_seq = torch.stack(tuple(h_o.values()), dim=1) # N * T * O * dim_h_o
        y_e_seq = torch.stack(tuple(y_e.values()), dim=1) # N * T * O * dim_y_e
        y_l_seq = torch.stack(tuple(y_l.values()), dim=1) # N * T * O * dim_y_l
        y_p_seq = torch.stack(tuple(y_p.values()), dim=1) # N * T * O * dim_y_p
        Y_s_seq = torch.stack(tuple(Y_s.values()), dim=1) # N * T * O * 1 * h * w
        Y_a_seq = torch.stack(tuple(Y_a.values()), dim=1) # N * T * O * D * h * w
        return h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq


class NTM(nn.Module):
    
    def __init__(self, o):
        super(NTM, self).__init__()
        self.o = o
        dim_y = o.dim_y_l + o.dim_y_p + o.dim_Y_s + o.dim_Y_a
        fcn_params = [o.dim_h_o] + o.fcn + [dim_y]
        self.fcn = smd.FCN(fcn_params, hid_trans='relu', out_trans=None)
        self.softmax = nn.Softmax(dim=1)
        self.st_gumbel_sigmoid = smd.STGumbelSigmoid()
        self.st_gumbel_softmax = smd.STGumbelSoftmax(1)
        self.permutation_matrix_calculator = smd.PermutationMatrixCalculator()
        self.ntm_cell = NTMCell(o)
        self.linear_pos_to_conf = nn.Linear(2, 1)
        self.linear_h_to_conf = nn.Linear(2, 1)
        self.t = 0
        self.att = torch.Tensor(o.T, o.O, self.ntm_cell.ha, self.ntm_cell.wa).cuda()
        self.mem = torch.Tensor(o.T, o.O, self.ntm_cell.ha, self.ntm_cell.wa).cuda()

    def forward(self, h_o_prev, y_e_prev, C_o, pos, phase, memor):
        """
        h_o_prev: N * O * dim_h_o
        y_e_prev: N * O * dim_y_e
        C_o:      N * C2_1 * C2_2
        """
        o = self.o
        perm_mat = torch.zeros(o.N, o.O, o.O)
        m = memor.cpu().detach()
        p = pos.cpu()
        for i in range(o.N):
            tree = cKDTree(m[i])
            r = tree.query(p[i], k=10)[1].reshape(-1)
            #r = np.array(r.tolist() + [sum(range(o.O)) - r.sum()])
            #print(r, p[i].tolist(), m[i].tolist())
            if(r.sum() != sum(range(o.O))):
                print(m[i])
                print(p[i])
                print(r)
            perm_mat[i, torch.arange(o.O), r] = 1
        perm_mat = perm_mat.cuda()
        perm_mat_inv = perm_mat.transpose(1, 2)
        
        #pos = pos / 120
        #inp = pos.unsqueeze(1).expand(o.N, o.O, 2).contiguous().view(-1, 2)
        #y_e_prev = self.linear_pos_to_conf(inp).tanh().abs().view(o.N, o.O, o.dim_y_e)
        #if "no_rep" not in o.exp_config:
        #    if o.train == 0:
        #        print(y_e_prev[0].view(-1))
        #    delta = torch.arange(0, o.O).type(torch.FloatTensor).cuda().unsqueeze(0) * 0.0001 # 1 * O
        #    y_e_prev_mdf = y_e_prev.squeeze(2).round() - Variable(delta)
        #    perm_mat = self.permutation_matrix_calculator(y_e_prev_mdf) # N * O * O
        #    perm_mat_inv = perm_mat.transpose(1, 2) # N * O * O
    
        if phase == 'obs':
            #if "no_tem" in o.exp_config:
            #    h_o_prev = Variable(torch.zeros_like(h_o_prev).cuda())W3
            #    y_e_prev = Variable(torch.zeros_like(y_e_prev).cuda())

            # Sort h_o_prev and y_e_prev
            if "no_rep" not in o.exp_config:
                h_o_prev = perm_mat.bmm(h_o_prev) # N * O * dim_h_o
            #    y_e_prev = perm_mat.bmm(y_e_prev) # N * O * dim_y_e
            #    if o.train == 0:
            #        print('Shuffled:')
            #        print(y_e_prev[0].view(-1))

            # Update h_o
            h_o_prev_split = torch.unbind(h_o_prev, 1) # N * dim_h_o
            h_o_split = {}
            k_split = {}
            r_split = {}
            for i in range(0, o.O):
                self.ntm_cell.i = i
                h_o_split[i], C_o, k_split[i], r_split[i] = self.ntm_cell(h_o_prev_split[i], C_o, pos, phase)
            h_o = torch.stack(tuple(h_o_split.values()), dim=1) # N * O * dim_h_o
            k = torch.stack(tuple(k_split.values()), dim=1) # N * O * C2_2
            r = torch.stack(tuple(r_split.values()), dim=1) # N * O * C2_2
            att = self.ntm_cell.att
            mem = self.ntm_cell.mem

            # Recover the original order of h_o
            if "no_rep" not in o.exp_config:
                #perm_mat_inv = perm_mat.transpose(1, 2) # N * O * O
                h_o = perm_mat_inv.bmm(h_o) # N * O * dim_h_o
                k = perm_mat_inv.bmm(k) # N * O * dim_c_2
                r = perm_mat_inv.bmm(r) # N * O * dim_c_2
                att = perm_mat_inv.data[self.ntm_cell.n].mm(att.view(o.O, -1)).view(o.O, -1, self.ntm_cell.wa) # O * ha * wa
                mem = perm_mat_inv.data[self.ntm_cell.n].mm(mem.view(o.O, -1)).view(o.O, -1, self.ntm_cell.wa) # O * ha * wa
                h_o_prev = perm_mat_inv.bmm(h_o_prev)
                #y_e_prev = perm_mat_inv.bmm(y_e_prev)
            
            if o.v > 0:
                self.att[self.t].copy_(att)
                self.mem[self.t].copy_(mem)
        else:
            h_o = h_o_prev

        y_e, y_l, y_p, Y_s, Y_a = self.generate_outputs(h_o, pos)

        # adaptive computation time
        if "act" in o.exp_config:
            #y_e_perm = perm_mat.bmm(y_e).round() # N * O * dim_y_e
            #y_e_mask = y_e_prev.round() + y_e_perm  # N * O * dim_y_e
            # y_e_mask = y_e_perm
            #y_e_mask = perm_mat.bmm(y_e).round() # N * O * dim_y_e
            #y_e_mask = y_e_mask.lt(0.5).type_as(y_e_mask)
            #y_e_mask = y_e_mask.cumsum(1)
            #y_e_mask = y_e_mask.lt(0.5).type_as(y_e_mask)
            #ones = Variable(torch.ones(y_e_mask.size(0), 1, o.dim_y_e).cuda())  # N * 1 * dim_y_e
            #y_e_mask = torch.cat((ones, y_e_mask[:, 0:o.O-1]), dim=1)
            #y_e_mask = perm_mat_inv.bmm(y_e_mask)  # N * O * dim_y_e
            y_e_mask = y_e.round() # N * O * dim_y_e
            y_e_inv_mask = y_e.round().lt(0.5).type_as(y_e) # N * O * dim_y_e
            h_o = y_e_mask * (h_o - h_o_prev) + h_o_prev  # N * O * dim_h_o
            # h_o = y_e_mask * h_o  # N * O * dim_h_o
            y_e = y_e_mask * y_e  # N * O * dim_y_e
            y_p = y_e_mask * y_p  # N * O * dim_y_p
            Y_a = y_e_mask.view(-1, o.O, o.dim_y_e, 1, 1) * Y_a  # N * O * D * h * w
            global_agent_pos = y_e_mask * pos.unsqueeze(1).expand(o.N, o.O, 2)
            global_tracker_offset = y_p[:, :, -2:] * torch.Tensor([o.H // 2, o.W // 2]).cuda()
            if phase == 'obs':
                memor = (y_e_inv_mask * memor) + global_agent_pos + global_tracker_offset

        if self.t == o.T - 1:
            print(y_e.data.view(-1, o.O)[0:1, 0:min(o.O, 10)])

        return memor, h_o, y_e, y_l, y_p, Y_s, Y_a
        
    def generate_outputs(self, h_o, pos):

        # Generate outputs
        # h_o = smd.CheckBP('h_o', 0)(h_o)
        o = self.o
        fcn_ip = h_o.view(-1, o.dim_h_o)
        a = self.fcn(fcn_ip) # NO * dim_y
        #a_e = a[:, 0:o.dim_y_e] # NO * dim_y_e
        a_l = a[:, 0:o.dim_y_l] # NO * dim_y_l
        a_p = a[:, o.dim_y_l:o.dim_y_l+o.dim_y_p] # NO * dim_y_p
        a_s = a[:, o.dim_y_l+o.dim_y_p:o.dim_y_l+o.dim_y_p+o.dim_Y_s] # NO * dim_Y_s
        a_a = a[:, o.dim_y_l+o.dim_y_p+o.dim_Y_s:o.dim_y_l+o.dim_y_p+o.dim_Y_s+o.dim_Y_a] # NO * dim_Y_aa

        #print(a.shape, a_l.shape, a_p.shape, a_s.shape, a_a.shape)
        # y_e
        #a_e = smd.CheckBP('a_e', 0)(a_e)
        #y_e = a_e.tanh().abs().view(o.N, o.O, o.dim_y_e)
        #if phase == 'obs':
        #    for j in range(o.N):
        #        mem[j][tuple(pos)] = y_e[j]
        #else:
        #    for j in range(o.N):
        #        y_e[j] = mem[j][tuple(pos)] if tuple(pos) in mem[j] else torch.rand_like(y_e[j])
        #p = torch.cat((a_e.view(-1, o.dim_y_e), y_e_prev.view(-1, o.dim_y_e)), 1)
        #y_e = self.linear_h_to_conf(p).tanh().abs().view(o.N, o.O, o.dim_y_e) # N * O * dim_y_e
        pos = pos / 60
        inp = pos.unsqueeze(1).expand(o.N, o.O, 2).contiguous().view(-1, 2)
        y_e = self.linear_pos_to_conf(inp).tanh().abs().view(o.N, o.O, o.dim_y_e)


        # y_l
        # a_l = smd.CheckBP('a_l', 0)(a_l)
        y_l = self.softmax(a_l)
        smd.norm_grad(y_l, 10)
        y_l = self.st_gumbel_softmax(y_l)
        y_l = y_l.view(-1, o.O, o.dim_y_l) # N * O * dim_y_l

        # y_p
        # a_p = smd.CheckBP('a_p', 0)(a_p)
        y_p = a_p.tanh()
        y_p = y_p.view(-1, o.O, o.dim_y_p) # N * O * dim_y_p

        # Y_s
        # a_s = smd.CheckBP('a_s', 0)(a_s)
        Y_s = a_s.sigmoid()
        Y_s = self.st_gumbel_sigmoid(Y_s)
        Y_s = Y_s.view(-1, o.O, 1, o.h, o.w) # N * O * 1 * h * w

        # Y_a
        # a_a = smd.CheckBP('a_a', 0)(a_a)
        Y_a = a_a.sigmoid()
        Y_a = Y_a.view(-1, o.O, o.D, o.h, o.w) # N * O * D * h * w

        return y_e, y_l, y_p, Y_s, Y_a


class NTMCell(nn.Module):

    def __init__(self, o):
        super(NTMCell, self).__init__()
        self.o = o
        self.linear_k = nn.Linear(o.dim_h_o, o.dim_C2_2)
        self.linear_b = nn.Linear(o.dim_h_o, 1)
        self.linear_e = nn.Linear(o.dim_h_o, o.dim_C2_2)
        self.linear_v = nn.Linear(o.dim_h_o, o.dim_C2_2)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)
        self.rnn_cell = nn.GRUCell(o.dim_C2_2, o.dim_h_o)
        self.ha = int(round(np.sqrt(o.dim_C2_1*o.H/o.W)))
        self.wa = int(round(np.sqrt(o.dim_C2_1*o.W/o.H)))
        self.att = torch.Tensor(o.O, self.ha, self.wa).cuda()
        self.mem = torch.Tensor(o.O, self.ha, self.wa).cuda()
        self.i = 0 # object id
        self.t = 0 # time
        self.n = 0 # sample id

    def forward(self, h_o_prev, C, pos, phase):
        """
        h_o_prev: N * dim_h_o
        C: N * C2_1 * C2_2
        """
        o = self.o
        n = self.n

        if o.v > 0:
            if self.i == 0:
                self.att.fill_(0.5)
                self.mem.fill_(0.5)
            self.mem[self.i].copy_(C.data[n].mean(1).view(self.ha, self.wa))

        # Addressing key
        k = self.linear_k(h_o_prev) # N * C2_2
        k_expand = k.unsqueeze(1).expand_as(C) # N * C2_1 * C2_2
        # Key strength, which equals to beta_pre.exp().log1p() + 1 but avoids 'inf' caused by exp()
        beta_pre = self.linear_b(h_o_prev)
        beta_pos = beta_pre.clamp(min=0)
        beta_neg = beta_pre.clamp(max=0)
        beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2)) # N * 1
        # Weighting
        C_cos = smd.Identity()(C)
        smd.norm_grad(C_cos, 1)
        s = self.cosine_similarity(C_cos, k_expand).view(-1, o.dim_C2_1) # N * C2_1
        w = self.softmax(s * beta) # N * C2_1

        # Read vector
        w1 = w.unsqueeze(1) # N * 1 * C2_1
        smd.norm_grad(w1, 1)
        r = w1.bmm(C).squeeze(1) # N * C2_2

        # RNN
        h_o = self.rnn_cell(r, h_o_prev)
        
        if "no_mem" not in o.exp_config:
            # Erase vector
            e = self.linear_e(h_o).sigmoid().unsqueeze(1) # N * 1 * C2_2
            # Write vector
            v = self.linear_v(h_o).unsqueeze(1) # N * 1 * C2_2
            # Update memory
            w2 = w.unsqueeze(2) # N * C2_1 * 1
            C = C * (1 - w2.bmm(e)) + w2.bmm(v) # N * C2_1 * C2_2

        if o.v > 0:
            self.att[self.i].copy_(w.data[n].view(self.ha, self.wa))

        return h_o, C, k, r
