import os
import os.path as path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.feature_extractor import FeatureExtractor
from modules.tracker_array import TrackerArray
from modules.renderer import Renderer
from modules.loss_calculator import LossCalculator
import modules.submodules as smd
import modules.utils as utils

def coords(H, W):
        dh, dw = 2/(H-1), 2/(W-1)
        coor_y = torch.arange(-1, 1+1e-5, dh).cuda().view(H, 1)
        coor_x = torch.arange(-1, 1+1e-5, dw).cuda().view(1, W)
        return coor_x.expand(H, W), coor_y.expand(H, W)

def local_coords(o, H, W):        
        coor_x, coor_y = coords(H, W)
        coor_x = coor_x.view(1, 1, 1, H, W).expand(o.N, o.T, 1, H, W)
        coor_y = coor_y.view(1, 1, 1, H, W).expand(o.N, o.T, 1, H, W)
        coor = torch.cat((coor_y, coor_x), 2) # N * T * 2 * H * W
        return coor

def global_coords(o, H, W, path, h, w, border=20):
        coor_x, coor_y = coords(H + 2*border, W + 2*border)
        x, y = (path + torch.Tensor([border, border])).unbind(2)
        pos_x, pos_y = x - int(h // 2), y - int(w // 2)
        global_coords = torch.zeros((o.N, o.T, 2, h, w)).cuda()
        for n in range(o.N):
            for t in range(o.T):
                tx, ty = int(pos_x[n, t].item()), int(pos_y[n, t].item())
                global_coords[n, t, 0] = coor_y[tx: tx + h, ty: ty + w]
                global_coords[n, t, 1] = coor_x[tx: tx + h, ty: ty + w]
        return global_coords

class Net(nn.Module):


    def __init__(self, o):
        super(Net, self).__init__()
        self.o = o

        # Modules
        device_ids = np.arange(0, o.G).tolist()
        self.feature_extractor = nn.DataParallel(FeatureExtractor(o), device_ids)
        self.tracker_array = TrackerArray(o)
        self.renderer = nn.DataParallel(Renderer(o), device_ids)
        self.renderer_vis = Renderer(o)
        self.loss_calculator = nn.DataParallel(LossCalculator(o), device_ids)
        
        # States
        self.states = {}
        self.states['h_o_prev'] = torch.Tensor(o.N, o.O, o.dim_h_o).cuda()
        self.states['y_e_prev'] = torch.Tensor(o.N, o.O, o.dim_y_e).cuda()
        self.reset_states()

        self.n = 0


    def forward(self, X_seq, **kwargs):
        o = self.o
        if 'X_bg_seq' in kwargs.keys():
            Y_b_seq = kwargs['X_bg_seq']

        X_base_img = None
        data, path, actions, phase = None, None, None, None
        coords_info = local_coords(o, o.H, o.W)
        if 'X_base_img' in kwargs.keys():
            X_base_img = kwargs['X_base_img']
            data, path, actions, phase = [X_base_img[i] for i in range(4)]
            _, H, W = data.shape
            coords_info = global_coords(o, H, W, path, o.H, o.W, border=20)
            if(sum(actions[:, 0]) < 0):
                self.reset_states()
                self.memory = None

        # Extract features
        X_seq_cat = torch.cat((X_seq, Variable(coords_info.clone())), 2) # N * T * D+2 * H * W
        C_o_seq = self.feature_extractor(X_seq_cat) # N * T * M * R
        C_o_seq = smd.CheckBP('C_o_seq')(C_o_seq)

        # Update trackers
        h_o_prev, y_e_prev = self.load_states('h_o_prev', 'y_e_prev')
        if (phase == 'pred'):
            print('$$$$$$$$' * 5)
            if (self.memory is None):
                self.memory = h_o_prev
            h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq = self.tracker_array(self.memory, y_e_prev, C_o_seq, path, phase)
        else:
            self.memory = None
            h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq = self.tracker_array(h_o_prev, y_e_prev, C_o_seq, path, phase) # N * T * O * ...
        if o.r == 1:
            self.save_states(h_o_prev=h_o_seq, y_e_prev=y_e_seq)

        if(phase == 'obs'):
            return None

        # Render the image
        ka = {}
        if o.bg == 1:
            ka['Y_b'] = Y_b_seq
        X_r_seq, area = self.renderer(y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq, **ka) # N * T * D * H * W
        if(o.train == 0):
            area = area.unsqueeze(0)

        # Calculate the loss
        ka = {'y_e': y_e_seq}
        if o.bg == 0:
            ka['Y_a'] = Y_a_seq
        else:
            ka['Y_b'] = Y_b_seq
            if o.metric == 0:
                ka['y_p'] = y_p_seq
        # print(X_r_seq.shape, X_seq.shape, area.shape, ka['y_e'].shape, ka['Y_a'].shape)
        loss = self.loss_calculator(X_r_seq, X_seq, area, **ka)
        loss = loss.sum() / (o.N * o.T)

        # Visualize
        if o.v  > 0:
            ka = {'X': X_seq, 'X_r': X_r_seq, 'y_e': y_e_seq, 'y_l': y_l_seq, 'y_p': y_p_seq, 
                  'Y_s': Y_s_seq, 'Y_a': Y_a_seq}
            if X_base_img:
                ka['X_base_img'] = X_base_img
            if o.bg == 1:
                ka['Y_b'] = Y_b_seq
                if o.metric == 1:
                    ka['X_org'] = kwargs['X_org_seq']
            self.visualize(**ka)

        return loss
        

    def visualize(self, **kwargs):
        o = self.o
        im_scale = 1
        obj_scale = 1
        n = 0#self.n
        self.n = (self.n + 1) % o.N
        H, W = o.H * im_scale, o.W * im_scale
        h, w = o.h * obj_scale, o.w * obj_scale
        if o.v == 2:
            save_dir = os.path.join(o.pic_dir, str(n))
        show_dict = {'input': kwargs['X'], 'input_recon': kwargs['X_r']}
        if o.bg == 1:
            if o.metric == 1:
                show_dict['org'] = kwargs['X_org']
        att_hor = 1
        if att_hor == 1:
            att = self.tracker_array.ntm.att.permute(0, 2, 1, 3).contiguous().view(o.T, self.tracker_array.ntm.ntm_cell.ha, -1)
            mem = self.tracker_array.ntm.mem.permute(0, 2, 1, 3).contiguous().view(o.T, self.tracker_array.ntm.ntm_cell.ha, -1)
        else:
            att = self.tracker_array.ntm.att.view(o.T, -1, self.tracker_array.ntm.ntm_cell.wa)
            mem = self.tracker_array.ntm.mem.view(o.T, -1, self.tracker_array.ntm.ntm_cell.wa)
        mem_max = 4#mem.max()
        mem_min = 0#mem.min()
        # print(mem_min, mem_max)
        mem = (mem - mem_min) / (mem_max - mem_min + 1e-20)

        for t in range(0, o.T):
            tao = o.batch_id * o.T + t
            
            # Images
            for img_kw, img_arg in show_dict.items():
                img = img_arg.data[n, t].permute(1, 2, 0).clamp(0, 1)
                if o.v == 1:
                    utils.imshow(img, H, W, img_kw)
                else:
                    if img_kw == 'input' or img_kw == 'org':
                        utils.mkdir(path.join(save_dir, img_kw))
                        utils.imwrite(img, path.join(save_dir, img_kw, "%05d" % (tao)))

            # Enforce to show object bounding boxes on the image
            if o.metric == 1 and "no_mem" not in o.exp_config:
                y_e = Variable(kwargs['y_e'].data[n:n+1].clone().round())
            else:
                y_e = Variable(kwargs['y_e'].data[n:n+1].clone())
            y_e_vis = y_e#Variable(kwargs['y_e'].data[n:n+1].clone().fill_(1))
            y_l = Variable(kwargs['y_l'].data[n:n+1].clone())
            y_p = Variable(kwargs['y_p'].data[n:n+1].clone())
            Y_s = Variable(kwargs['Y_s'].data[n:n+1].clone()) # 1 * T * O * 1 * h * w
            Y_a = Variable(kwargs['Y_a'].data[n:n+1].clone()) # 1 * T * O * D * h * w
            Y_s.data[:, :, :, :, 0, :].fill_(1)
            Y_s.data[:, :, :, :, -1, :].fill_(1)
            Y_s.data[:, :, :, :, :, 0].fill_(1)
            Y_s.data[:, :, :, :, :, -1].fill_(1)
            Y_a.data[:, :, :, :, 0, :].fill_(1)
            Y_a.data[:, :, :, :, -1, :].fill_(1)
            Y_a.data[:, :, :, :, :, 0].fill_(1)
            Y_a.data[:, :, :, :, :, -1].fill_(1)
            if o.bg == 0:
                X_r_vis, _a = self.renderer_vis(y_e_vis, y_l, y_p, Y_s, Y_a) # 1 * T * D * H * W
            else:
                Y_b = Variable(kwargs['Y_b'].data[n:n+1].clone())
                X_r_vis, _a = self.renderer_vis(y_e_vis, y_l, y_p, Y_s, Y_a, Y_b=Y_b) # 1 * T * D * H * W
            img = X_r_vis.data[0, t, 0:o.D].permute(1, 2, 0).clamp(0, 1)
            if o.v == 1:
                utils.imshow(img, H, W, 'X_r_vis')
            else:
                utils.mkdir(path.join(save_dir, 'X_r_vis'))
                utils.imwrite(img, path.join(save_dir, 'X_r_vis', "%05d" % (tao)))

            # Objects
            y_e, Y_s, Y_a = y_e.data[0, t], Y_s.data[0, t], Y_a.data[0, t] # O * D * h * w
            if o.task == 'mnist':
                Y_o = (y_e.view(-1, 1, 1, 1) * Y_a).permute(2, 0, 3, 1).contiguous().view(o.h, o.O*o.w, o.D)
                Y_o_v = (y_e.view(-1, 1, 1, 1) * Y_a).permute(0, 2, 3, 1).contiguous().view(o.O*o.h, o.w, o.D)
            else:
                Y_o = (y_e.view(-1, 1, 1, 1) * Y_s * Y_a).permute(2, 0, 3, 1).contiguous().view(o.h, o.O*o.w, o.D)
                Y_o_v = (y_e.view(-1, 1, 1, 1) * Y_a * Y_a).permute(0, 2, 3, 1).contiguous().view(o.O*o.h, o.w, o.D)
            if o.v == 1:
                utils.imshow(Y_o, h, w * o.O, 'Y_o', 1)
            else:
                utils.mkdir(path.join(save_dir, 'Y_o'))
                utils.imwrite(Y_o, path.join(save_dir, 'Y_o', "%05d" % (tao)))
            # utils.imshow(Y_o_v, h, w * o.O, 'Y_o_v')
            # utils.mkdir(path.join(save_dir, 'Y_o_v'))
            # utils.imwrite(Y_o_v, path.join(save_dir, 'Y_o_v', "%05d" % (tao)))
            
            # Attention and memory
            if o.task != 'duke':
                cmap='hot'
                att_c = utils.heatmap(att[t], cmap)
                mem_c = utils.heatmap(mem[t], cmap)
                if o.v == 1:
                    sa = 10
                    utils.imshow(att_c, att_c.size(0)*sa, att_c.size(1)*sa, 'att')
                    utils.imshow(mem_c, mem_c.size(0)*sa, mem_c.size(1)*sa, 'mem')
                else:
                    utils.mkdir(os.path.join(save_dir, 'att'))
                    utils.mkdir(os.path.join(save_dir, 'mem'))
                    utils.imwrite(att_c, os.path.join(save_dir, 'att', "%05d" % (tao)))
                    utils.imwrite(mem_c, os.path.join(save_dir, 'mem', "%05d" % (tao)))

        if o.v == 2:
            if 'X_base_img' in kwargs.keys():
                utils.mkdir(os.path.join(save_dir, 'base'))
                data, path, actions = kwargs['X_base_img']
                torch.save((data[n], path[n], actions[n]), os.path.join(save_dir, 'base', str(o.batch_id) + '.pt')) 
    def reset_states(self):
        for state in self.states.values():
            state.fill_(0)


    def load_states(self, *args):
        states = [Variable(self.states[arg].clone()) for arg in args]
        return states if len(states) > 1 else states[0]


    def save_states(self, **kwargs):
        for kw, arg in kwargs.items():
            self.states[kw].copy_(arg.data[:, -1])
