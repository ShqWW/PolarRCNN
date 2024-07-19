import torch 
from torch import nn
import math

class FixAnchorHead(nn.Module):
    def __init__(self, cfg=None):
        super(FixAnchorHead, self).__init__()
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.center_w, self.center_h = cfg.center_w, cfg.center_h
        self.num_priors = cfg.num_priors
        weight = torch.load('./clr_anchor.pkl', map_location=torch.device('cpu'))
        embeddings = weight['heads.prior_embeddings.weight'].detach().cpu()
        init_embeddings = self.get_init_embeddings(embeddings)
        self.register_buffer(name='anchor_embeddings', tensor=init_embeddings) #get embeddings (freezen during training)
        self.register_buffer(name='prior_id', tensor=torch.linspace(0, self.num_priors-1, self.num_priors, dtype=torch.int32))

    def forward(self, x):
        batch_size = x[0].shape[0]
        anchor_embeddings = self.anchor_embeddings.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        anchor_id = self.prior_id.unsqueeze(0).repeat(batch_size, 1)
        pred_dict = {'anchor_embeddings': anchor_embeddings, 'anchor_id': anchor_id}
        return pred_dict

    def get_init_embeddings(self, embeddings):
        embeddings = embeddings.detach().clone()
        ys, xs, thetas = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
        xs = self.img_w*xs
        ys = self.img_h*(1-ys)
        rs = (xs-self.center_w)*torch.sin(thetas*math.pi) + (ys-self.center_h)*torch.cos(thetas*math.pi)
        thetas = thetas - 0.5
        rs = rs/self.img_w
        init_embeddings = torch.cat((thetas.unsqueeze(-1), rs.unsqueeze(-1)), dim=-1)
        return init_embeddings


# class FixAnchorHead(nn.Module):
#     def __init__(self, cfg=None):
#         super(FixAnchorHead, self).__init__()
#         self.img_w, self.img_h = cfg.img_w, cfg.img_h
#         self.center_w, self.center_h = cfg.center_w, cfg.center_h
#         self.num_priors = 192
#         init_anchor_xyt = self.get_init_mebeddings()
#         rts = self.get_rho_theta(init_anchor_xyt)
#         # self.anchor_embeddings = nn.Embedding(self.num_priors, 2)
#         # self.anchor_embeddings.weight.data = rts
#         self.register_buffer(name='anchor_embeddings', tensor=rts) #get embeddings (freezen during 
#         print('hand2')

#     def forward(self, x):
#         batch_size = x[0].shape[0]
#         anchor_embeddings = self.anchor_embeddings.clone().unsqueeze(0).repeat(batch_size, 1, 1)
#         pred_dict = {'anchor_embeddings': anchor_embeddings}
#         return pred_dict
        
#     def get_init_mebeddings(self):
#         init_anchor_xyt = torch.zeros(self.num_priors, 3)
#         bottom_priors_nums = self.num_priors * 3 // 4
#         left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8
#         strip_size = 0.5 / (left_priors_nums // 2 - 1)
#         bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)

#         for i in range(left_priors_nums):
#             init_anchor_xyt[i, 0] = (i // 2) * strip_size 
#             init_anchor_xyt[i, 1] = 0
#             init_anchor_xyt[i, 2] = 0.16 if i % 2 == 0 else 0.32

#         for i in range(left_priors_nums, left_priors_nums + bottom_priors_nums):
#             init_anchor_xyt[i, 0] = 0
#             init_anchor_xyt[i, 1] = ((i - left_priors_nums) // 4 + 1) * bottom_strip_size
#             init_anchor_xyt[i, 2] =  0.2 * (i % 4 + 1)

#         for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
#             init_anchor_xyt[i, 0] = ((i - left_priors_nums - bottom_priors_nums) // 2) * strip_size
#             init_anchor_xyt[i, 1] = 1
#             init_anchor_xyt[i, 2] = 0.68 if i % 2 == 0 else 0.84
#         return init_anchor_xyt

#     def get_rho_theta(self, init_anchor_xyt):
#         ys, xs, thetas = init_anchor_xyt[..., 0], init_anchor_xyt[..., 1], init_anchor_xyt[..., 2]
#         xs = self.img_w*xs
#         ys = self.img_h*(1-ys)
#         rs = (xs-self.center_w)*torch.sin(thetas*math.pi) + (ys-self.center_h)*torch.cos(thetas*math.pi)
#         thetas = thetas - 0.5
#         rs = rs/self.img_w
#         rts = torch.cat((thetas.unsqueeze(-1), rs.unsqueeze(-1)), dim=-1)
#         return rts


class LearnedAnchorHead(nn.Module):
    def __init__(self, cfg=None):
        super(LearnedAnchorHead, self).__init__()
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.center_w, self.center_h = cfg.center_w, cfg.center_h
        self.num_priors = cfg.num_priors
        init_anchor_xyt = self.get_init_mebeddings()
        rts = self.get_rho_theta(init_anchor_xyt)
        self.anchor_embeddings = nn.Embedding(self.num_priors, 2)
        self.anchor_embeddings.weight.data = rts
        
    def get_init_mebeddings(self):
        init_anchor_xyt = torch.zeros(self.num_priors, 3)
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)

        for i in range(left_priors_nums):
            init_anchor_xyt[i, 0] = (i // 2) * strip_size 
            init_anchor_xyt[i, 1] = 0
            init_anchor_xyt[i, 2] = 0.16 if i % 2 == 0 else 0.32

        for i in range(left_priors_nums, left_priors_nums + bottom_priors_nums):
            init_anchor_xyt[i, 0] = 0
            init_anchor_xyt[i, 1] = ((i - left_priors_nums) // 4 + 1) * bottom_strip_size
            init_anchor_xyt[i, 2] =  0.2 * (i % 4 + 1)

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            init_anchor_xyt[i, 0] = ((i - left_priors_nums - bottom_priors_nums) // 2) * strip_size
            init_anchor_xyt[i, 1] = 1
            init_anchor_xyt[i, 2] = 0.68 if i % 2 == 0 else 0.84
        return init_anchor_xyt

    def get_rho_theta(self, init_anchor_xyt):
        ys, xs, thetas = init_anchor_xyt[..., 0], init_anchor_xyt[..., 1], init_anchor_xyt[..., 2]
        xs = self.img_w*xs
        ys = self.img_h*(1-ys)
        rs = (xs-self.center_w)*torch.sin(thetas*math.pi) + (ys-self.center_h)*torch.cos(thetas*math.pi)
        thetas = thetas - 0.5
        rs = rs/self.img_w
        rts = torch.cat((thetas.unsqueeze(-1), rs.unsqueeze(-1)), dim=-1)
        return rts




