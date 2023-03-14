import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4, ResNet


class ESPT(nn.Module):
    def __init__(self, way, support_shot, query_shot, alpha, seed, resnet=False, is_pretraining=False, num_cat=None):
        super().__init__()
        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()
        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.feature_map_H = 5
        self.feature_map_w = 5
        self.resolution = 25
        self.dim = num_channel

        self.way = way
        self.support_shot = support_shot
        self.query_shot = query_shot
        self.resnet = resnet
        self.rnd = np.random.RandomState(seed)
        self.is_pretraining = is_pretraining

        if self.is_pretraining:
            self.num_cat = num_cat
            self.mat_cat = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.dim), requires_grad=True)
            self.scale = nn.Parameter(torch.FloatTensor([10.0]), requires_grad=True)
            self.alpha_pretrain = nn.Parameter(torch.FloatTensor([alpha]), requires_grad=False)
        else:
            self.scale = nn.Parameter(torch.FloatTensor([10.0]), requires_grad=True)
            self.alpha = nn.Parameter(torch.FloatTensor([alpha]), requires_grad=False)

    def get_feature_map(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        if self.resnet:
            feature_map = feature_map / np.sqrt(640)

        return feature_map.view(batch_size, self.dim, -1).permute(0, 2, 1).contiguous()

    @staticmethod
    def get_recon_dist(support, query, alpha, woodbury=False):
        reg = support.size(1) / support.size(2)
        lam = reg * alpha
        st = support.permute(0, 2, 1).contiguous()

        if woodbury:
            sts = st.matmul(support)
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()
            weight = query.matmul(m_inv).matmul(st)
        else:
            sst = support.matmul(st)
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse()
            weight = query.matmul(st).matmul(m_inv)

        Q_bar = weight.matmul(support)
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0).contiguous()

        return dist, weight

    def get_neg_l2_dist(self, feature_map, way, support_shot, query_shot, return_weight=False):
        support = feature_map[:way * support_shot].view(way, support_shot * self.resolution, self.dim)
        query = feature_map[way * support_shot:].view(way * query_shot * self.resolution, self.dim)

        if self.is_pretraining:
            alpha = self.alpha_pretrain
        else:
            alpha = self.alpha

        recon_dist, weight = self.get_recon_dist(query=query, support=support, alpha=alpha)
        neg_l2_dist = recon_dist.neg().view(way * query_shot, self.resolution, way).mean(1)

        if return_weight:
            return neg_l2_dist, weight
        else:
            return neg_l2_dist

    def pretrain_forward(self, inp):
        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)
        feature_map = feature_map.view(batch_size * self.resolution, self.dim)

        recon_dist, weight = self.get_recon_dist(query=feature_map, support=self.mat_cat, alpha=self.alpha_pretrain)
        neg_l2_dist = recon_dist.neg().view(batch_size, self.resolution, self.num_cat).mean(1)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction

    def get_flip_feature_map(self, inp, flip):
        flip_input = torch.flip(inp, [flip]).contiguous()
        ori_and_flip_inp = torch.cat((inp, flip_input), dim=0)
        ori_and_flip_feature_map = self.get_feature_map(ori_and_flip_inp)
        ori_feature_map = ori_and_flip_feature_map[:inp.size(0)]
        flip_feature_map = ori_and_flip_feature_map[inp.size(0):]

        flip_feature_map = flip_feature_map.view(inp.size(0), self.feature_map_H, self.feature_map_w, self.dim)
        flip_feature_map = torch.flip(flip_feature_map, [flip + 1]).contiguous()
        flip_feature_map = flip_feature_map.view(inp.size(0), self.resolution, self.dim)

        return ori_feature_map, flip_feature_map

    def get_rot_feature_map(self, inp, rotation):
        rot_input = torch.rot90(inp, rotation, [2, 3]).contiguous()
        ori_and_rot_inp = torch.cat((inp, rot_input), dim=0)
        ori_and_rot_feature_map = self.get_feature_map(ori_and_rot_inp)
        ori_feature_map = ori_and_rot_feature_map[:inp.size(0)]
        rot_feature_map = ori_and_rot_feature_map[inp.size(0):]

        rot_feature_map = rot_feature_map.view(inp.size(0), self.feature_map_H, self.feature_map_w, self.dim)
        rot_feature_map = torch.rot90(rot_feature_map, -rotation, [1, 2]).contiguous()
        rot_feature_map = rot_feature_map.view(inp.size(0), self.resolution, self.dim)

        return ori_feature_map, rot_feature_map

    # def LR_forward(self, inp):
    #     feature_map = self.get_feature_map(inp)
    #
    #     neg_l2_dist, weight = self.get_neg_l2_dist(
    #         feature_map=feature_map, way=self.way, support_shot=self.support_shot,
    #         query_shot=self.query_shot, return_weight=True
    #     )
    #
    #     logits = neg_l2_dist * self.scale
    #     log_prediction = F.log_softmax(logits, dim=1)
    #
    #     return log_prediction

    def ESPT_forward(self, inp, transform_list, transform_type):
        transform = self.rnd.choice(transform_list)
        if transform_type == 'flip':  # flip_list in [1, 2]
            ori_feature_map, trans_feature_map = self.get_flip_feature_map(inp, transform)
        elif transform_type == 'rotation':  # rot_list in [1, 2, 3]
            ori_feature_map, trans_feature_map = self.get_rot_feature_map(inp, transform)
        else:
            ori_feature_map, trans_feature_map = None, None

        neg_l2_dist_ori, weight_ori = self.get_neg_l2_dist(
            feature_map=ori_feature_map, way=self.way, support_shot=self.support_shot,
            query_shot=self.query_shot, return_weight=True
        )

        neg_l2_dist_trans, weight_trans = self.get_neg_l2_dist(
            feature_map=trans_feature_map, way=self.way, support_shot=self.support_shot,
            query_shot=self.query_shot, return_weight=True
        )

        logits_ori = neg_l2_dist_ori * self.scale
        log_prediction_ori = F.log_softmax(logits_ori, dim=1)

        # Euclidean similarity
        query_size = self.way * self.query_shot
        dictionary_size = self.way * self.support_shot * self.resolution
        weight_ori = weight_ori.permute(1, 0, 2).contiguous()
        weight_ori = F.normalize(weight_ori, dim=2, p=2)
        weight_ori = weight_ori.view(query_size * self.resolution, dictionary_size)

        weight_trans = weight_trans.permute(1, 0, 2).contiguous()
        weight_trans = F.normalize(weight_trans, dim=2, p=2)
        weight_trans = weight_trans.view(query_size * self.resolution, dictionary_size)

        weight_similarity = (weight_ori.detach() - weight_trans).pow(2).sum(1).mean(0)

        return log_prediction_ori, weight_similarity

    def meta_test(self, inp, way, support_shot, query_shot):
        feature_map = self.get_feature_map(inp)
        neg_l2_dist = self.get_neg_l2_dist(
            feature_map=feature_map, way=way, support_shot=support_shot, query_shot=query_shot
        )
        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index
