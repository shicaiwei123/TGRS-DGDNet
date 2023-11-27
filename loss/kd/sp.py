from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys


class SP(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self, reduction='mean'):
        super(SP, self).__init__()
        self.reduction = reduction

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t, reduction=self.reduction)

        return loss



class DAD(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(DAD, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        fm_t = fm_t.view(fm_t.size(0), -1)

        fm_s = fm_s - torch.mean(fm_s, dim=1, keepdim=True)
        fm_t = fm_t - torch.mean(fm_t, dim=1, keepdim=True)

        # fm_s = PCA_svd(fm_s, 512)
        # fm_t = PCA_svd(fm_t, 512)

        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))
        fm_s_trans = fm_s.t()
        fm_s_trans_factors = torch.sqrt(torch.sum(fm_s_trans * fm_s_trans, 0))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.mm(fm_s_factors.unsqueeze(1), fm_s_trans_factors.unsqueeze(0))
        G_s = torch.mm(fm_s, fm_s.t())
        G_s = (G_s / (fm_s_normal_factors+1e-16))

        fm_t_factors = torch.sqrt(torch.sum(fm_t * fm_t, 1))
        fm_t_trans = fm_t.t()
        fm_t_trans_factors = torch.sqrt(torch.sum(fm_t_trans * fm_t_trans, 0))
        fm_t_normal_factors = torch.mm(fm_t_factors.unsqueeze(1), fm_t_trans_factors.unsqueeze(0))
        G_t = torch.mm(fm_t, fm_t.t())
        G_t = (G_t / (fm_t_normal_factors+1e-16))

        loss = F.mse_loss(G_s, G_t, reduction='none')

        return loss


class DGD(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(DGD, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_t = fm_t.view(fm_t.size(0), fm_s.size(1), -1)

        # fm_s = PCA_svd(fm_s, 512)
        # fm_t = PCA_svd(fm_t, 512)
        # print(fm_s.shape)
        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))
        fm_s_trans = fm_s.permute(0, 2, 1)
        fm_s_trans_factors = torch.sqrt(torch.sum(fm_s_trans * fm_s_trans, 2))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.bmm(fm_s_factors.unsqueeze(2),fm_s_trans_factors.unsqueeze(1))
        G_s = torch.bmm(fm_s_trans, fm_s)
        # G_s = (G_s / (fm_s_normal_factors + 1e-16))

        fm_t_factors = torch.sqrt(torch.sum(fm_t * fm_t, 1))
        fm_t_trans = fm_t.permute(0, 2, 1)
        fm_t_trans_factors = torch.sqrt(torch.sum(fm_t_trans * fm_t_trans, 2))
        fm_t_normal_factors = torch.bmm(fm_t_factors.unsqueeze(2),fm_t_trans_factors.unsqueeze(1))
        # print(fm_t_normal_factors.shape)
        G_t = torch.bmm(fm_t_trans, fm_t)
        # G_t = (G_t / (fm_t_normal_factors + 1e-16))

        loss = F.mse_loss(G_s, G_t, reduction='none')
        loss = torch.mean(loss, dim=(1, 2))
        # loss = torch.sum(loss, dim=1)

        return loss

