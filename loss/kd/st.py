from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss.kd.pkt import PKTCosSim
from loss.kd.sp import SP
from lib.model_arch_utils import SelfAttention


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T
        if torch.cuda.is_available():
            self.bce_loss = nn.BCELoss().cuda()
        else:
            self.bce_loss = nn.BCELoss()

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/ self.T, dim=1),
                        F.softmax(out_t/ self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss

class MultiSoftTarget(nn.Module):
    def __init__(self, T):
        super(MultiSoftTarget, self).__init__()
        self.T = T
        self.scale_selector = nn.Sequential(nn.Linear(5, 5), nn.Softmax())
        self.self_attention = SelfAttention(5, 5, 5)

    def forward(self, out_s_multi, out_t_multi, add_mode='avg'):

        loss_list = []
        for i in range(out_s_multi.shape[2]):
            out_s = torch.squeeze(out_s_multi[:, :, i])
            out_t = torch.squeeze(out_t_multi[:, :, i])

            loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                            F.softmax(out_t / self.T, dim=1),
                            reduction='batchmean') * self.T * self.T
            # print(loss)
            loss_list.append(loss)
            # print(loss)

        # print(loss_list)

        if add_mode=='sa':

            # loss_sum = torch.sum(self.scale_selector(weight) * loss_list)
            loss_list = torch.tensor(loss_list)
            loss_list=torch.unsqueeze(loss_list,0)
            loss_list = torch.unsqueeze(loss_list, 2)
            loss_list = loss_list.cuda()
            loss_sum = torch.sum(self.self_attention(loss_list,loss_list))

        else:
            loss_list = torch.tensor(loss_list)
            loss_list = loss_list.cuda()
            loss_sum = torch.sum(loss_list) / out_s_multi.shape[2]
        return loss_sum
