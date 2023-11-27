import torch
import torch.nn as nn
import math
from math import sqrt


class ChannelAttention(nn.Module):
    '''
    通道注意力模块
    '''

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    '''
    空间注意力模块
    '''

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Flatten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x


import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, input_dims):
        super(Discriminator, self).__init__()
        self.args = args
        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            # nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class SPP(nn.Module):
    def __init__(self, merge='avg'):
        super(SPP, self).__init__()
        if merge == 'max':
            self.pooling_2x2 = nn.AdaptiveMaxPool2d((2, 2))
            self.pooling_1x1 = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
            self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_normal = x
        feature_list = []
        x_2x2 = self.pooling_2x2(x_normal)
        x_1x1 = self.pooling_1x1(x_normal)

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)  # B X C X feature_num

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        x_feature = torch.cat((x_2x2_flatten, x_1x1_flatten), dim=2)

        return x_feature


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        print(m.weight)
    else:
        print('error')


class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize
        with torch.no_grad():
            self.fc_squeeze.apply(init_weights)
            self.fc_visual.apply(init_weights)
            self.fc_skeleton.apply(init_weights)

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            # print(tensor.shape)
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out * skeleton * sk_out


class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, q, x):
        # x: batch, n, dim_in

        q = self.linear_q(q)  # batch, n, dim_k
        k = self.linear_k(q)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        k = torch.transpose(k, 1, 2)

        dist = torch.bmm(q, k)  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att
