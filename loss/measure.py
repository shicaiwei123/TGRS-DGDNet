import torch
import torch.nn as nn

import torch
import contextlib


def rsvd(input, rank):
    """
    Randomized SVD torch function
    Extremely fast computation of the truncated Singular Value Decomposition, using
    randomized algorithms as described in Halko et al. 'finding structure with randomness
    usage :
    Parameters:
    -----------
    * input : Tensor (2D matrix) whose SVD we want
    * rank : (int) number of components to keep
    Returns:
    * (u,s,v) : tuple, classical output as the builtin torch svd function
    """
    assert len(input.shape) == 2, "input tensor must be 2D"
    (m, n) = input.shape
    p = torch.min(torch.tensor([2 * rank, n]))
    x = torch.randn(n, p, device=input.device)
    y = torch.matmul(input, x)

    # get an orthonormal basis for y
    uy, sy, _ = torch.svd(y)
    rcond = torch.finfo(input.dtype).eps * m
    tol = sy.max() * rcond
    num = torch.sum(sy > tol)
    W1 = uy[:, :num]

    B = torch.matmul(W1.T, input)
    W2, s, v = torch.svd(B)
    u = torch.matmul(W1, W2)
    k = torch.min(torch.tensor([rank, u.shape[1]]))
    return (u[:, :k], s[:k], v[:, :k])


class PA_Measure(nn.Module):
    def __init__(self):
        super(PA_Measure, self).__init__()

    def forward(self, x, y):
        y = y.t()

        ##implement with svd
        # t=torch.mm(x,y)
        # print(t.shape)
        # u, s, v = rsvd(t,x.shape[0])
        # s_sum=torch.sum(s)
        # s_sum=s_sum.cuda()

        ## implement with defination
        x_factors = torch.sqrt(torch.sum(x * x, 1))
        y_factors = torch.sqrt(torch.sum(y * y, 0))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.mm(x_factors.unsqueeze(1), y_factors.unsqueeze(0))
        G_s = torch.mm(x, y)

        G_s = (G_s / (fm_s_normal_factors+1e-32))
        # G_s = torch.max(G_s,0)[0]
        # print(G_s)

        s_sum = torch.sum(G_s)/(x.shape[0]**2)

        return s_sum
