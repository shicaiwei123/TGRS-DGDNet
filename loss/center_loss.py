import torch
import torch.nn as nn
from torch.autograd.function import Function
from loss.kd.st import SoftTarget
import numpy as np

kl_loss = SoftTarget(T=2)


# class CenterLoss(nn.Module):
#     def __init__(self, num_classes, feat_dim, size_average=True):
#         super(CenterLoss, self).__init__()
#         self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
#         self.centerlossfunc = CenterlossFunc.apply
#         self.feat_dim = feat_dim
#         self.size_average = size_average
#
#     def forward(self, feat, label):
#         batch_size = feat.size(0)
#         feat = feat.view(batch_size, -1)
#         # To check the dim of centers and features
#         if feat.size(1) != self.feat_dim:
#             raise ValueError("Center's dim: {0} should be equal to input feature's \
#                             dim: {1}".format(self.feat_dim, feat.size(1)))
#         batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
#         loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
#         return loss
#
#
# class CenterlossFunc(Function):
#     @staticmethod
#     def forward(ctx, feature, label, centers, batch_size):
#         centers_batch = centers.index_select(0, label.long())
#         diff=torch.sum(torch.sqrt((feature - centers_batch).pow(2)),dim=1)
#
#         k = torch.min(torch.tensor(5.0) / diff, torch.ones(feature.shape[0]).cuda())
#         # # print(diff,k)
#         # # print(k)
#         k=torch.unsqueeze(k,dim=1)
#         # # print(k.shape,feature.shape)
#         centers_batch = centers_batch + k*(feature - centers_batch)
#
#         ctx.save_for_backward(feature, label, centers, batch_size,centers_batch)
#
#
#         return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size
#         # loss = kl_loss(feature, centers_batch)
#         # return loss
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         feature, label, centers, batch_size,centers_batch = ctx.saved_tensors
#         # centers_batch = centers.index_select(0, label.long())
#
#         diff = centers_batch - feature
#         # init every iteration
#         counts = centers.new_ones(centers.size(0))
#         ones = centers.new_ones(label.size(0))
#         grad_centers = centers.new_zeros(centers.size())
#
#         counts = counts.scatter_add_(0, label.long(), ones)
#         grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
#         grad_centers = grad_centers / counts.view(-1, 1)
#         return - grad_output * diff / batch_size, None, grad_centers / batch_size, None
#
#
# def main(test_cuda=False):
#     print('-' * 80)
#     device = torch.device("cuda" if test_cuda else "cpu")
#     ct = CenterLoss(10, 2, size_average=True).to(device)
#     y = torch.Tensor([0, 0, 2, 1]).to(device)
#     feat = torch.zeros(4, 2).to(device).requires_grad_()
#     print(list(ct.parameters()))
#     print(ct.centers.grad)
#     out = ct(y, feat)
#     print(out.item())
#     out.backward()
#     print(ct.centers.grad)
#     print(feat.grad)
#
#
# if __name__ == '__main__':
#     torch.manual_seed(999)
#     main(test_cuda=False)
#     if torch.cuda.is_available():
#         main(test_cuda=True)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        centers = self.centers.index_select(0, labels.long())

        distmat = (x - centers).pow(2).sum(dim=1)
        batch_size = x.shape[0]

        # classes = torch.arange(self.num_classes).long()
        # if self.use_gpu: classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        for i in range(self.num_classes):


        return loss
