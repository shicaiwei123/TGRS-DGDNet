import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# 训练参数

parser = ArgumentParser()

parser.add_argument('--train_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_warmup', type=bool, default=True)
parser.add_argument('--total_epoch', type=int, default=5)
parser.add_argument('--lr_decrease', type=str, default='cos', help='the methods of learning rate decay  ')
parser.add_argument('--mixup', type=bool, default=False, help='using mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=15)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--p', default=[0, 0], help='para for modality dropout')

# parser.add_argument('modal', type=str, default='multi')
parser.add_argument('--T', type=float, default=2.0, help='temperature for ST')

parser.add_argument('--model_name', type=str, default='mnist_cnn_best')
parser.add_argument('--log_name', type=str, default='.csv')
parser.add_argument('--backbone', type=str, default='couple_cross_fc')
parser.add_argument('--method', type=str, default='mad')
parser.add_argument('--kd_mode', type=str, default='st', help='mode of kd, which can be:'
                                                              'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                              'sp/sobolev/cc/lwm/irg/vid/ofd/afd')

parser.add_argument('--embemdding_dim', type=int, default=512)
parser.add_argument('--unimodal_center', type=float, default=1.0)
parser.add_argument('--dataset',type=str,default='huston2013')
parser.add_argument('--optim', type=str, default='sgd')

parser.add_argument('--intra', type=int, default=0)
parser.add_argument('--patch_size', type=int, default=7)
parser.add_argument('--lambda_center_loss', type=float, default=0)
parser.add_argument('--margin', type=int, default=1)

parser.add_argument('data_root', type=str,
                    default='/home/bigspace/shicaiwei/remote_sensing/houston2013')
parser.add_argument('pair_modalities', type=str, default='hsi+lidar')
parser.add_argument('drop_mode', type=str, default='average')
parser.add_argument('gpu', type=int, default=0)
parser.add_argument('version', type=int, default=0)
parser.add_argument('--lambda_kd_feature', type=float, default=60.0, help='trade-off parameter for kd loss')


args = parser.parse_args()
args.pair_modalities = args.pair_modalities.split('+')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if len(args.pair_modalities) == 3:
    args.name = args.dataset+ "_" + args.pair_modalities[0] + "_" + args.pair_modalities[1] + "_" + args.pair_modalities[
        2] + "_" + args.backbone + "_version_" + str(
        args.version) + "_unimodal_center_" + str(args.unimodal_center)
else:
    args.name = args.dataset+ "_" + args.pair_modalities[0] + "_" + args.pair_modalities[
        1] + "_" + args.backbone + "_version_" + str(
        args.version)+ "_unimodal_center_" + str(args.unimodal_center)
