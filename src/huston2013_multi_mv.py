'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.surf_corr_repr import SURF_MV
from src.augsburg_dataloader import augsburg_multi_dataloader
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.multimdal_fusion_config import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_share
from lib.processing_utils import get_file_list
import torch.optim as optim

import cv2
import numpy as np
import datetime
import random


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main(args):
    train_loader = huston2013_multi_dataloader(train=True, args=args)
    test_loader = huston2013_multi_dataloader(train=False, args=args)
    args.location='after'
    args.log_name = args.name
    args.model_name = args.name
    args.drop_mode='average'

    if isinstance(args.p, str):
        args.p = eval(args.p)

    if 1 not  in args.p:
        args.model_root = args.model_root + "/mv_drop"
        args.log_root = args.log_root + "/mv_drop"
    else:
        args.model_root=args.model_root+"/mv_full"
        args.log_root=args.log_root+"/mv_full"

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = SURF_MV(args, modality_1_channel, modality_2_channel)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
    #                        weight_decay=args.weight_decay)

    args.retrain = False
    train_base_multi_share(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                           test_loader=test_loader,
                           args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
