'''
a template for train, you need to fix your own main function
'''

import sys
import os

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_General, HSI_Lidar_Couple_Cross_Drop
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.huston_multi_general import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_share, train_base_multi_share_unimodal_center
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

    args.log_name = args.name
    args.model_name = args.name
    # print(args.unimodal_center)
    if isinstance(args.p, str):
        args.p = eval(args.p)

    if 1 not in args.p:
        args.model_root = os.path.join(args.model_root, args.backbone, args.fusion_method,
                                       'drop' + '_unimodal_center_' + str(args.unimodal_center))
        args.log_root = os.path.join(args.log_root, args.backbone, args.fusion_method,
                                     'drop' + '_unimodal_center_' + str(args.unimodal_center))
    else:
        args.model_root = os.path.join(args.model_root, args.backbone, args.fusion_method,
                                       'full' + '_unimodal_center_' + str(args.unimodal_center))
        args.log_root = os.path.join(args.log_root, args.backbone, args.fusion_method,
                                     'full' + '_unimodal_center_' + str(args.unimodal_center))

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = HSI_Lidar_Couple_General(args, modality_1_channel, modality_2_channel)
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

    if args.unimodal_center:
        train_base_multi_share_unimodal_center(model=model, cost=criterion, optimizer=optimizer,
                                               train_loader=train_loader,
                                               test_loader=test_loader,
                                               args=args)
    else:
        train_base_multi_share(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                               test_loader=test_loader,
                               args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
