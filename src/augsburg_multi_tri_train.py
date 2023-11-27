'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_Share_TRI,HSI_Lidar_Couple_Cross_TRI
from src.augsburg_dataloader import augsburg_multi_dataloader_tri
from configuration.augsburg_multi_config import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_tri
from lib.processing_utils import get_file_list
import torch.optim as optim
import os
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
    train_loader = augsburg_multi_dataloader_tri(train=True, args=args)
    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)

    args.log_name = args.name
    args.model_name = args.name
    args.location='after'

    if isinstance(args.p, str):
        args.p = eval(args.p)

    if torch.sum(torch.tensor(args.p)) == 0:
        args.model_root = os.path.join(args.model_root,args.dataset,args.drop_mode,'share_unimodal_center_drop')
        args.log_root = os.path.join(args.log_root,args.dataset,args.drop_mode,'share_unimodal_center_drop')
        print(args.model_root)
    else:
        args.model_root=os.path.join(args.model_root,args.dataset,args.drop_mode,'share_unimodal_center_full')
        args.log_root = os.path.join(args.log_root, args.dataset, args.drop_mode, 'share_unimodal_center_full')

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]

    model = HSI_Lidar_Couple_Share_TRI(args, modality_1_channel, modality_2_channel, modality_3_channel)
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
    train_base_multi_tri(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                     test_loader=test_loader,
                     args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
