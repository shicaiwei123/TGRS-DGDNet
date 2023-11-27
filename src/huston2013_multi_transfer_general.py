'''
a template for train, you need to fix your own main function
'''

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
import datetime
import random
from itertools import chain

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_General, HSI_Lidar_Couple_Cross_Drop
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.huston_transfer_general import args
from loss.kd import *
from lib.model_develop import train_knowledge_distill_fc_feature_share_unimodal_center
from lib.processing_utils import get_file_list


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


    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    args.p = [1, 1]
    teacher_model = HSI_Lidar_Couple_General(args, modality_1_channel, modality_2_channel)

    args.p = [0, 0]
    student_model = HSI_Lidar_Couple_General(args, modality_1_channel, modality_2_channel)

    # 初始化并且固定teacher 网络参数

    if args.backbone=='RESNET':

        # teacher_model.load_state_dict(
        #     torch.load(os.path.join(args.model_root, 'share_unimodal_center_full/hsi_lidar_couple_cross_fc_version_' + str(3) + '.pth')))
        teacher_model.load_state_dict(
            torch.load(os.path.join(args.model_root,
                                    'RESNET/share/full_unimodal_center_1/hsi_lidar_RESNET_version_1.pth')))
        teacher_model.eval()
    elif args.backbone=='ALEX':
        # 初始化并且固定teacher 网络参数
        teacher_model.load_state_dict(
            torch.load(os.path.join(args.model_root,
                                    'ALEX/share/full_unimodal_center_1/hsi_lidar_ALEX_version_0.pth')))
    else:
        raise  RuntimeError('Error backbone')

    for param in teacher_model.parameters():
        param.requires_grad = False

    args.model_root = os.path.join(args.model_root, 'share_center_transfer', 'unimodal_center_kd_' + args.location)
    args.log_root = os.path.join(args.log_root, 'share_center_transfer', 'unimodal_center_kd_' + args.location)

    # 如果有GPU
    if torch.cuda.is_available():
        teacher_model.cuda()
        student_model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    elif args.kd_mode == 'at':
        criterionKD = AT(args.p)
    elif args.kd_mode == 'fitnet':
        criterionKD = Hint()

    else:
        raise Exception('Invalid kd mode...')
    if torch.cuda.is_available():
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.SGD(chain(student_model.parameters(),
                                              *[c.parameters() for c in criterionKD[1:]]),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
        else:
            optimizer = torch.optim.SGD(student_model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.Adam(chain(student_model.parameters(),
                                               *[c.parameters() for c in criterionKD[1:]]),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
        else:
            optimizer = torch.optim.Adam(student_model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
    else:
        print('optim error')
        optimizer = None

    # warp nets and criterions for train and test
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    train_knowledge_distill_fc_feature_share_unimodal_center(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                                                             train_loader=train_loader,
                                                             test_loader=test_loader,
                                                             args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
