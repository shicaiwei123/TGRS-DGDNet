'''
a template for train, you need to fix your own main function
'''

import sys
import os

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_Share_TRI, HSI_Lidar_Couple_Cross_Drop
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.unimodal_center_share_kd_unimodal_center import args
import torch
import torch.nn as nn
from src.augsburg_dataloader import augsburg_multi_dataloader_tri
from lib.processing_utils import get_file_list
import torch.optim as optim
from loss.kd import *
import cv2
import numpy as np
import datetime
import random
from lib.model_develop import train_knowledge_distill_fc_feature_share_tri_unimodal_center


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main(args):


    args.class_num=7
    args.drop_mode='average'

    train_loader = augsburg_multi_dataloader_tri(train=True, args=args)
    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)

    args.log_name = args.name
    args.model_name = args.name

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]


    args.p = [1, 1, 1]
    teacher_model = HSI_Lidar_Couple_Share_TRI(args, modality_1_channel, modality_2_channel, modality_3_channel)

    args.p = [0, 0, 0]
    student_model = HSI_Lidar_Couple_Share_TRI(args, modality_1_channel, modality_2_channel, modality_3_channel)

    # 初始化并且固定teacher 网络参数
    # teacher_model.load_state_dict(
    #     torch.load(os.path.join(args.model_root,
    #                             'augsburg/average/share_unimodal_center_full/augsburg_hsi_sar_dsm_couple_cross_fc_version_' + str(1) + '.pth')))

    teacher_model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                'augsburg/average/share_unimodal_center_full/augsburg_hsi_sar_dsm_couple_cross_fc_version_' + str(
                                    1) + '_labma_unimodal3.0.pth')))

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    args.model_root = os.path.join(args.model_root, 'share_transfer', 'unimodal_center_kd', 'margin_' + str(args.margin))
    args.log_root = os.path.join(args.log_root, 'share_transfer', 'unimodal_center_kd', 'margin_' + str(args.margin))
    # 如果有GPU
    if torch.cuda.is_available():
        student_model.cuda()  # 将所有的模型参数移动到GPU上
        teacher_model.cuda()
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

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, student_model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
    #                        weight_decay=args.weight_decay)

    # warp nets and criterions for train and test
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    args.retrain = False
    train_knowledge_distill_fc_feature_share_tri_unimodal_center(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                                                 train_loader=train_loader,
                                                 test_loader=test_loader,
                                                 args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
