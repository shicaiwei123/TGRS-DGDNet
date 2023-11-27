'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple_Baseline, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_Share
from src.huston2013_dataloader import huston2013_multi_dataloader
from models.single_modality_model import Single_Modality_transfer
from configuration.single_transfer_single_config import args
import torch
from loss.kd import *
from lib.model_develop import train_knowledge_distill_fc_feature
from itertools import chain
import torch.optim as optim

import os
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

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    teacher_model = Single_Modality_transfer(modality_1_channel, args, pretrained=True)
    student_model = Single_Modality_transfer(modality_2_channel, args, pretrained=True)
    args.preserve_modality = args.student_data

    # 初始化并且固定teacher 网络参数
    model_dict = torch.load(os.path.join(args.model_root,
                                         args.teacher_data + "_" + args.student_data + '_hallucination_ensemble_multi_version_1.pth'))
    model_1_dict={}
    model_2_dict={}
    for key in model_dict.keys():
        # print(key)
        key_split=key.split('.')
        if key_split[0]=='modality_1_model':
            key_split.remove('modality_1_model')
            model_1_dict['.'.join(key_split)]=model_dict[key]
        else:
            key_split.remove('modality_2_model')
            model_2_dict['.'.join(key_split)]=model_dict[key]

    teacher_model.load_state_dict(model_1_dict)
    student_model.load_state_dict(model_2_dict)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 如果有GPU
    if torch.cuda.is_available():
        teacher_model.cuda()  # 将所有的模型参数移动到GPU上
    student_model.cuda()
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

    train_knowledge_distill_fc_feature(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
