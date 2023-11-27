'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.surf_corr_repr import SURF_MV_TRI
from src.augsburg_dataloader import augsburg_multi_dataloader_tri
from configuration.mmformer import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_share_tri
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

    args.location='after'
    args.drop_mode='average'
    args.miss = None
    args.class_num = 7
    train_loader = augsburg_multi_dataloader_tri(train=True, args=args)
    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)

    args.log_name = args.name
    args.model_name = args.name

    if isinstance(args.p, str):
        args.p = eval(args.p)

    if 1 not in args.p:
        args.model_root = args.model_root + "/mv_drop"
        args.log_root = args.log_root + "/mv_drop"
    else:
        args.model_root = args.model_root + "/mv_full"
        args.log_root = args.log_root + "/mv_full"

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]

    model = SURF_MV_TRI(args, modality_1_channel, modality_2_channel, modality_3_channel)
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
    train_base_multi_share_tri(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader,
                           test_loader=test_loader,
                           args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
