import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_General, HSI_Lidar_Couple_DAD, \
    HSI_Lidar_Couple_Cross_Drop, HSI_Lidar_Couple_Cross_Drop_Auxi
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.multimdal_fusion_config import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi, calc_accuracy_multi
from lib.processing_utils import get_file_list
import torch.optim as optim

import os
import numpy as np
import random


def deeppix_main_share(args):
    '''利用unimodal center 正则化训练的多模太模型'''
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/huston2013_pixel7"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'lidar']
    args.backbone = 'RESNET'
    args.fusion_method = 'share'
    # args.pair_modalities = ['hsi', 'ms']

    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = HSI_Lidar_Couple_General(args, modality_1_channel, modality_2_channel)
    for i in range(3):
        i = i + 0
        path = os.path.join(args.model_root,
                            'RESNET/share/drop_unimodal_center_0/' + args.pair_modalities[0] + '_' +
                            args.pair_modalities[
                                1] + '_RESNET_version_' + str(i) + '.pth')
        print(path)
        model.load_state_dict(torch.load(path))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]

        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=True)
            print(result)
        print('\n')


def deeppix_main_kd_unimodal_center_for_student(args):
    '''利用unimodal center 正则化训练的多模太模型做教师,并且对学生也使用unimodal增强'''
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/huston2013_pixel7"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'lidar']
    args.location = 'after'
    args.backbone = 'ALEX'
    args.fusion_method = 'share'

    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    mean_list = []
    model = HSI_Lidar_Couple_General(args, modality_1_channel, modality_2_channel)
    for i in range(3):
        resuly_list = []
        path = 'share_center_transfer/unimodal_center_kd_' + args.location + '/hsi_lidar_ALEX_version_' + str(
            i) + '_lambda_kd_feature_0.5_lambda_center_loss_1.5.pth'
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, path)))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]
        print(path)
        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=False)
            print(result)
            resuly_list.append(result)
        resuly_arr = np.array(resuly_list)
        mean = np.mean(resuly_arr, axis=0)
        print(mean)
        mean_list.append(list(mean))
        print("\n")
    mean_arr = np.array(mean_list)
    mean = np.mean(mean_arr, axis=0)
    print(mean)


if __name__ == '__main__':
    deeppix_main_share(args)
    #
    # deeppix_main_kd_unimodal_center_for_student(args)

    # deeppix_main_dgd_unimodal_center(args)
