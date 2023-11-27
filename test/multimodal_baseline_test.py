'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Baseline, HSI_Lidar_MDMB, HSI_Lidar_Couple, HSI_Lidar_CCR, \
    HSI_Lidar_Couple_Late, HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_Share, HSI_Lidar_Couple_DAD, \
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


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def deeppix_main_share(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/huston2013_pixel7"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'lidar']
    # args.pair_modalities = ['hsi', 'ms']

    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = HSI_Lidar_Couple_Share(args, modality_1_channel, modality_2_channel)
    for i in range(3):
        i = i + 0
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'share_full/'+args.pair_modalities[0]+'_'+args.pair_modalities[1]+'_couple_cross_fc_version_' + str(i) + '.pth')))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]

        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=True)
            print(result)
        print('\n')


def deeppix_main_auxi(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/huston2013_pixel7"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'lidar']

    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = HSI_Lidar_Couple_Cross_Drop_Auxi(args, modality_1_channel, modality_2_channel)
    for i in range(4):
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'auxi/margin_1/mse_hsi+lidar_lr_0.001_version_' + str(i) + '.pth')))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]

        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=True)
            print(result)

        print("\n")


def deeppix_main_kd(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/huston2013_pixel7"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'lidar']

    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    mean_list=[]
    model = HSI_Lidar_Couple_Share(args, modality_1_channel, modality_2_channel)
    for i in range(3):
        resuly_list = []
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'share_center_transfer/kd/margin_0/dad_hsi+lidar_lr_0.001_version_' + str(i)+'_lambda_kd_feature_30.0_intra_0' + '.pth')))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]
        print(os.path.join(args.model_root, 'share_center_transfer/kd/margin_0/dad_hsi+lidar_lr_0.001_version_' + str(i)+'_lambda_kd_feature_30.0_intra_0' + '.pth'))
        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=True)
            print(result)
            resuly_list.append(result)
        resuly_arr=np.array(resuly_list)
        mean=np.mean(resuly_arr,axis=0)
        print(mean)
        mean_list.append(list(mean))
        print("\n")
    mean_arr=np.array(mean_list)
    mean=np.mean(mean_arr,axis=0)
    print(mean)









def deeppix_main_kd_hm(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/huston2013_pixel7"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'ms']

    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    mean_list=[]
    model = HSI_Lidar_Couple_Share(args, modality_1_channel, modality_2_channel)
    for i in range(3):
        resuly_list = []
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'share_transfer/kd/margin_0/dad_hsi+ms_lr_0.001_version_' + str(i)+'_lambda_kd_feature_40.0_intra_0' + '.pth')))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]
        print(os.path.join(args.model_root, 'share_transfer/kd/margin_0/dad_hsi+ms_lr_0.001_version_' + str(i)+'_lambda_kd_feature_40.0_intra_0' + '.pth'))
        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=True)
            print(result)
            resuly_list.append(result)
        resuly_arr=np.array(resuly_list)
        mean=np.mean(resuly_arr,axis=0)
        print(mean)
        mean_list.append(list(mean))
        print("\n")
    mean_arr=np.array(mean_list)
    mean=np.mean(mean_arr,axis=0)
    print(mean)











if __name__ == '__main__':
    # deeppix_main(args)
    deeppix_main_share(args=args)
    # deeppix_main_kd(args)
    # deeppix_main_kd_hm(args)
    # deeppix_main_kd_cross(args)