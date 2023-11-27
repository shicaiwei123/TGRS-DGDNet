'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.surf_mmfomer import HSI_Lidar_Couple_Former,HSI_Lidar_Couple_Former_TRI
from src.huston2013_dataloader import huston2013_multi_dataloader
from src.augsburg_dataloader import augsburg_multi_dataloader,augsburg_multi_dataloader_tri
from configuration.mmformer import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi, calc_accuracy_multi,calc_accuracy_multi_tri
from lib.processing_utils import get_file_list
import torch.optim as optim

import os
import numpy as np
import random


def deeppix_main_share(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/huston2013_pixel7"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'lidar']
    args.location='after'
    # args.pair_modalities = ['hsi', 'ms']

    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    mean_list=[]
    model = HSI_Lidar_Couple_Former(args, modality_1_channel, modality_2_channel)
    for i in range(3):
        resuly_list=[]
        i = i + 0
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'mmformer_full/'+"huston2013_"+args.pair_modalities[0]+'_'+args.pair_modalities[1]+'_couple_cross_fc_version_' + str(i) + '_unimodal_center_1.5.pth')))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]

        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=True)
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

def augsburg_main_share(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/augsburg"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'sar']
    args.location='after'

    # print(args)

    test_loader = augsburg_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    mean_list=[]
    model = HSI_Lidar_Couple_Former(args, modality_1_channel, modality_2_channel)
    args.class_num=7
    for i in range(3):
        resuly_list=[]
        i = i + 0
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'mmformer_drop/augsburg_'+args.pair_modalities[0]+'_'+args.pair_modalities[1]+'_couple_cross_fc_version_' + str(i) + '.pth')))
        model.eval()

        modality_combination = [[1, 0], [0, 1], [1, 1]]

        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi(model=model, loader=test_loader, args=args, hter=False, verbose=True)
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


def augsburg_main_share_tri(args):
    args.miss=None
    args.class_num=7
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/augsburg"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'sar', 'dsm']
    args.location='after'

    # print(args)

    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]

    mean_list = []
    model = HSI_Lidar_Couple_Former_TRI(args, modality_1_channel, modality_2_channel,modality_3_channel)

    for i in range(3):
        resuly_list=[]
        i = i + 0
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root, 'mmformer_drop/augsburg_hsi_sar_dsm_couple_cross_fc_version_' + str(i) + '.pth')))
        model.eval()

        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

        for p in modality_combination:
            model.p = p

            args.retrain = False
            result = calc_accuracy_multi_tri(model=model, loader=test_loader, args=args, hter=False, verbose=True)
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
    deeppix_main_share(args=args)
    #
    # augsburg_main_share(args)
    # augsburg_main_share_tri(args=args)
