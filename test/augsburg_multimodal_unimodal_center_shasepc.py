import sys

sys.path.append('..')
from models.shaspec import ShaSpec_Classfication_TRI
from src.augsburg_dataloader import augsburg_multi_dataloader, augsburg_multi_dataloader_tri
from configuration.augsburg_multi_config import args
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi, calc_accuracy_multi_tri
from lib.processing_utils import get_file_list
import torch.optim as optim

import os
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"]='3'


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main_share(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/augsburg"
    args.modal = 'multi'
    args.pair_modalities = ['hsi', 'sar', 'dsm']
    args.location = 'after'

    # print(args)

    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]

    mean_list = []
    model = ShaSpec_Classfication_TRI(args, modality_1_channel, modality_2_channel, modality_3_channel)
    for i in range(5):
        i = i + 0
        resuly_list = []
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_root,
                             'shaspec_full/huston2013_hsi_sar_dsm_couple_cross_fc_version_'+str(i)+'_unimodal_center_1.0.pth')))
        model.eval()

        modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]

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


def deeppix_main_share_kd(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/augsburg"
    args.modal = 'multi'
    args.location='after'
    args.pair_modalities = ['hsi', 'sar', 'dsm']

    # print(args)

    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]

    mean_list = []
    model = ShaSpec_Classfication_TRI(args, modality_1_channel, modality_2_channel, modality_3_channel)
    for i in range(3):
        path = os.path.join(args.model_root,
                            'share_transfer/kd_after/margin_0/dad_hsi+sar+dsm_lr_0.001_version_0_lambda_kd_feature_0.4_intra_0.pth')
        print(path)
        i = i + 0
        resuly_list = []
        model.load_state_dict(
            torch.load(path))
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


def deeppix_main_share_kd_unimodal_center(args):
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "../data/augsburg"
    args.modal = 'multi'
    args.location='after'
    args.pair_modalities = ['hsi', 'sar', 'dsm']

    # print(args)

    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]
    modality_3_channel = modality_to_channel[args.pair_modalities[2]]

    mean_list = []
    model = HSI_Lidar_Couple_Share_TRI(args, modality_1_channel, modality_2_channel, modality_3_channel)
    for i in range(3):
        path = os.path.join(args.model_root,
                            'share_transfer/unimodal_center_kd/margin_0/dad_hsi+sar+dsm_lr_0.001_version_' + str(
                                i) + '_lambda_kd_feature_5.0_lambda_center_loss_0.5.pth')
        print(path)
        i = i + 0
        resuly_list = []
        model.load_state_dict(
            torch.load(path))
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
    args.drop_mode = 'average'
    # deeppix_main_share(args)
    deeppix_main_share_kd(args)
    # deeppix_main_share_kd_unimodal_center(args)
