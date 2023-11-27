import sys
import os

sys.path.append('..')
import matplotlib.pyplot as plt

import numpy as np
import rscls
from scipy import stats
import time
import torch
from lib.processing_utils import load_mat
from src.augsburg_dataloader import augsburg_multi_dataloader_tri
from src.huston2013_dataloader import huston2013_multi_dataloader
from tqdm import tqdm
from models.resnet_ensemble import HSI_Lidar_Couple_Cross, HSI_Lidar_Couple_Share_TRI,HSI_Lidar_Couple_Share
from models.shaspec import ShaSpec_Classfication, ShaSpec_Classfication_TRI,ShaSpec_Classfication
from models.single_modality_model import Single_Modality
import torch.multiprocessing
from models.surf_mmfomer import HSI_Lidar_Couple_Former_TRI,HSI_Lidar_Couple_Former
from models.surf_corr_repr import SURF_CR_TRI,SURF_MV_TRI,SURF_MV,SURF_CR_TWO

torch.multiprocessing.set_sharing_strategy('file_system')
import torch

ensemble = 1  # times of snapshot ensemble
patch = 7
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def augsburg_prediction_plot(model, data_lodaer, name):
    imx = 332
    imy = 485
    model.eval()
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    label_full = []
    for batch_sample in tqdm(iter(data_lodaer), desc="Full forward pass", total=len(data_lodaer), disable=False):
        m1 = batch_sample['m_1']
        m2 = batch_sample['m_2']
        m3 = batch_sample['m_3']
        label = batch_sample['label']

        m1 = torch.repeat_interleave(m1, 180, 1)
        m2 = torch.repeat_interleave(m2, 4, 1)

        if use_cuda:
            m1 = m1.cuda()
            m2 = m2.cuda()
            m3 = m3.cuda()
        with torch.no_grad():
            outputs_batch = model(m1, m2, m3)
            if isinstance(outputs_batch, tuple):
                outputs_batch = model(m1, m2, m3)[0]
        outputs_full.append(outputs_batch)
        label_full.append(label)

    outputs_full = torch.cat(outputs_full, dim=0)
    label_full = torch.cat(label_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)

    # raw classification
    predict_arr = np.array(labels_predicted.cpu())
    # label_arr = np.array(label_full.cpu())
    # acc = np.sum(label_arr == predict_arr) / (len(label_arr))
    # print(acc)
    pre_all_1 = np.reshape(predict_arr, (1, imx, imy), 'F')
    pre1 = np.int8(stats.mode(pre_all_1, axis=0)[0]).reshape(imx, imy)

    # after post processin using superpixel-based refinement
    # rscls.save_cmap(pre1, 'jet', 'pre.png')
    save_figure(pre1, name)


def huston_prediction_plot(model, data_lodaer, name):
    imx = 349
    imy = 1905
    model.eval()
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    for batch_sample in tqdm(iter(data_lodaer), desc="Full forward pass", total=len(data_lodaer), disable=False):
        m1 = batch_sample['m_1']
        m2 = batch_sample['m_2']
        if use_cuda:
            m1 = m1.cuda()
            m2 = m2.cuda()
        with torch.no_grad():
            m1 = torch.repeat_interleave(m1, 144, dim=1)
            outputs_batch = model(m1, m2)
            if isinstance(outputs_batch, tuple):
                outputs_batch = model(m1, m2)[0]

        # print(outputs_batch.shape)
        outputs_full.append(outputs_batch)

    outputs_full = torch.cat(outputs_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)

    # raw classification
    predict_arr = np.array(labels_predicted.cpu())
    pre_all_1 = np.reshape(predict_arr, (1, imx, imy), 'F')
    pre1 = np.int8(stats.mode(pre_all_1, axis=0, )[0]).reshape(imx, imy)

    # after post processin using superpixel-based refinement
    # rscls.save_cmap(pre1, 'jet', 'pre.png')
    save_figure_huston(pre1, name)


def save_figure(pre1, name):
    img_new = np.zeros(list(pre1.shape) + [3], dtype='uint8')

    label_color = {2: [240, 240, 85], 5: [85, 221, 213], 1: [228, 55, 38], 4: [130, 222, 39],
                   0: [55, 107, 52], 3: [170, 228, 49], 6: [86, 132, 193]}

    label_color_huston = {3: [63, 127, 83], 6: [107, 89, 199], 2: [155, 192, 27], 5: [136, 74, 43],
                          1: [86, 177, 18], 4: [95, 190, 90], 7: [255, 255, 255], 8: [205, 173, 210],
                          9: [213, 51, 35], 10: [118, 35, 28], 11: [60, 94, 142], 12: [231, 222, 76],
                          13: [217, 139, 45], 14: [86, 34, 133], 15: [233, 140, 84]}

    for key in label_color.keys():
        # r通道标签为5的位置，设置为0
        img_new[..., 0][pre1 == key] = label_color[key][0]
        # g通道标签为5的位置，设置为255
        img_new[..., 1][pre1 == key] = label_color[key][1]
        # b通道标签为5的位置，设置为255
        img_new[..., 2][pre1 == key] = label_color[key][2]

    plt.figure("new")
    plt.imshow(img_new)
    plt.axis("off")
    # plt.show()
    plt.savefig(name, dpi=800, bbox_inches='tight', pad_inches=-0.1)


def save_figure_huston(pre1, name):
    img_new = np.zeros(list(pre1.shape) + [3], dtype='uint8')

    label_color = {2: [63, 127, 83], 5: [107, 89, 199], 1: [155, 192, 27], 4: [136, 74, 43],
                   0: [86, 177, 18], 3: [95, 190, 90], 6: [255, 255, 255], 7: [205, 173, 210],
                   8: [213, 51, 35], 9: [118, 35, 28], 10: [60, 94, 142], 11: [231, 222, 76],
                   12: [217, 139, 45], 13: [86, 34, 133], 14: [233, 140, 84]}

    for key in label_color.keys():
        # r通道标签为5的位置，设置为0
        img_new[..., 0][pre1 == key] = label_color[key][0]
        # g通道标签为5的位置，设置为255
        img_new[..., 1][pre1 == key] = label_color[key][1]
        # b通道标签为5的位置，设置为255
        img_new[..., 2][pre1 == key] = label_color[key][2]

    plt.figure("new")
    plt.imshow(img_new)
    plt.axis("off")
    # plt.show()
    plt.savefig(name, dpi=800, bbox_inches='tight', pad_inches=-0.1)


def augsburg_plot():
    # from configuration.augsburg_multi_config import args
    from configuration.mmformer import args
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "/home/bigspace/shicaiwei/remote_sensing/full/"
    args.modal = 'dsm'
    args.pair_modalities = [args.modal, args.modal, args.modal]
    args.miss = 'None'
    args.drop_mode = 'average'
    args.location = 'after'
    args.class_num=7

    modality_to_channel = {'hsi': 180, 'sar': 4, 'dsm': 1}
    test_loader = augsburg_multi_dataloader_tri(train=False, args=args)
    args.p = [0, 0, 1]

    model = HSI_Lidar_Couple_Share_TRI(args, 180, 4, 1)

    # for i in range(5):
    model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                "share_transfer/unimodal_center_kd/margin_0/dad_hsi+sar+dsm_lr_0.001_version_1_lambda_kd_feature_10.0_lambda_center_loss_0.5.pth")))

    #mmformer
    model=HSI_Lidar_Couple_Former_TRI(args, 180, 4, 1)
    model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                "mmformer_drop/augsburg_hsi_sar_dsm_couple_cross_fc_version_2.pth")))


    model=SURF_CR_TRI(args,180,4,1)

    model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                "corr_drop/augsburg_hsi_sar_dsm_couple_cross_fc_version_2.pth")))



    model=SURF_MV_TRI(args,180,4,1)

    model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                "mv_drop/augsburg_hsi_sar_dsm_couple_cross_fc_version_2.pth")))



    model = HSI_Lidar_Couple_Share_TRI(args, 180, 4, 1)

    # for i in range(5):
    model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                'share_transfer/unimodal_center_kd/margin_0/dad_hsi+sar+dsm_lr_0.001_version_1_lambda_kd_feature_5.0_lambda_center_loss_0.5.pth')))


    model.eval()

    augsburg_prediction_plot(model, test_loader, './augsburg_dsm_shaspe.png')


def huston_plot():
    # from configuration.multimdal_fusion_config import args
    from configuration.mmformer import args
    args.log_name = args.name
    args.model_name = args.name
    args.data_root = "/home/bigspace/shicaiwei/remote_sensing/full/"
    args.modal = 'lidar'
    args.pair_modalities = ['lidar', 'lidar']
    args.drop_mode='average'
    args.batch_size=256
    args.location = 'after'


    test_loader = huston2013_multi_dataloader(train=False, args=args)

    # model = Hallucination_ensemble(args, channel_dict=modality_to_channel,
    #                                modality_1_dict=None,
    #                                modality_2_dict=None)
    #
    args.p = [0, 1]
    model = ShaSpec_Classfication(args, 144, 1)

    # model = Single_Modality(144, args, pretrained=False)

    # for i in range(5):
    args.p = [0, 1]

    model = HSI_Lidar_Couple_Share(args, 144, 1)

    # for i in range(5):
    model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                "share_center_transfer/unimodal_center_kd_after/margin_0/dad_hsi+lidar_lr_0.001_version_0_lambda_kd_feature_30.0_lambda_center_loss_1.5.pth")))

    # # mmformer
    model = HSI_Lidar_Couple_Former(args, 144, 1)
    model.load_state_dict(
        torch.load(os.path.join(args.model_root,
                                "mmformer_drop/hsi_lidar_couple_cross_fc_version_1.pth")))
    #
    # model = SURF_CR_TWO(args, 144, 1)
    #
    # model.load_state_dict(
    #     torch.load(os.path.join(args.model_root,
    #                             "corr_drop/hsi_lidar_couple_cross_fc_version_0.pth")))
    # #
    # model = SURF_MV(args, 144, 1)
    #
    # model.load_state_dict(
    #     torch.load(os.path.join(args.model_root,
    #                             "mv_drop/hsi_lidar_couple_cross_fc_version_0.pth")))
    # #
    # model = HSI_Lidar_Couple_Share(args, 144, 1)
    #
    # # for i in range(5):
    # model.load_state_dict(
    #     torch.load(os.path.join(args.model_root,
    #                             'share_full/hsi_lidar_couple_cross_fc_version_2.pth')))

    model.eval()


    huston_prediction_plot(model, test_loader, './huston_hsi_rf.png')


if __name__ == '__main__':
    huston_plot()
    # augsburg_plot()
    # imx = 332
    # imy = 485
    # a=load_mat('/home/shicaiwei/data/remote_sensing/full/dsm_Y_test.mat')
    # b=np.reshape(a, (1, imx, imy), 'A')
    # print(1)
