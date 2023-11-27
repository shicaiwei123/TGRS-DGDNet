import torch
import torch.nn as nn
import torchvision.transforms as ttf
import torch.optim as optim
from models.resnet18_se import resnet18_se
from models.base_model import MDMB_extract, MDMB_fusion, Couple_CNN, CCR, MDMB_fusion_middle, MDMB_fusion_share
from lib.model_develop_utils import modality_drop


class fusion_module(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       )

    def forward(self, x):
        x = self.block_5_1(x)
        x = self.block_5_2(x)
        return x


# class ShaSpec_Classfication(nn.Module):
#     def __init__(self, args, modality_1_channel, modality_2_channel):
#         super().__init__()
#
#         self.modality_encoder_1 = Couple_CNN(modality_1_channel)
#         self.modality_encoder_2 = Couple_CNN(modality_2_channel)
#         channel_list = [modality_1_channel, modality_2_channel]
#         max_channel_index = channel_list.index(max(channel_list))
#         self.min_channel_index = channel_list.index(min(channel_list))
#
#         self.share_encoder = Couple_CNN(channel_list[max_channel_index])
#         self.modality_projector = torch.nn.Conv2d(channel_list[self.min_channel_index], channel_list[max_channel_index],
#                                                   1, 1,
#                                                   0)
#         self.fusion_projector = fusion_module(256)
#         self.modality_1_transfer = fusion_module(128)
#         self.modality_2_transfer = fusion_module(128)
#         self.target_classifier = nn.Linear(128, args.class_num)
#         self.unimodal_target_classifier = nn.Linear(128, args.class_num)
#         self.modality_classifier = nn.Linear(128, 2)
#         self.pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.p = args.p
#         self.args = args
#
#     def forward(self, modality1, modality2):
#         m1_feature_specific = self.modality_encoder_1(modality1)
#         m2_feature_specific = self.modality_encoder_2(modality2)
#
#         m1_feature_specific_cache = m1_feature_specific
#         m2_feature_specific_cache = m2_feature_specific
#
#         if self.min_channel_index == 0:
#             modality1_transfer = self.modality_projector(modality1)
#             m1_feature_share = self.share_encoder(modality1_transfer)
#             m2_feature_share = self.share_encoder(modality2)
#         elif self.min_channel_index == 1:
#             modality_transfer = self.modality_projector(modality2)
#             m2_feature_share = self.share_encoder(modality_transfer)
#             m1_feature_share = self.share_encoder(modality1)
#         else:
#             raise ValueError
#
#         m1_feature_share_cache = m1_feature_share
#         m2_feature_share_cache = m2_feature_share
#
#         # print(m1_feature_specific.shape)
#
#         [m1_feature_specific, m2_feature_specific], p = modality_drop([m1_feature_specific, m2_feature_specific],
#                                                                       self.p, args=self.args)
#         data = [m1_feature_share, m2_feature_share]
#         for index in range(len(data)):
#             data[index] = data[index] * p[:, index]
#
#         # padding share representation for missing modality
#
#         [m1_feature_share, m2_feature_share] = data
#
#         m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
#         m1_feature_share[m1_missing_index] = m2_feature_share[m1_missing_index]
#
#         m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
#         m2_feature_share[m2_missing_index] = m1_feature_share[m2_missing_index]
#
#         # fusion
#
#         m1_fusion_out = self.fusion_projector(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
#         m1_feature_specific = self.modality_1_transfer(m1_feature_specific)
#
#         # print(m1_feature_specific.shape)
#         m1_fusion_out = m1_fusion_out + m1_feature_specific
#
#         m2_fusion_out = self.fusion_projector(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
#         m2_feature_specific = self.modality_2_transfer(m2_feature_specific)
#         m2_fusion_out = m2_fusion_out + m2_feature_specific
#
#         fusion_feature = torch.cat([m1_fusion_out, m2_fusion_out], dim=1)
#
#         fusion_feature = self.pooling(fusion_feature)
#         fusion_feature = fusion_feature.view(fusion_feature.shape[0], -1)
#
#         # calculate loss
#
#         target_predict = self.target_classifier(fusion_feature)
#
#         m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
#         m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)
#
#         m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
#             m1_feature_specific_cache_pooling.shape[0],
#             -1)
#         m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
#             m2_feature_specific_cache_pooling.shape[0],
#             -1)
#
#         specific_feature = torch.cat((m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling), dim=0)
#         specific_feature_label = torch.cat(
#             [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
#              torch.ones(m2_feature_specific_cache_pooling.shape[0])], dim=0).long().cuda()
#
#         dco_predict = self.modality_classifier(specific_feature)
#
#         m1_predict = self.unimodal_target_classifier(m1_feature_specific_cache_pooling)
#         m2_predict = self.unimodal_target_classifier(m2_feature_specific_cache_pooling)
#
#         return target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m1_predict, m2_predict


class ShaSpec_Classfication(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.modality_encoder_1 = Couple_CNN(modality_1_channel)
        self.modality_encoder_2 = Couple_CNN(modality_2_channel)
        channel_list = [modality_1_channel, modality_2_channel]
        max_channel_index = channel_list.index(max(channel_list))
        self.min_channel_index = channel_list.index(min(channel_list))

        self.share_encoder = Couple_CNN(channel_list[max_channel_index])
        self.modality_projector = torch.nn.Conv2d(channel_list[self.min_channel_index], channel_list[max_channel_index],
                                                  1, 1,
                                                  0)
        self.fusion_projector = nn.Conv2d(256, 128, 1, 1, 0)
        self.fusion = fusion_module(256)
        self.modality_2_transfer = fusion_module(128)
        self.target_classifier = nn.Linear(64, args.class_num)
        self.unimodal_target_classifier = nn.Linear(128, args.class_num)
        self.modality_classifier = nn.Linear(128, 2)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.p = args.p
        self.args = args

    def forward(self, modality1, modality2):
        m1_feature_specific = self.modality_encoder_1(modality1)
        m2_feature_specific = self.modality_encoder_2(modality2)

        m1_feature_specific_cache = m1_feature_specific
        m2_feature_specific_cache = m2_feature_specific

        if self.min_channel_index == 0:
            modality1_transfer = self.modality_projector(modality1)
            m1_feature_share = self.share_encoder(modality1_transfer)
            m2_feature_share = self.share_encoder(modality2)
        elif self.min_channel_index == 1:
            modality_transfer = self.modality_projector(modality2)
            m2_feature_share = self.share_encoder(modality_transfer)
            m1_feature_share = self.share_encoder(modality1)
        else:
            raise ValueError

        m1_feature_share_cache = m1_feature_share
        m2_feature_share_cache = m2_feature_share

        [m1_feature_specific, m2_feature_specific], p = modality_drop([m1_feature_specific, m2_feature_specific],
                                                                      self.p, args=self.args)
        data = [m1_feature_share, m2_feature_share]
        for index in range(len(data)):
            data[index] = data[index] * p[:, index]

        # padding share representation for missing modality

        [m1_feature_share, m2_feature_share] = data

        m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
        m1_feature_share[m1_missing_index] = m2_feature_share[m1_missing_index]

        m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
        m2_feature_share[m2_missing_index] = m1_feature_share[m2_missing_index]

        # fusion

        m1_fusion_out = self.fusion_projector(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
        m1_fusion_out = m1_fusion_out + m1_feature_specific

        m2_fusion_out = self.fusion_projector(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
        m2_fusion_out = m2_fusion_out + m2_feature_specific

        fusion_feature = self.fusion(torch.cat([m1_fusion_out, m2_fusion_out], dim=1))

        fusion_feature = self.pooling(fusion_feature)
        fusion_feature = fusion_feature.view(fusion_feature.shape[0], -1)

        # calculate loss

        target_predict = self.target_classifier(fusion_feature)

        m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
        m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)

        m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
            m1_feature_specific_cache_pooling.shape[0],
            -1)
        m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
            m2_feature_specific_cache_pooling.shape[0],
            -1)

        specific_feature = torch.cat((m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling), dim=0)
        specific_feature_label = torch.cat(
            [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0])], dim=0).long().cuda()

        dco_predict = self.modality_classifier(specific_feature)

        m1_predict = self.unimodal_target_classifier(m1_feature_specific_cache_pooling)
        m2_predict = self.unimodal_target_classifier(m2_feature_specific_cache_pooling)

        return target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m1_predict, m2_predict


class ShaSpec_Classfication_DGD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.modality_encoder_1 = Couple_CNN(modality_1_channel)
        self.modality_encoder_2 = Couple_CNN(modality_2_channel)
        channel_list = [modality_1_channel, modality_2_channel]
        max_channel_index = channel_list.index(max(channel_list))
        self.min_channel_index = channel_list.index(min(channel_list))

        self.share_encoder = Couple_CNN(channel_list[max_channel_index])
        self.modality_projector = torch.nn.Conv2d(channel_list[self.min_channel_index], channel_list[max_channel_index],
                                                  1, 1,
                                                  0)
        self.fusion_projector = nn.Conv2d(256, 128, 1, 1, 0)
        self.fusion = fusion_module(256)
        self.modality_2_transfer = fusion_module(128)
        self.target_classifier = nn.Linear(64, args.class_num)
        self.unimodal_target_classifier = nn.Linear(128, args.class_num)
        self.modality_classifier = nn.Linear(128, 2)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.p = args.p
        self.args = args

    def forward(self, modality1, modality2):
        m1_feature_specific = self.modality_encoder_1(modality1)
        m2_feature_specific = self.modality_encoder_2(modality2)

        m1_feature_specific_cache = m1_feature_specific
        m2_feature_specific_cache = m2_feature_specific

        if self.min_channel_index == 0:
            modality1_transfer = self.modality_projector(modality1)
            m1_feature_share = self.share_encoder(modality1_transfer)
            m2_feature_share = self.share_encoder(modality2)
        elif self.min_channel_index == 1:
            modality_transfer = self.modality_projector(modality2)
            m2_feature_share = self.share_encoder(modality_transfer)
            m1_feature_share = self.share_encoder(modality1)
        else:
            raise ValueError

        m1_feature_share_cache = m1_feature_share
        m2_feature_share_cache = m2_feature_share

        [m1_feature_specific, m2_feature_specific], p = modality_drop([m1_feature_specific, m2_feature_specific],
                                                                      self.p, args=self.args)
        data = [m1_feature_share, m2_feature_share]
        for index in range(len(data)):
            data[index] = data[index] * p[:, index]

        # padding share representation for missing modality

        [m1_feature_share, m2_feature_share] = data

        m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
        m1_feature_share[m1_missing_index] = m2_feature_share[m1_missing_index]

        m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
        m2_feature_share[m2_missing_index] = m1_feature_share[m2_missing_index]

        # fusion

        m1_fusion_out = self.fusion_projector(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
        m1_fusion_out = m1_fusion_out + m1_feature_specific

        m2_fusion_out = self.fusion_projector(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
        m2_fusion_out = m2_fusion_out + m2_feature_specific

        fusion_feature = self.fusion(torch.cat([m1_fusion_out, m2_fusion_out], dim=1))

        fusion_feature_cache = fusion_feature

        fusion_feature = self.pooling(fusion_feature)
        fusion_feature = fusion_feature.view(fusion_feature.shape[0], -1)

        # calculate loss

        target_predict = self.target_classifier(fusion_feature)

        m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
        m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)

        m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
            m1_feature_specific_cache_pooling.shape[0],
            -1)
        m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
            m2_feature_specific_cache_pooling.shape[0],
            -1)

        specific_feature = torch.cat((m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling), dim=0)
        specific_feature_label = torch.cat(
            [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0])], dim=0).long().cuda()

        dco_predict = self.modality_classifier(specific_feature)

        m1_predict = self.unimodal_target_classifier(m1_feature_specific_cache_pooling)
        m2_predict = self.unimodal_target_classifier(m2_feature_specific_cache_pooling)

        return target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, fusion_feature_cache, m2_feature_share_cache, m1_predict, m2_predict


class ShaSpec_Classfication_TRI(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()

        self.modality_encoder_1 = Couple_CNN(modality_1_channel)
        self.modality_encoder_2 = Couple_CNN(modality_2_channel)
        self.modality_encoder_3 = Couple_CNN(modality_3_channel)

        self.share_encoder = Couple_CNN(modality_1_channel)
        self.modality_projector_2 = torch.nn.Conv2d(modality_2_channel, modality_1_channel,
                                                    1, 1,
                                                    0)

        self.modality_projector_3 = torch.nn.Conv2d(modality_3_channel, modality_1_channel,
                                                    1, 1,
                                                    0)
        self.fusion_projector = nn.Conv2d(256, 128, 1, 1, 0)
        self.fusion = fusion_module(384)
        self.modality_2_transfer = fusion_module(128)
        self.target_classifier = nn.Linear(64, args.class_num)
        self.unimodal_target_classifier = nn.Linear(128, args.class_num)
        self.modality_classifier = nn.Linear(128, 3)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.p = args.p
        self.args = args

    def forward(self, modality1, modality2, modality3):
        m1_feature_specific = self.modality_encoder_1(modality1)
        m2_feature_specific = self.modality_encoder_2(modality2)
        m3_feature_specific = self.modality_encoder_3(modality3)

        m1_feature_specific_cache = m1_feature_specific
        m2_feature_specific_cache = m2_feature_specific
        m3_feature_specific_cache = m3_feature_specific

        modality_2_transfer = self.modality_projector_2(modality2)
        modality_3_transfer = self.modality_projector_3(modality3)
        m1_feature_share = self.share_encoder(modality1)
        m2_feature_share = self.share_encoder(modality_2_transfer)
        m3_feature_share = self.share_encoder(modality_3_transfer)

        m1_feature_share_cache = m1_feature_share
        m2_feature_share_cache = m2_feature_share
        m3_feature_share_cache = m3_feature_share

        [m1_feature_specific, m2_feature_specific, m3_feature_specific], p = modality_drop(
            [m1_feature_specific, m2_feature_specific, m3_feature_specific],
            self.p, args=self.args)
        data = [m1_feature_share, m2_feature_share, m3_feature_share]
        for index in range(len(data)):
            data[index] = data[index] * p[:, index]

        # padding share representation for missing modality

        [m1_feature_share, m2_feature_share, m3_feature_share] = data

        m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
        m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
        m3_missing_index = [not bool(i) for i in (torch.sum(m3_feature_share, dim=[1, 2, 3]))]

        # print(((~(
        #         torch.tensor(m2_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        #
        # print(m1_missing_index)
        # print(m1_feature_share[m1_missing_index] )

        m1_feature_share[m1_missing_index] = (m2_feature_share[m1_missing_index] + m3_feature_share[
            m1_missing_index])
        m1_c = (((~(torch.tensor(m2_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        m1_c = m1_c.unsqueeze(1)
        m1_c = m1_c.unsqueeze(2)
        m1_c = m1_c.unsqueeze(3)
        m1_c = m1_c.repeat(1, 128, 4, 4).cuda()
        m1_feature_share[m1_missing_index] = m1_feature_share[m1_missing_index] / m1_c[m1_missing_index]

        # m1_feature_share=m1_feature_share/

        # / (((~(
        # torch.tensor(m2_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1)[m1_missing_index])  # 1 0 --1   0 0 2

        m2_feature_share[m2_missing_index] = (m1_feature_share[m2_missing_index] + m3_feature_share[
            m2_missing_index])
        m2_c = (((~(torch.tensor(m1_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        m2_c = m2_c.unsqueeze(1)
        m2_c = m2_c.unsqueeze(2)
        m2_c = m2_c.unsqueeze(3)
        m2_c = m2_c.repeat(1, 128, 4, 4).cuda()
        m2_feature_share[m2_missing_index] = m2_feature_share[m2_missing_index] / m2_c[m2_missing_index]

        # / ((~(torch.tensor(m1_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1)[m2_missing_index]

        m3_feature_share[m3_missing_index] = (m1_feature_share[m3_missing_index] + m3_feature_share[
            m3_missing_index])
        m3_c = (((~(torch.tensor(m1_missing_index) ^ torch.tensor(m2_missing_index))).float() + 1))
        m3_c = m3_c.unsqueeze(1)
        m3_c = m3_c.unsqueeze(2)
        m3_c = m3_c.unsqueeze(3)
        m3_c = m3_c.repeat(1, 128, 4, 4).cuda()
        m3_feature_share[m3_missing_index] = m3_feature_share[m3_missing_index] / m3_c[m3_missing_index]

        # fusion

        m1_fusion_out = self.fusion_projector(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
        m1_fusion_out = m1_fusion_out + m1_feature_specific

        m2_fusion_out = self.fusion_projector(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
        m2_fusion_out = m2_fusion_out + m2_feature_specific

        m3_fusion_out = self.fusion_projector(torch.cat((m3_feature_specific, m3_feature_share), dim=1))
        m3_fusion_out = m3_fusion_out + m3_feature_specific

        fusion_feature = self.fusion(torch.cat([m1_fusion_out, m2_fusion_out, m3_fusion_out], dim=1))

        fusion_feature = self.pooling(fusion_feature)
        fusion_feature = fusion_feature.view(fusion_feature.shape[0], -1)

        # calculate loss

        target_predict = self.target_classifier(fusion_feature)

        m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
        m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)
        m3_feature_specific_cache_pooling = self.pooling(m3_feature_specific_cache)

        m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
            m1_feature_specific_cache_pooling.shape[0],
            -1)
        m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
            m2_feature_specific_cache_pooling.shape[0],
            -1)

        m3_feature_specific_cache_pooling = m3_feature_specific_cache_pooling.view(
            m3_feature_specific_cache_pooling.shape[0],
            -1)

        specific_feature = torch.cat(
            (m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling, m3_feature_specific_cache_pooling),
            dim=0)
        specific_feature_label = torch.cat(
            [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0]) + 1], dim=0).long().cuda()

        dco_predict = self.modality_classifier(specific_feature)

        m1_predict = self.unimodal_target_classifier(m1_feature_specific_cache_pooling)
        m2_predict = self.unimodal_target_classifier(m2_feature_specific_cache_pooling)
        m3_predict = self.unimodal_target_classifier(m3_feature_specific_cache_pooling)

        return target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m3_feature_share_cache, m1_predict, m2_predict, m3_predict

class ShaSpec_Classfication_TRI_DGD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()

        self.modality_encoder_1 = Couple_CNN(modality_1_channel)
        self.modality_encoder_2 = Couple_CNN(modality_2_channel)
        self.modality_encoder_3 = Couple_CNN(modality_3_channel)

        self.share_encoder = Couple_CNN(modality_1_channel)
        self.modality_projector_2 = torch.nn.Conv2d(modality_2_channel, modality_1_channel,
                                                    1, 1,
                                                    0)

        self.modality_projector_3 = torch.nn.Conv2d(modality_3_channel, modality_1_channel,
                                                    1, 1,
                                                    0)
        self.fusion_projector = nn.Conv2d(256, 128, 1, 1, 0)
        self.fusion = fusion_module(384)
        self.modality_2_transfer = fusion_module(128)
        self.target_classifier = nn.Linear(64, args.class_num)
        self.unimodal_target_classifier = nn.Linear(128, args.class_num)
        self.modality_classifier = nn.Linear(128, 3)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.p = args.p
        self.args = args

    def forward(self, modality1, modality2, modality3):
        m1_feature_specific = self.modality_encoder_1(modality1)
        m2_feature_specific = self.modality_encoder_2(modality2)
        m3_feature_specific = self.modality_encoder_3(modality3)

        m1_feature_specific_cache = m1_feature_specific
        m2_feature_specific_cache = m2_feature_specific
        m3_feature_specific_cache = m3_feature_specific

        modality_2_transfer = self.modality_projector_2(modality2)
        modality_3_transfer = self.modality_projector_3(modality3)
        m1_feature_share = self.share_encoder(modality1)
        m2_feature_share = self.share_encoder(modality_2_transfer)
        m3_feature_share = self.share_encoder(modality_3_transfer)

        m1_feature_share_cache = m1_feature_share
        m2_feature_share_cache = m2_feature_share
        m3_feature_share_cache = m3_feature_share

        [m1_feature_specific, m2_feature_specific, m3_feature_specific], p = modality_drop(
            [m1_feature_specific, m2_feature_specific, m3_feature_specific],
            self.p, args=self.args)
        data = [m1_feature_share, m2_feature_share, m3_feature_share]
        for index in range(len(data)):
            data[index] = data[index] * p[:, index]

        # padding share representation for missing modality

        [m1_feature_share, m2_feature_share, m3_feature_share] = data

        m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
        m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
        m3_missing_index = [not bool(i) for i in (torch.sum(m3_feature_share, dim=[1, 2, 3]))]

        # print(((~(
        #         torch.tensor(m2_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        #
        # print(m1_missing_index)
        # print(m1_feature_share[m1_missing_index] )

        m1_feature_share[m1_missing_index] = (m2_feature_share[m1_missing_index] + m3_feature_share[
            m1_missing_index])
        m1_c = (((~(torch.tensor(m2_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        m1_c = m1_c.unsqueeze(1)
        m1_c = m1_c.unsqueeze(2)
        m1_c = m1_c.unsqueeze(3)
        m1_c = m1_c.repeat(1, 128, 4, 4).cuda()
        m1_feature_share[m1_missing_index] = m1_feature_share[m1_missing_index] / m1_c[m1_missing_index]

        # m1_feature_share=m1_feature_share/

        # / (((~(
        # torch.tensor(m2_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1)[m1_missing_index])  # 1 0 --1   0 0 2

        m2_feature_share[m2_missing_index] = (m1_feature_share[m2_missing_index] + m3_feature_share[
            m2_missing_index])
        m2_c = (((~(torch.tensor(m1_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1))
        m2_c = m2_c.unsqueeze(1)
        m2_c = m2_c.unsqueeze(2)
        m2_c = m2_c.unsqueeze(3)
        m2_c = m2_c.repeat(1, 128, 4, 4).cuda()
        m2_feature_share[m2_missing_index] = m2_feature_share[m2_missing_index] / m2_c[m2_missing_index]

        # / ((~(torch.tensor(m1_missing_index) ^ torch.tensor(m3_missing_index))).float() + 1)[m2_missing_index]

        m3_feature_share[m3_missing_index] = (m1_feature_share[m3_missing_index] + m3_feature_share[
            m3_missing_index])
        m3_c = (((~(torch.tensor(m1_missing_index) ^ torch.tensor(m2_missing_index))).float() + 1))
        m3_c = m3_c.unsqueeze(1)
        m3_c = m3_c.unsqueeze(2)
        m3_c = m3_c.unsqueeze(3)
        m3_c = m3_c.repeat(1, 128, 4, 4).cuda()
        m3_feature_share[m3_missing_index] = m3_feature_share[m3_missing_index] / m3_c[m3_missing_index]

        # fusion

        m1_fusion_out = self.fusion_projector(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
        m1_fusion_out = m1_fusion_out + m1_feature_specific

        m2_fusion_out = self.fusion_projector(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
        m2_fusion_out = m2_fusion_out + m2_feature_specific

        m3_fusion_out = self.fusion_projector(torch.cat((m3_feature_specific, m3_feature_share), dim=1))
        m3_fusion_out = m3_fusion_out + m3_feature_specific

        fusion_feature = self.fusion(torch.cat([m1_fusion_out, m2_fusion_out, m3_fusion_out], dim=1))

        fusion_feature = self.pooling(fusion_feature)
        fusion_feature = fusion_feature.view(fusion_feature.shape[0], -1)

        # calculate loss

        target_predict = self.target_classifier(fusion_feature)

        m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
        m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)
        m3_feature_specific_cache_pooling = self.pooling(m3_feature_specific_cache)

        m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
            m1_feature_specific_cache_pooling.shape[0],
            -1)
        m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
            m2_feature_specific_cache_pooling.shape[0],
            -1)

        m3_feature_specific_cache_pooling = m3_feature_specific_cache_pooling.view(
            m3_feature_specific_cache_pooling.shape[0],
            -1)

        specific_feature = torch.cat(
            (m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling, m3_feature_specific_cache_pooling),
            dim=0)
        specific_feature_label = torch.cat(
            [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0]) + 1], dim=0).long().cuda()

        dco_predict = self.modality_classifier(specific_feature)

        m1_predict = self.unimodal_target_classifier(m1_feature_specific_cache_pooling)
        m2_predict = self.unimodal_target_classifier(m2_feature_specific_cache_pooling)
        m3_predict = self.unimodal_target_classifier(m3_feature_specific_cache_pooling)

        return target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m3_feature_share_cache,fusion_feature, m1_predict, m2_predict, m3_predict
