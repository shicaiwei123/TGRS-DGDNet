import torch.nn as nn
import torchvision.models as tm
import torch

import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from models.base_model import MDMB_extract, MDMB_fusion, Couple_CNN, CCR, MDMB_fusion_middle, MDMB_fusion_share, \
    MDMB_fusion_spp, MDMB_fusion_baseline, MDMB_fusion_dad, MDMB_fusion_dropout
from lib.model_develop_utils import modality_drop
from lib.PositionalEncoding import LearnedPositionalEncoding
from lib.Transformer import mmTransformerModel


class HSI_Lidar_Couple_Former(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_share(256, args)
        self.p = args.p
        self.args = args

        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.embedding_dim = args.embemdding_dim
        self.seq_length = 4 * 4
        self.dropout_rate = 0.1

        self.linear_project = []
        self.position_encoding = []
        self.pe_dropout = []
        self.intra_transformer = []
        self.restore = []
        self.bns = []
        self.relus = []

        for i in range(2):
            self.bns.append(nn.BatchNorm2d(128))
            self.relus.append(nn.LeakyReLU())
            self.linear_project.append(nn.Conv2d(128, args.embemdding_dim, kernel_size=3, stride=1, padding=1))
            self.restore.append(nn.Conv2d(args.embemdding_dim, 128, kernel_size=3, stride=1, padding=1))
            # self.shadow_tokens.append(nn.Parameter(torch.zeros(1, 512, 512)).cuda())
            self.position_encoding.append(LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            ))

            self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
            self.intra_transformer.append(
                mmTransformerModel(modal_num=3, dim=self.embedding_dim, depth=1, heads=8, mlp_dim=4096))

        self.bns = nn.ModuleList(self.bns)
        self.relus = nn.ModuleList(self.relus)
        self.linear_project = nn.ModuleList(self.linear_project)
        self.position_encoding = nn.ModuleList(self.position_encoding)
        self.pe_dropout = nn.ModuleList(self.pe_dropout)
        self.intra_transformer = nn.ModuleList(self.intra_transformer)
        self.restore = nn.ModuleList(self.restore)

        self.auxi_bone = MDMB_fusion_share(128, args)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        hsi_save = x_hsi
        lidar_save = x_lidar

        [x_hsi, x_lidar], p = modality_drop([x_hsi, x_lidar], self.p, args=self.args)

        x = [x_hsi, x_lidar]

        for i in range(2):
            x[i] = self.bns[i](x[i])
            x[i] = self.relus[i](x[i])
            x[i] = self.linear_project[i](x[i])

        for i in range(2):
            x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(x[i].size(0), -1, self.embedding_dim)
            x[i] = self.position_encoding[i](x[i])
            x[i] = self.pe_dropout[i](x[i])
            x[i] = self.intra_transformer[i](x[i])
            x[i] = self._reshape_output(x[i])
            x[i] = self.restore[i](x[i])
        x_hsi = x[0]
        x_lidar = x[1]

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x, feature = self.share_bone(x)

        hsi_out, _ = self.auxi_bone(hsi_save)
        lidar_out, _ = self.auxi_bone(lidar_save)

        return x, feature, p, hsi_out, lidar_out

    def _reshape_output(self, x):
        x = x.view(x.size(0), 4, 4, self.embedding_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class HSI_Lidar_Couple_Former_TRI(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()

        self.modality_1 = Couple_CNN(input_channel=modality_1_channel)
        self.modality_2 = Couple_CNN(input_channel=modality_2_channel)
        self.modality_3 = Couple_CNN(input_channel=modality_3_channel)

        self.share_bone = MDMB_fusion_share(384, args)
        self.p = args.p
        self.args = args

        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args
        self.embedding_dim = args.embemdding_dim
        self.seq_length = 4 * 4
        self.dropout_rate = 0.1

        self.linear_project = []
        self.position_encoding = []
        self.pe_dropout = []
        self.intra_transformer = []
        self.restore = []
        self.bns = []
        self.relus = []

        for i in range(3):
            self.bns.append(nn.BatchNorm2d(128))
            self.relus.append(nn.LeakyReLU())
            self.linear_project.append(nn.Conv2d(128, args.embemdding_dim, kernel_size=3, stride=1, padding=1))
            self.restore.append(nn.Conv2d(args.embemdding_dim, 128, kernel_size=3, stride=1, padding=1))
            # self.shadow_tokens.append(nn.Parameter(torch.zeros(1, 512, 512)).cuda())
            self.position_encoding.append(LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            ))

            self.pe_dropout.append(nn.Dropout(p=self.dropout_rate))
            self.intra_transformer.append(
                mmTransformerModel(modal_num=3, dim=self.embedding_dim, depth=1, heads=8, mlp_dim=4096))

        self.bns = nn.ModuleList(self.bns)
        self.relus = nn.ModuleList(self.relus)
        self.linear_project = nn.ModuleList(self.linear_project)
        self.position_encoding = nn.ModuleList(self.position_encoding)
        self.pe_dropout = nn.ModuleList(self.pe_dropout)
        self.intra_transformer = nn.ModuleList(self.intra_transformer)
        self.restore = nn.ModuleList(self.restore)

        self.auxi_bone = MDMB_fusion_share(128, args)

    def forward(self, m1, m2, m3):
        x_m1 = self.modality_1(m1)
        x_m2 = self.modality_2(m2)
        x_m3 = self.modality_3(m3)

        m1_save = x_m1
        m2_save = x_m2
        m3_save = x_m3

        [x_m1, x_m2, x_m3], p = modality_drop([x_m1, x_m2, x_m3], self.p, args=self.args)

        x = [x_m1, x_m2, x_m3]

        for i in range(3):
            x[i] = self.bns[i](x[i])
            x[i] = self.relus[i](x[i])
            x[i] = self.linear_project[i](x[i])

        for i in range(3):
            x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            x[i] = x[i].view(x[i].size(0), -1, self.embedding_dim)
            x[i] = self.position_encoding[i](x[i])
            x[i] = self.pe_dropout[i](x[i])
            x[i] = self.intra_transformer[i](x[i])
            x[i] = self._reshape_output(x[i])
            x[i] = self.restore[i](x[i])
        x_m1 = x[0]
        x_m2 = x[1]
        x_m3 = x[2]

        x = torch.cat((x_m1, x_m2, x_m3), dim=1)
        x, feature = self.share_bone(x)

        m1_out, _ = self.auxi_bone(m1_save)
        m2_out, _ = self.auxi_bone(m2_save)
        m3_out, _ = self.auxi_bone(m3_save)

        return x, feature, p, m1_out, m2_out, m3_out

    def _reshape_output(self, x):
        x = x.view(x.size(0), 4, 4, self.embedding_dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

