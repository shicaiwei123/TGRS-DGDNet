import torch.nn as nn
import torchvision.models as tm
import torch

from models.base_model import MDMB_extract, MDMB_fusion, Couple_CNN, CCR, MDMB_fusion_middle, MDMB_fusion_share, \
    MDMB_fusion_spp, MDMB_fusion_baseline, MDMB_fusion_dad, MDMB_fusion_dropout, AlexNet, ResNet18, CCR_CNN, WCRN
from lib.model_arch_utils import Flatten, MMTM, SPP, ChannelAttention, SelfAttention
from models.single_modality_model import Single_Modality
from lib.model_develop_utils import modality_drop



class HSI_Lidar_MDMB(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()

        self.special_bone_hsi = MDMB_extract(input_channel=144)
        self.special_bone_lidar = MDMB_extract(input_channel=1)
        self.share_bone = MDMB_fusion(256, 15)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x = self.share_bone(x)
        return x


class HSI_Lidar_Couple(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_modality_1 = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_modality_2 = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion(256, args.class_num)

    def forward(self, modality_1, modality_2):
        x_modality_1 = self.special_bone_modality_1(modality_1)
        x_modality_2 = self.special_bone_modality_2(modality_2)

        x = torch.cat((x_modality_1, x_modality_2), dim=1)
        x_dropout, x = self.share_bone(x)
        return x_dropout, x


class HSI_Lidar_Couple_Baseline(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_modality_1 = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_modality_2 = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_baseline(256, args.class_num)

    def forward(self, modality_1, modality_2):
        x_modality_1 = self.special_bone_modality_1(modality_1)
        x_modality_2 = self.special_bone_modality_2(modality_2)

        x = torch.cat((x_modality_1, x_modality_2), dim=1)
        x = self.share_bone(x)
        return x

class HSI_Lidar_Couple_DAD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_dad(256, args.class_num)

    def forward(self, modality_1, modality_2):
        x_modality_1 = self.special_bone_hsi(modality_1)
        x_modality_2 = self.special_bone_lidar(modality_2)

        x = torch.cat((x_modality_1, x_modality_2), dim=1)
        x_whole, x_feature = self.share_bone(x)
        return x_whole, x_feature


class HSI_Lidar_Couple_Late(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=144)
        self.special_bone_lidar = Couple_CNN(input_channel=1)
        self.share_bone = MDMB_fusion_middle(128, 15)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        x = self.share_bone(x_hsi, x_lidar)
        return x


class HSI_Lidar_Couple_Share(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_share(256, args)
        self.p = args.p
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        hsi_save = x_hsi
        lidar_save = x_lidar

        [x_hsi, x_lidar], p = modality_drop([x_hsi, x_lidar], self.p, args=self.args)

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x, feature = self.share_bone(x)

        x_hsi_out = torch.cat((hsi_save, torch.zeros_like(lidar_save)), dim=1)
        x_hsi_out, _ = self.share_bone(x_hsi_out)

        x_lidar_out = torch.cat((torch.zeros_like(hsi_save), lidar_save), dim=1)
        x_lidar_out, _ = self.share_bone(x_lidar_out)

        return x, feature, p, x_hsi_out, x_lidar_out

        # if 1 in self.args.p:
        #
        #     x = torch.cat((x_hsi, x_lidar), dim=1)
        #     x, feature = self.share_bone(x)
        #
        #     # x_hsi_out=torch.cat((x_hsi,torch.zeros_like(x_lidar)),dim=1)
        #     # x_hsi_out,_=self.share_bone(x_hsi_out)
        #     #
        #     # x_lidar_out=torch.cat((torch.zeros_like(x_hsi),x_lidar),dim=1)
        #     # x_lidar_out,_=self.share_bone(x_lidar_out)
        #
        #     return x, feature, p, x_hsi, x_lidar
        # else:
        #
        #     x = torch.cat((x_hsi, x_lidar), dim=1)
        #     x, feature = self.share_bone(x)
        #     return x, feature, p, x_hsi, x_lidar

class HSI_Lidar_Couple_Share_inter_modality(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_share(256, args)
        self.p = args.p
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        hsi_save = x_hsi
        lidar_save = x_lidar


        return x_hsi,x_lidar

        # if 1 in self.args.p:
        #
        #     x = torch.cat((x_hsi, x_lidar), dim=1)
        #     x, feature = self.share_bone(x)
        #
        #     # x_hsi_out=torch.cat((x_hsi,torch.zeros_like(x_lidar)),dim=1)
        #     # x_hsi_out,_=self.share_bone(x_hsi_out)
        #     #
        #     # x_lidar_out=torch.cat((torch.zeros_like(x_hsi),x_lidar),dim=1)
        #     # x_lidar_out,_=self.share_bone(x_lidar_out)
        #
        #     return x, feature, p, x_hsi, x_lidar
        # else:
        #
        #     x = torch.cat((x_hsi, x_lidar), dim=1)
        #     x, feature = self.share_bone(x)
        #     return x, feature, p, x_hsi, x_lidar



class HSI_Lidar_Couple_General(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        if args.backbone == 'MDL':
            self.modality_1 = Couple_CNN(input_channel=modality_1_channel)
            self.modality_2 = Couple_CNN(input_channel=modality_2_channel)
            output_channel = 128
        elif args.backbone == 'ALEX':
            self.modality_1 = AlexNet(input_channel=modality_1_channel)
            self.modality_2 = AlexNet(input_channel=modality_2_channel)
            output_channel = 256
        elif args.backbone == 'RESNET':
            self.modality_1 = ResNet18(input_channel=modality_1_channel)
            self.modality_2 = ResNet18(input_channel=modality_2_channel)
            output_channel = 128
        else:
            raise Exception('invalid backbone')

        if args.fusion_method == 'share':
            self.share_bone = MDMB_fusion_share(output_channel * 2, args)
        elif args.fusion_method == 'middle':
            self.share_bone = MDMB_fusion_middle(output_channel, args)
        else:
            raise Exception('invalid fusion')

        self.p = args.p
        self.args = args

    def forward(self, m1, m2):
        x_1 = self.modality_1(m1)
        x_2 = self.modality_2(m2)

        x1_save = x_1
        x2_save = x_2

        if self.args.fusion_method=='share':

            [x_hsi, x_lidar], p = modality_drop([x_1, x_2], self.p, args=self.args)

            x = torch.cat((x_hsi, x_lidar), dim=1)
            x, feature = self.share_bone(x)

            x_hsi_out = torch.cat((x1_save, torch.zeros_like(x2_save)), dim=1)
            x_hsi_out, _ = self.share_bone(x_hsi_out)

            x_lidar_out = torch.cat((torch.zeros_like(x1_save), x2_save), dim=1)
            x_lidar_out, _ = self.share_bone(x_lidar_out)

            return x, feature, p, x_hsi_out, x_lidar_out
        elif self.args.fusion_method=='middle':
            [x_hsi, x_lidar], p = modality_drop([x_1, x_2], self.p, args=self.args)

            x, feature = self.share_bone(x_hsi,x_lidar)

            x_hsi_out, _ = self.share_bone(x1_save,torch.zeros_like(x2_save))

            x_lidar_out, _ = self.share_bone(torch.zeros_like(x1_save),x2_save)

            return x, feature, p, x_hsi_out, x_lidar_out


class HSI_Lidar_Couple_Share_Auxi(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_share(256, args)
        self.auxi_bone = MDMB_fusion_share(128, args)
        self.p = args.p
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        hsi_save = x_hsi
        lidar_save = x_lidar

        [x_hsi, x_lidar], p = modality_drop([x_hsi, x_lidar], self.p, args=self.args)

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x, feature = self.share_bone(x)

        x_hsi_out, _ = self.auxi_bone(hsi_save)

        x_lidar_out, _ = self.auxi_bone(lidar_save)

        return x, feature, p, x_hsi_out, x_lidar_out

        # if 1 in self.args.p:
        #
        #     x = torch.cat((x_hsi, x_lidar), dim=1)
        #     x, feature = self.share_bone(x)
        #
        #     # x_hsi_out=torch.cat((x_hsi,torch.zeros_like(x_lidar)),dim=1)
        #     # x_hsi_out,_=self.share_bone(x_hsi_out)
        #     #
        #     # x_lidar_out=torch.cat((torch.zeros_like(x_hsi),x_lidar),dim=1)
        #     # x_lidar_out,_=self.share_bone(x_lidar_out)
        #
        #     return x, feature, p, x_hsi, x_lidar
        # else:
        #
        #     x = torch.cat((x_hsi, x_lidar), dim=1)
        #     x, feature = self.share_bone(x)
        #     return x, feature, p, x_hsi, x_lidar


class HSI_Lidar_Couple_Share_TRI_Auxi(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()

        self.modality_1 = Couple_CNN(input_channel=modality_1_channel)
        self.modality_2 = Couple_CNN(input_channel=modality_2_channel)
        self.modality_3 = Couple_CNN(input_channel=modality_3_channel)
        self.share_bone = MDMB_fusion_share(384, args.class_num)
        self.auxi_bone = MDMB_fusion_share(128, args.class_num)
        self.p = args.p
        self.args = args

    def forward(self, m1, m2, m3):
        x_1 = self.modality_1(m1)
        x_2 = self.modality_2(m2)
        x_3 = self.modality_3(m3)

        x1_save = x_1
        x2_save = x_2
        x3_save = x_3

        [x1, x2, x3], p = modality_drop([x_1, x_2, x_3], self.p, args=self.args)
        # print(p)

        x1_out, _ = self.auxi_bone(x1_save)

        x2_out, _ = self.auxi_bone(x2_save)

        x3_out, _ = self.auxi_bone(x3_save)

        x = torch.cat((x1, x2, x3), dim=1)
        x, feature = self.share_bone(x)
        return x, feature, p, x1_out, x2_out, x3_out


class HSI_Lidar_Couple_Share_Unimodal_Center(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_share(256, args.class_num)
        self.p = args.p
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        p=self.p


        x = torch.cat((x_hsi, x_lidar), dim=1)
        x, feature = self.share_bone(x)

        x_hsi_out = torch.cat((x_hsi, torch.zeros_like(x_lidar)), dim=1)
        x_hsi_out, _ = self.share_bone(x_hsi_out)

        x_lidar_out = torch.cat((torch.zeros_like(x_hsi), x_lidar), dim=1)
        x_lidar_out, _ = self.share_bone(x_lidar_out)

        return x, feature, p, x_hsi_out, x_lidar_out


class HSI_Lidar_Couple_Share_TRI(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()

        self.modality_1 = Couple_CNN(input_channel=modality_1_channel)
        self.modality_2 = Couple_CNN(input_channel=modality_2_channel)
        self.modality_3 = Couple_CNN(input_channel=modality_3_channel)
        self.share_bone = MDMB_fusion_share(384, args)
        self.p = args.p
        self.args = args

    def forward(self, m1, m2, m3):
        x_1 = self.modality_1(m1)
        x_2 = self.modality_2(m2)
        x_3 = self.modality_3(m3)

        x1_save = x_1
        x2_save = x_2
        x3_save = x_3

        [x1, x2, x3], p = modality_drop([x_1, x_2, x_3], self.p, args=self.args)
        # print(p)

        x1_out = torch.cat([x1_save, torch.zeros_like(x2_save), torch.zeros_like(x3_save)], dim=1)
        x1_out, _ = self.share_bone(x1_out)

        x2_out = torch.cat([torch.zeros_like(x1_save), x2_save, torch.zeros_like(x3_save)], dim=1)
        x2_out, _ = self.share_bone(x2_out)

        x3_out = torch.cat([torch.zeros_like(x1_save), torch.zeros_like(x2_save), x3_save], dim=1)
        x3_out, _ = self.share_bone(x3_out)

        x = torch.cat((x1, x2, x3), dim=1)
        x, feature = self.share_bone(x)
        return x, feature, p, x1_out, x2_out, x3_out


class HSI_Lidar_Couple_Cross(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(

            nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lidar_block_1 = nn.Sequential(

            nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.share_bone = MDMB_fusion(256, args.class_num)

    def forward(self, hsi, lidar):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)

        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_hsi_lidar = self.lidar_block_2(hsi)
        x_lidar_hsi = self.hsi_block_2(lidar)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi) / 2, (x_lidar + x_hsi_lidar) / 2), dim=1)
        joint_2 = torch.cat((x_hsi, x_hsi_lidar), dim=1)
        joint_3 = torch.cat((x_lidar_hsi, x_lidar), dim=1)


        x1 = self.share_bone(joint_1)
        x2 = self.share_bone(joint_2)
        x3 = self.share_bone(joint_3)
        # x = (x1 + x2 + x3) / 3
        return x1, x2, x3


class HSI_Lidar_Couple_Cross_General(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        if args.backbone == 'MDL':
            self.modality_1 = Couple_CNN(input_channel=modality_1_channel)
            self.modality_2 = Couple_CNN(input_channel=modality_2_channel)
            output_channel = 128
        elif args.backbone == 'ALEX':
            self.modality_1 = AlexNet(input_channel=modality_1_channel)
            self.modality_2 = AlexNet(input_channel=modality_2_channel)
            output_channel = 256
        elif args.backbone == 'RESNET':
            self.modality_1 = ResNet18(input_channel=modality_1_channel)
            self.modality_2 = ResNet18(input_channel=modality_2_channel)
            output_channel = 256
        else:
            raise Exception('invalid backbone')

        self.share_bone = MDMB_fusion(output_channel * 2, args.class_num)

    def forward(self, m1, m2):
        m1 = self.modality_1.block_1(m1)
        m2 = self.modality_2.block_2(m2)

        m1_save = m1
        m2_save = m2

        [m1, m2], p = modality_drop([m1, m2], self.p, args=self.args)

        x_m1 = self.modality_1.block_2(m1)
        x_m1 = self.modality_1.block_3(x_m1)

        x_m2 = self.modality_2.block_2(m2)
        x_m2 = self.modality_2.block_3(x_m2)

        x_m1_m2 = self.modality_2.block_2(m1)
        x_m1_m2 = self.modality_2.block_3(x_m1_m2)

        x_m2_m1 = self.modality_1.block_2(m2)
        x_m2_m1 = self.modality_1.block_3(x_m2_m1)

        joint_1 = torch.cat(((x_m1 + x_m2_m1) / 2, (x_m2 + x_m1_m2) / 2), dim=1)
        joint_2 = torch.cat((x_m1, x_m1_m2), dim=1)
        joint_3 = torch.cat((x_m2_m1, x_m2), dim=1)
        #

        x1, x_feature = self.share_bone(joint_1)
        x2, _ = self.share_bone(joint_2)
        x3, _ = self.share_bone(joint_3)

        x_hsi_out = torch.cat((m1_save, torch.zeros_like(m2_save)), dim=1)
        x_hsi_out, _ = self.share_bone(x_hsi_out)

        x_lidar_out = torch.cat((torch.zeros_like(m1_save), m2_save), dim=1)
        x_lidar_out, _ = self.share_bone(x_lidar_out)

        # x = (x1 + x2 + x3) / 3
        return x1, x2, x3, x_feature, x_hsi_out, x_lidar_out


class HSI_Lidar_Couple_Cross_DAD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(
            # nn.Conv2d(modality_1_channel, 16, kernel_size=3, stride=1, padding=1,
            #                                        bias=False),
            #                              nn.BatchNorm2d(16),
            #                              nn.ReLU(),
            nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lidar_block_1 = nn.Sequential(
            # nn.Conv2d(modality_2_channel, 16, kernel_size=3, stride=1, padding=1,
            #                                          bias=False),
            #                                nn.BatchNorm2d(16),
            #                                nn.ReLU(),
            nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.share_bone = MDMB_fusion_dad(256, args.class_num)

    def forward(self, hsi, lidar):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)
        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_hsi_lidar = self.lidar_block_2(hsi)
        x_lidar_hsi = self.hsi_block_2(lidar)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi) / 2, (x_lidar + x_hsi_lidar) / 2), dim=1)
        joint_2 = torch.cat((x_hsi, x_hsi_lidar), dim=1)
        joint_3 = torch.cat((x_lidar_hsi, x_lidar), dim=1)
        #
        # joint_1 = torch.cat((x_hsi, x_lidar_hsi), dim=1)
        # joint_2 = torch.cat((x_hsi_lidar, x_lidar), dim=1)
        # joint_3 = torch.cat((x_hsi + x_lidar_hsi, x_lidar + x_hsi_lidar), dim=1)

        x1, x_feature = self.share_bone(joint_1)
        return x1, x_feature


class HSI_Lidar_CCR(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=144)
        self.special_bone_lidar = Couple_CNN(input_channel=1)
        self.share_bone = CCR(256, 15)

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x_origin = torch.cat((x_lidar, x_hsi), dim=1)
        # x_origin = x
        x, x_rec = self.share_bone(x)
        return x, x_origin, x_rec

class HSI_Lidar_Couple_Cross_TRI(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(
            nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lidar_block_1 = nn.Sequential(
            nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dsm_block_1 = nn.Sequential(
            nn.Conv2d(modality_3_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.dsm_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.share_bone = MDMB_fusion(384, args.class_num)
        self.p = args.p
        self.args = args

    def forward(self, hsi, lidar, dsm):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)
        dsm = self.dsm_block_1(dsm)

        [hsi, lidar, dsm], p = modality_drop([hsi, lidar, dsm], p=self.p, args=self.args)

        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_dsm = self.dsm_block_2(dsm)

        x_hsi_lidar = self.lidar_block_2(hsi)
        x_hsi_dsm = self.dsm_block_2(hsi)

        x_lidar_hsi = self.hsi_block_2(lidar)
        x_lidar_dsm = self.dsm_block_2(lidar)

        x_dsm_hsi = self.hsi_block_2(dsm)
        x_dsm_lidar = self.lidar_block_2(dsm)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi + x_dsm_hsi) / 3, (x_lidar + x_hsi_lidar + x_dsm_lidar) / 3,
                             (x_dsm + x_hsi_dsm + x_lidar_dsm) / 3),
                            dim=1)
        joint_2 = torch.cat((x_hsi, x_hsi_lidar, x_hsi_dsm), dim=1)
        joint_3 = torch.cat((x_lidar_hsi, x_lidar, x_dsm), dim=1)
        #
        # joint_1 = torch.cat((x_hsi, x_lidar_hsi), dim=1)
        # joint_2 = torch.cat((x_hsi_lidar, x_lidar), dim=1)
        # joint_3 = torch.cat((x_hsi + x_lidar_hsi, x_lidar + x_hsi_lidar), dim=1)

        x1 = self.share_bone(joint_1)
        x2 = self.share_bone(joint_2)
        x3 = self.share_bone(joint_3)
        # x2 = joint_2
        # x3 = joint_3
        # x = (x1 + x2 + x3) / 3
        return x1


class HSI_Lidar_Couple_Cross_TRI_DAD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, modality_3_channel):
        super().__init__()
        self.hsi_block_1 = nn.Sequential(
            nn.Conv2d(modality_1_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lidar_block_1 = nn.Sequential(
            nn.Conv2d(modality_2_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dsm_block_1 = nn.Sequential(
            nn.Conv2d(modality_3_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.hsi_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.lidar_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                     bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.ReLU(),
                                           )

        self.dsm_block_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         )

        self.share_bone = MDMB_fusion_dad(384, args.class_num)

    def forward(self, hsi, lidar, dsm):
        hsi = self.hsi_block_1(hsi)
        lidar = self.lidar_block_1(lidar)
        dsm = self.dsm_block_1(dsm)

        x_hsi = self.hsi_block_2(hsi)
        x_lidar = self.lidar_block_2(lidar)
        x_dsm = self.dsm_block_2(dsm)

        x_hsi_lidar = self.lidar_block_2(hsi)
        x_hsi_dsm = self.dsm_block_2(hsi)

        x_lidar_hsi = self.hsi_block_2(lidar)
        x_lidar_dsm = self.dsm_block_2(lidar)

        x_dsm_hsi = self.hsi_block_2(dsm)
        x_dsm_lidar = self.lidar_block_2(dsm)

        joint_1 = torch.cat(((x_hsi + x_lidar_hsi + x_dsm_hsi) / 3, (x_lidar + x_hsi_lidar + x_dsm_lidar) / 3,
                             (x_dsm + x_hsi_dsm + x_lidar_dsm) / 3),
                            dim=1)
        # joint_2 = torch.cat((x_hsi, x_hsi_lidar, x_hsi_dsm), dim=1)
        # joint_3 = torch.cat((x_lidar_hsi, x_lidar, x_dsm), dim=1)
        #
        # joint_1 = torch.cat((x_hsi, x_lidar_hsi), dim=1)
        # joint_2 = torch.cat((x_hsi_lidar, x_lidar), dim=1)
        # joint_3 = torch.cat((x_hsi + x_lidar_hsi, x_lidar + x_hsi_lidar), dim=1)

        x1, x_feature = self.share_bone(joint_1)
        return x1, x_feature
