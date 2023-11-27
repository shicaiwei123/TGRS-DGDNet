import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import os
from lib.processing_utils import read_txt, get_file_list, load_mat
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as tt
import scipy.io as scio


class LCZ_multi(Dataset):
    def __init__(self, modality_path_1, modality_path_2, label_path, args, data_transform=None, isdict=True):
        self.modality_1 = load_mat(modality_path_1)
        self.modality_2 = load_mat(modality_path_2)
        self.label_data = load_mat(label_path)
        self.data_transform = data_transform
        self.isdict = isdict
        self.args = args
        self.label_dict = {'2': 0, '4': 1, '5': 2, '6': 3, '8': 4, '11': 5, '12': 6, '13': 7, '14': 8, '17': 9}

    def __len__(self):
        dataset_len = self.modality_1.shape[0]
        return dataset_len

    def __getitem__(self, idx):

        modality_data_1 = self.modality_1[idx, :]
        modality_1 = np.reshape(modality_data_1, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_data_2 = self.modality_2[idx, :]
        modality_2 = np.reshape(modality_data_2, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_label = self.label_data[idx]
        modality_label = int(modality_label)
        # print(modality_label)
        modality_label = self.label_dict[str(modality_label)]

        sample = {"m_1": modality_1, "m_2": modality_2, "label": modality_label}
        if self.data_transform is not None:
            sample = self.data_transform(sample)

        if self.isdict:
            return sample
        else:
            return sample["m_1"], sample["m_2"], sample["label"]



class LCZ_single(Dataset):
    def __init__(self, modality_path_1, label_path, args, data_transform=None, isdict=True):
        self.modality_1 = load_mat(modality_path_1)
        self.label_data = load_mat(label_path)
        self.data_transform = data_transform
        self.isdict = isdict
        self.args = args
        self.label_dict = {'2': 0, '4': 1, '5': 2, '6': 3, '8': 4, '11': 5, '12': 6, '13': 7, '14': 8, '17': 9}


    def __len__(self):
        return self.modality_1.shape[0]

    def __getitem__(self, idx):
        modality_data_1 = self.modality_1[idx, :]
        modality_1 = np.reshape(modality_data_1, (self.args.patch_size, self.args.patch_size, -1), order='F')
        modality_label = int(self.label_data[idx])
        # print(modality_label)
        modality_label = self.label_dict[str(modality_label)]
        # print(modality_label)

        if self.isdict:
            sample = {"m_1": modality_1, "label": modality_label}
            if self.data_transform is not None:
                sample = self.data_transform(sample)
            return sample
        else:

            if self.data_transform is not None:
                modality_1 = self.data_transform(modality_1)
            return modality_1, modality_label
