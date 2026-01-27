import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio
import cv2


class augmentation(object):
    def __init__(self, aug_large=True, key_mode='last'):
        self.aug_large = aug_large
        self.key_mode = key_mode

    def apply_global_shift(self, image, mask, max_shift=100):
        """模拟相机抖动产生的整体位移"""
        image, mask = image[0], mask[0]  # [1,h,w] -> [h,w]
        h, w = image.shape[:2]
        # 随机生成较大的位移量
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)
        # 保证位移足够大
        while abs(dx) < 50 and abs(dy) < 50:
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)

        # 旋转角度
        angle = np.random.uniform(-3, 3)
        # 缩放因子
        scale = 1.0 + np.random.uniform(-0.1, 0.1)
        # 计算变换矩阵
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        # # 构建平移矩阵
        # M = np.float32([[1, 0, dx], [0, 1, dy]])
        # 应用平移缩放与旋转
        shifted_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
        shifted_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

        shifted_image = np.expand_dims(shifted_image, axis=0)  # [h,w] -> [1,h,w]
        shifted_mask = np.expand_dims(shifted_mask, axis=0)  # [h,w] -> [1,h,w]

        return shifted_image, shifted_mask

    def __call__(self, input, target):  # input c,t,h,w target c,h,w
        if random.random() < 0.5:
            input = input[:, :, ::-1, :]
            target = target[:, ::-1, :]
        if random.random() < 0.5:
            input = input[:, :, :, ::-1]
            target = target[:, :, ::-1]
        if random.random() < 0.5:
            input = input.transpose(0, 1, 3, 2)
            target = target.transpose(0, 2, 1)
        if self.aug_large and random.random() < 0.2:
            # 获取关键帧
            if self.key_mode == 'mid':
                last_frame = input[2, :, :, :]
                last_mask = target[2, :, :, :]
            else:  # 'last'
                last_frame = input[:, -1, :, :]
                last_mask = target[:, :, :]
            # 应用位移
            shifted_img, shifted_mask = self.apply_global_shift(last_frame, last_mask, max_shift=100)
            if self.key_mode == 'mid':
                input[2, :, :, :] = torch.from_numpy(shifted_img)
                target[2, :, :, :] = torch.from_numpy(shifted_mask)
            else:
                input[:, -1, :, :] = torch.from_numpy(shifted_img)
                target[:, :, :] = torch.from_numpy(shifted_mask)

        return input, target


# load image
class TrainSetLoader(Dataset):
    def __init__(self, root, fullSupervision=False, key_mode='last'):

        txtpath = root + 'train.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.root = root
        self.fullSupervision = fullSupervision
        self.train_mean = 105.4025
        self.train_std = 26.6452
        self.tranform = augmentation(aug_large=False, key_mode=key_mode)
        self.key_mode = key_mode

    def __getitem__(self, index):
        img_path_mix = self.root + self.imgs_arr[index]

        # Read Mix
        MixData_mat = scio.loadmat(img_path_mix)

        MixData_Img = MixData_mat.get('Mix')  # MatData
        MixData_Img = MixData_Img.astype(np.float32)

        # Read Label/Tgt
        img_path_tgt = img_path_mix.replace('.mat', '.png')
        # print(img_path_mix)
        if self.fullSupervision:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
        else:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks_centroid')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0

        # Mix preprocess
        MixData_Img = (MixData_Img - self.train_mean) / self.train_std
        MixData = torch.from_numpy(MixData_Img)

        MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)  # the last five frame c,t,h,w

        [m_L, n_L] = np.shape(LabelData_Img)
        if m_L == 512 and n_L == 512:
            # Tgt preprocess
            LabelData = torch.from_numpy(LabelData_Img)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            return MixData_out, TgtData_out, m_L, n_L

        else:
            # Tgt preprocess
            img_pad = np.zeros([1, 5, 512, 512])
            mask_pad = np.zeros([1, 512, 512])
            LabelData_Img = LabelData_Img[np.newaxis, :, :]
            MixData_out = np.array(MixData_out)

            _, _, actH, actW = MixData_out.shape  # MixData_out, img_aug
            img_pad[:, :, 0:actH, 0:actW] = MixData_out  # img_aug MixData_out
            mask_pad[:, 0:actH, 0:actW] = LabelData_Img  # mask_aug LabelData_Img
            img_pad = torch.from_numpy(np.ascontiguousarray(img_pad)).float()
            mask_pad = torch.from_numpy(np.ascontiguousarray(mask_pad)).float()

            return img_pad, mask_pad, m_L, n_L  # mask_pad[2]

    def __len__(self):
        return len(self.imgs_arr)


class TestSetLoader(Dataset):
    def __init__(self, root, fullSupervision=False):

        txtpath = root + 'test.txt'
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.root = root
        self.fullSupervision = fullSupervision

    def __getitem__(self, index):
        img_path_mix = self.root + self.imgs_arr[index]

        # Read Mix
        MixData_mat = scio.loadmat(img_path_mix)

        MixData_Img = MixData_mat.get('Mix')  # MatData
        MixData_Img = MixData_Img.astype(np.float32)

        # Read Label/Tgt
        img_path_tgt = img_path_mix.replace('.mat', '.png')
        if self.fullSupervision:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
        else:
            img_path_tgt = img_path_tgt.replace('Mix', 'masks_centroid')
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0

        # Mix preprocess
        train_mean = 105.4025
        train_std = 26.6452
        MixData_Img = (MixData_Img - train_mean) / train_std
        MixData = torch.from_numpy(MixData_Img)

        MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)  # the last five frame，unsqueeze(0)表示在第0维增加通道数1

        [m_L, n_L] = np.shape(LabelData_Img)
        if m_L == 512 and n_L == 512:
            # Tgt preprocess
            LabelData = torch.from_numpy(LabelData_Img)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            return MixData_out, TgtData_out, m_L, n_L

        else:
            # Tgt preprocess
            [n, t, m_M, n_M] = shape(MixData_out)
            LabelData_Img_1 = np.zeros([512, 512])
            LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img
            LabelData = torch.from_numpy(LabelData_Img_1)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            MixData_out_1 = torch.zeros([n, t, 512, 512])
            MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out
            return MixData_out_1, TgtData_out, m_L, n_L

    def __len__(self):
        return len(self.imgs_arr)
