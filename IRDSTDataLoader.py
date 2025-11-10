import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from numpy import *
import numpy as np
import scipy.io as scio
import cv2
import glob


def extract_numbers(filename):
    import re
    """从文件名中提取多个数字并返回元组，用于排序"""
    # 提取文件名（不含路径）
    filename = os.path.basename(filename)
    # 使用正则表达式提取括号内的数字
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))  # 转换为整数用于排序
    return 0  # 如果没有匹配的括号数字，返回0


def PadImg(img, times):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img


def random_crop_seq(img_seq, mask_seq, patch_size):
    img_seq, mask_seq = img_seq[:, 0], mask_seq[:, 0]  # t,h,w
    _, h, w = img_seq.shape
    if min(h, w) < patch_size:
        for i in range(len(img_seq)):
            img_seq[i, :, :] = np.pad(img_seq[i, :, :],
                                      ((0, 0), (0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                                      mode='constant')
            mask_seq[i, :, :] = np.pad(mask_seq[i, :, :],
                                       ((0, 0), (0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                                       mode='constant')
            _, h, w = img_seq.shape

    # cur_prob = random.random()

    if mask_seq.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask_seq > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0]) - 1)
        h_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], h - patch_size))
        w_start = random.randint(max(0, loc[2][idx] - patch_size), min(loc[2][idx], w - patch_size))

    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch_seq = img_seq[:, h_start:h_end, w_start:w_end]
    mask_patch_seq = mask_seq[:, h_start:h_end, w_start:w_end]

    return np.expand_dims(img_patch_seq, axis=1), np.expand_dims(mask_patch_seq, axis=1)


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
        M[1, 2] += dy  #w/o LSA
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

    def __call__(self, input, target):  # t,c,h,w
        if random.random() < 0.5:
            input = input[:, :, ::-1, :]
            target = target[:, :, ::-1, :]
        if random.random() < 0.5:
            input = input[:, :, :, ::-1]
            target = target[:, :, :, ::-1]
        if random.random() < 0.5:
            input = input[::-1, :, :, :]
            target = target[::-1, :, :, :]
        if random.random() < 0.5:
            input = input.transpose(0, 1, 3, 2)
            target = target.transpose(0, 1, 3, 2)
        if self.aug_large and random.random() < 0.1:  # 20%的概率应用大位移  # 修改0.5原0.2
            # """随机挑选一帧应用大位移"""
            # # 随机挑选帧索引（假设总共 5 帧）
            # rand_idx = random.randint(0, input.shape[0] - 1)
            #
            # sel_frame = input[rand_idx, :, :, :]
            # sel_mask = target[rand_idx, :, :, :]
            #
            # shifted_img, shifted_mask = self.apply_global_shift(sel_frame, sel_mask, max_shift=100)
            #
            # input[rand_idx, :, :, :] = torch.from_numpy(shifted_img)
            # target[rand_idx, :, :, :] = torch.from_numpy(shifted_mask)

            # 获取关键帧
            if self.key_mode == 'mid':
                last_frame = input[2, :, :, :]
                last_mask = target[2, :, :, :]
            else:  # 'last'
                last_frame = input[-1, :, :, :]
                last_mask = target[-1, :, :, :]
            # 应用位移
            shifted_img, shifted_mask = self.apply_global_shift(last_frame, last_mask, max_shift=100)
            if self.key_mode == 'mid':
                input[2, :, :, :] = torch.from_numpy(shifted_img)
                target[2, :, :, :] = torch.from_numpy(shifted_mask)
            else:
                input[-1, :, :, :] = torch.from_numpy(shifted_img)
                target[-1, :, :, :] = torch.from_numpy(shifted_mask)

            # # --- 以50%概率随机挑选一个历史帧增强 ---
            # if random.random() < 0.5:
            #     rand_idx = random.randint(0, input.shape[0] - 2)  # 历史帧索引范围：[0,3]
            #     hist_frame = input[rand_idx, :, :, :]
            #     hist_mask = target[rand_idx, :, :, :]
            #
            #     shifted_img, shifted_mask = self.apply_global_shift(hist_frame, hist_mask, max_shift=100)
            #     input[rand_idx, :, :, :] = torch.from_numpy(shifted_img)
            #     target[rand_idx, :, :, :] = torch.from_numpy(shifted_mask)

        return input, target


# load image
class IRDST_TrainSetLoader(Dataset):
    def __init__(self, root, num_frames=5, fullSupervision=True, key_mode='last'):
        self.num_frames = num_frames
        self.txtpath = root + 'ImageSets/train_new.txt'
        self.img_path = root + 'images/'
        self.mask_path = root + 'masks/'
        self.imgs_arr = []
        self.root = root
        self.fullSupervision = fullSupervision
        self.train_mean = 94.96572
        self.train_std = 37.13109
        # self._transforms = make_train_transform(train_size=train_size)
        self.frames_info = {
            'dataset': {}
        }
        self.tranform = augmentation(aug_large=True, key_mode=key_mode)
        self.key_mode = key_mode

        with open(self.txtpath, 'r') as f:
            video_names = f.readlines()
            video_names = [name.strip() for name in video_names]  # 去除列表中字符串元素首尾空白字符
            print('dataset-train num of videos: {}'.format(len(video_names)))
            for video_name in video_names:
                frames = glob.glob(os.path.join(self.img_path, video_name, '*.png'))
                frames = sorted(frames, key=extract_numbers)
                self.frames_info['dataset'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in
                                                           frames]  # 移除扩展名.png
                self.imgs_arr.extend([('dataset', video_name, frame_index) for frame_index in range(len(frames))])

                # self.frames_info['dataset'][video_name] = [frame_path.split('\\')[-1][:-4] for frame_path in frames]
                # self.img_ids.extend([('dataset', video_name, frame_index) for frame_index in range(len(frames))])

    def __getitem__(self, index):
        img_ids_i = self.imgs_arr[index]
        dataset, video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[dataset][video_name])
        # center_frame_name = self.frames_info[dataset][video_name][frame_index]
        """mid"""
        if self.key_mode == 'mid':
            frame_indices = [(x + vid_len) % vid_len for x in
                             range(frame_index - self.num_frames // 2, frame_index + self.num_frames // 2 + 1, 1)]
        else:
            frame_indices = [(x + vid_len) % vid_len for x in
                             range(frame_index - self.num_frames + 1, frame_index + 1, 1)]
        """last"""

        assert len(frame_indices) == self.num_frames

        frame_ids = []
        img_list = []
        mask_list = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)

            img_path = os.path.join(self.img_path, video_name, frame_name + '.png')
            gt_path = os.path.join(self.mask_path, video_name, frame_name + '.png')
            img_i = Image.open(img_path).convert('L')
            img_i = np.expand_dims(np.array(img_i, dtype=np.float32), axis=0)
            img_i = (img_i - self.train_mean) / self.train_std
            img_list.append(img_i)

            gt = Image.open(gt_path)
            gt = np.array(gt, dtype=np.float32)
            gt[gt > 0] = 255
            gt = np.expand_dims(gt / 255.0, axis=0)
            mask_list.append(gt)  # c,1,h,w

        # masks = torch.from_numpy(masks_list)
        # imgs = torch.from_numpy(img_list)

        _, h, w = mask_list[-1].shape
        if h == 512 and w == 512:
            # Tgt preprocess
            img_list, mask_list = self.tranform(np.array(img_list), np.array(mask_list))  # t,c,h,w
            img = torch.from_numpy(np.ascontiguousarray(img_list)).float()
            mask = torch.from_numpy(np.ascontiguousarray(mask_list)).float()
            # img = torch.Tensor(img)
            # mask = torch.Tensor(mask).float()
            if self.key_mode == 'mid':
                return img.permute(1, 0, 2, 3), mask[2], h, w, frame_ids[2]
            else:
                return img.permute(1, 0, 2, 3), mask[-1], h, w, frame_ids[-1]  # , mask[-1] frame_ids[-1]

        else:
            # Tgt preprocess
            # mask_pad = np.zeros([512, 512])
            # mask_pad[0:h, 0:w] = mask_list[-1]  # 前4帧都是历史，用来估计最后一帧
            # mask_pad = torch.from_numpy(np.expand_dims(mask_pad, axis=0)).float()
            img_pad = np.zeros([self.num_frames, 1, 512, 512])
            mask_pad = np.zeros([self.num_frames, 1, 512, 512])
            """先增强后pad"""  # pad必须hw一致否则不同样本无法堆叠形成一个batch
            img = np.stack(img_list, axis=0)  # [t,c,h,w]
            mask = np.stack(mask_list, axis=0)
            # img, mask = random_crop_seq(img, mask, patch_size=128)  # crop128
            img_aug, mask_aug = self.tranform(img, mask)  # t,c,h,w

            _, _, actH, actW = img_aug.shape
            img_pad[:, :, 0:actH, 0:actW] = img_aug
            mask_pad[:, :, 0:actH, 0:actW] = mask_aug

            """先pad后增强"""
            # for i in range(self.num_frames):
            #     img_pad[i, 0, 0:h, 0:w] = img_list[i]
            #     mask_pad[i, 0, 0:h, 0:w] = mask_list[i]
            #
            # img_pad, mask_pad = self.tranform(img_pad, mask_pad)
            img_pad = torch.from_numpy(np.ascontiguousarray(img_pad)).float()
            mask_pad = torch.from_numpy(np.ascontiguousarray(mask_pad)).float()

            # img_pad = torch.Tensor(img_pad).float()
            if self.key_mode == 'mid':
                return img_pad.permute(1, 0, 2, 3), mask_pad[2], h, w, frame_ids[2]  # mask_pad[2]
            else:
                return img_pad.permute(1, 0, 2, 3), mask_pad[-1], h, w, frame_ids[-1]

    def __len__(self):
        return len(self.imgs_arr)


class IRDST_TestSetLoader(Dataset):
    def __init__(self, root, num_frames=5, fullSupervision=True, key_mode='last'):
        self.num_frames = num_frames
        self.txtpath = root + 'ImageSets/val_new.txt'  # # val_snr_smaller_than_3.txt  val_new.txt
        self.img_path = root + 'images/'
        self.mask_path = root + 'masks/'
        self.imgs_arr = []
        self.root = root
        self.fullSupervision = fullSupervision
        self.test_mean = 94.96572
        self.test_std = 37.13109
        # self._transforms = make_train_transform(train_size=train_size)
        self.frames_info = {
            'dataset': {}
        }
        self.key_mode = key_mode

        with open(self.txtpath, 'r') as f:
            video_names = f.readlines()
            video_names = [name.strip() for name in video_names]  # 去除列表中字符串元素首尾空白字符
            print('dataset-test num of videos: {}'.format(len(video_names)))
            for video_name in video_names:
                frames = glob.glob(os.path.join(self.img_path, video_name, '*.png'))
                frames = sorted(frames, key=extract_numbers)
                self.frames_info['dataset'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
                self.imgs_arr.extend([('dataset', video_name, frame_index) for frame_index in range(len(frames))])

    def __getitem__(self, index):
        img_ids_i = self.imgs_arr[index]
        dataset, video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[dataset][video_name])
        # center_frame_name = self.frames_info[dataset][video_name][frame_index]
        """mid"""
        if self.key_mode == 'mid':
            frame_indices = [(x + vid_len) % vid_len for x in
                             range(frame_index - self.num_frames // 2, frame_index + self.num_frames // 2 + 1, 1)]
        else:
            frame_indices = [(x + vid_len) % vid_len for x in
                             range(frame_index - self.num_frames + 1, frame_index + 1, 1)]
        """last"""
        assert len(frame_indices) == self.num_frames

        frame_ids = []
        img_list = []
        mask_list = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)

            img_path = os.path.join(self.img_path, video_name, frame_name + '.png')
            gt_path = os.path.join(self.mask_path, video_name, frame_name + '.png')
            img_i = Image.open(img_path).convert('L')
            img_i = np.expand_dims(np.array(img_i, dtype=np.float32), axis=0)
            img_i = (img_i - self.test_mean) / self.test_std
            img_list.append(torch.Tensor(img_i))

            gt = Image.open(gt_path)
            gt = np.array(gt, dtype=np.float32)
            gt[gt > 0] = 255
            gt = np.expand_dims(gt / 255.0, axis=0)
            mask_list.append(torch.Tensor(gt))  # c,1,h,w

        # masks = torch.from_numpy(masks_list)
        # imgs = torch.from_numpy(img_list)

        _, h, w = mask_list[-1].shape
        if h == 512 and w == 512:
            # Tgt preprocess
            img = torch.Tensor(img_list).float()
            mask = torch.Tensor(mask_list).float()
            if self.key_mode == 'mid':
                return img.permute(1, 0, 2, 3), mask[2], h, w, frame_ids[2]
            else:
                return img.permute(1, 0, 2, 3), mask[-1], h, w, frame_ids[-1]  # , mask[-1] frame_ids[-1]

        else:
            # Tgt preprocess

            mask_pad = np.zeros([512, 512])
            if self.key_mode == 'mid':
                mask_pad[0:h, 0:w] = mask_list[2]  # 前4帧都是历史，用来估计最后一帧
            else:
                mask_pad[0:h, 0:w] = mask_list[-1]  # 前4帧都是历史，用来估计最后一帧
            mask_pad = np.expand_dims(mask_pad, axis=0)
            img_pad = torch.zeros([self.num_frames, 1, 512, 512])
            # img_pad = []
            for i in range(self.num_frames):
                # pad512or256
                img_pad[i, 0, 0:h, 0:w] = img_list[i]
                # pad32
                # img = PadImg(img_list[i][0], 32)
                # img_pad.append(np.expand_dims(img, axis=0))

            # img_pad = np.stack(img_pad, axis=0)  # [t,c,h,w]
            # mask_pad = PadImg(mask_list[-1][0], 32)
            # mask_pad = torch.from_numpy(np.expand_dims(mask_pad, axis=0)).float()

            # img_pad = torch.Tensor(img_pad).float()
            img_pad = torch.from_numpy(np.ascontiguousarray(img_pad)).float()
            mask_pad = torch.from_numpy(np.ascontiguousarray(mask_pad)).float()
            if self.key_mode == 'mid':
                return img_pad.permute(1, 0, 2, 3), mask_pad, h, w, video_name, frame_ids[2]
            else:
                return img_pad.permute(1, 0, 2, 3), mask_pad, h, w, video_name, frame_ids[-1]

    def __len__(self):
        return len(self.imgs_arr)


if __name__ == '__main__':
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    train_dataset = IRDST_TrainSetLoader('/data/dcy/IRDST/', fullSupervision=True, key_mode='mid')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False,
                              num_workers=0, pin_memory=True)  # 调试时改0

    for i, data in enumerate(train_loader):
        SeqData_t, TgtData_t, m, n, _ = data
