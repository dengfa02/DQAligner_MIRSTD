import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


class AdaFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        """Adaptive parameter adjustment"""

    def forward(self, pred, target):
        # pred = torch.sigmoid(pred)
        # 计算面积自适应权重
        area_weight = self._get_area_weight(target)  # [N,1,1,1]
        smooth = 1

        intersection = pred.sigmoid() * target
        iou = (intersection.sum() + smooth) / (pred.sigmoid().sum() + target.sum() - intersection.sum() + smooth)
        iou = torch.clamp(iou, min=1e-6, max=1 - 1e-6).detach()
        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # 16*1*256*256

        target = target.type(torch.long)
        at = target * area_weight + (1 - target) * (1 - area_weight)
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        F_loss = (1 - pt) ** (1 - iou + 1e-6) * BCE_loss

        F_loss = at * F_loss
        # total_loss = (1 - iou) * F_loss.mean() + iou * iou
        return F_loss.sum()

    def _get_area_weight(self, target):
        # 小目标增强权重
        area = target.sum(dim=(1, 2, 3))  # [N,1]
        return torch.sigmoid(1 - area / (area.max() + 1)).view(-1, 1, 1, 1)

    def adafocal_gradient(self, iou, weight, x):
        sigmoid_x = 1 / (1 + np.exp(-x))  # sigmoid(x)
        term1 = weight * sigmoid_x * (1 - iou) * (1 - sigmoid_x) ** (1 - iou) * np.log(sigmoid_x)
        term2 = weight * (1 - sigmoid_x) ** (2 - iou)
        return term1 - term2


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
