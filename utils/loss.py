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


def y_axis_formatter(y, pos):
    if abs(y) < 1e-10:  # 如果接近0（考虑浮点精度）
        return '0'  # 直接显示0.0
    else:
        return f'$e^{{{y:.1f}}}$'  # 其他值显示为e^形式


class EdgeEnhanceLoss(nn.Module):
    def __init__(self, sigma=3, device='cuda'):
        super().__init__()
        self.sigma = sigma  # 边缘扩展范围控制

    def forward(self, pred, target):
        # 生成距离变换图
        target_np = target.detach().cpu().numpy().astype(np.uint8)

        # 批量处理 (假设输入为[N,1,H,W])
        dist_maps = []
        for batch in range(target_np.shape[0]):
            # 计算距离变换（背景到前景边界的距离）
            dist_map = distance_transform_edt(target_np[batch, 0])
            dist_maps.append(dist_map)

            # 转换为PyTorch张量并移至原设备
        dist_map = torch.from_numpy(np.stack(dist_maps)).float().to(self.device)
        edge_weight = torch.exp(-dist_map ** 2 / (2 * self.sigma ** 2))

        # 边缘区域加权交叉熵
        return F.binary_cross_entropy(pred, target, weight=edge_weight)


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def IOU(self, pred, mask):
        smooth = 1

        intersection = pred * mask
        loss = (intersection.sum() + smooth) / (pred.sum() + mask.sum() - intersection.sum() + smooth)

        # intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        # pred_sum = torch.sum(pred, dim=(1, 2, 3))
        # target_sum = torch.sum(mask, dim=(1, 2, 3))
        # loss = (intersection_sum + smooth) / \
        #     (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        return loss

    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        loss_iou = self.IOU(pred, mask)

        return loss_iou


class FocalIoULoss(nn.Module):
    def __init__(self):
        super(FocalIoULoss, self).__init__()

    def forward(self, inputs, targets):
        # [b, c, h, w] = inputs.size()
        inputs = torch.nn.Sigmoid()(inputs)
        inputs = 0.999 * (inputs - 0.5) + 0.5
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        intersection = torch.mul(inputs, targets)
        smooth = 1

        IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)

        alpha = 0.75
        gamma = 2
        gamma = gamma

        pt = torch.exp(-BCE_loss)
        F_loss = torch.mul(((1 - pt) ** gamma), BCE_loss)
        at = targets * alpha + (1 - targets) * (1 - alpha)

        F_loss = (1 - IoU) * (F_loss) ** (IoU * 0.5 + 0.5)
        F_loss_map = at * F_loss
        F_loss_sum = F_loss_map.sum()

        return F_loss_sum


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        alpha: 权重系数，用于平衡正负样本。
        gamma: 调节易分类样本的聚焦参数。
        reduction: 指定返回的损失类型，'none' | 'mean' | 'sum'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # 自带激活16*1*256*256

        targets = targets.type(torch.long)
        at = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        F_loss = at * F_loss
        return F_loss.sum()
        # # 计算交叉熵损失
        # inputs = torch.nn.Sigmoid()(inputs)
        # ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 使用'none'以避免后续的均值或求和操作
        # pt = torch.exp(-ce_loss)  # pt是模型输出概率的补数，即1 - p_t
        # loss = ce_loss * ((1 - pt) ** self.gamma)  # 应用focal loss公式
        #
        # if self.alpha >= 0:
        #     alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # 根据标签调整权重
        #     loss = alpha_t * loss  # 应用权重调整
        #
        # if self.reduction == 'mean':
        #     loss = loss.mean()  # 返回均值损失
        # elif self.reduction == 'sum':
        #     loss = loss.sum()  # 返回求和损失
        # return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1

        intersection = pred * target
        # intersection_sum = intersection.sum()
        # pred_sum = torch.sum(pred, dim=(1, 2, 3))
        # target_sum = torch.sum(target, dim=(1, 2, 3))

        loss = (2 * intersection.sum() + smooth) / \
               (pred.sum() + target.sum() + smooth)
        loss = 1 - loss.mean()

        return loss


class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)  # Var

        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w  # 宽度方向上的归一化位置索引
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth))
                                                           - torch.arctan(
            (target_centery) / (target_centerx + smooth))))

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss


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


if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 800  # 图片像素
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    inputs = torch.randn(16, 1, 256, 256)
    target = (torch.rand(16, 1, 256, 256) > 0.9).float()  # 约10%的像素为1
    loss_fun = AdaFocalLoss()
    loss = loss_fun(inputs, target)
    loss_fun.gradient()
    print(loss)

    # u_values = np.linspace(10, 100, 1000)
    # iou_values = np.linspace(0.01, 0.99, 10)
    # # iou_values = [0.1, 0.5, 0.9]
    # # weight_values = [0.5, 0.75, 1.0]
    # y = 1
    # plt.figure(figsize=(10, 8))
    # plt.plot(u_values, -1/u_values, '-', label=f'IoU={iou_values}')
    # for i, iou in enumerate(iou_values):
    #     plt.plot(u_values, iou / u_values, '-', label=f'y=0')
    # plt.xlabel('u', fontsize=20)
    # plt.ylabel('g', fontsize=20)
    # plt.legend(fontsize=16)
    # plt.show()
