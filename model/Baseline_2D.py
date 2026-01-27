# -*- coding: utf-8 -*-
import sys

sys.path.append('../')
from thop import clever_format, profile
import time
from model.dcn.modules.deform_conv import DeformConv
from model.Motion import *
from model.ObjectQuery import *
import os
from torch.nn import init

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=True, act="relu", mode=None):
        super().__init__()
        self.mode = mode
        pad = (ksize - 1) // 2
        if self.mode == 'cdc':
            self.conv = Conv2d_cd(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad,
                                  groups=groups, bias=bias)  # nn.Conv2d
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad,
                                  groups=groups,
                                  bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="relu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,
                              mode=None)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, depthwise=True, act='relu'):
        super(ResNet, self).__init__()
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = Conv(in_channels, out_channels, ksize=3, stride=stride, act=act)
        self.dconv2 = BaseConv(out_channels, out_channels, ksize=3, stride=stride, groups=out_channels, act=act,
                               mode=None)
        self.pconv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, groups=1,
                                              bias=True),
                                    nn.BatchNorm2d(out_channels))  # , eps=0.001, momentum=0.03)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))  # , eps=0.001, momentum=0.03)
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.dconv2(out)
        out = self.pconv2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.act(out)
        return out


class feat_extract(nn.Module):
    def __init__(self, input_channels, block=ResNet):  # ResNet
        super().__init__()
        param_channels = [16, 32, 64, 128]
        param_blocks = [2, 2, 2, 2]
        self.conv_init = nn.Conv3d(input_channels, param_channels[0], (1, 1, 1), stride=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')  # nearest

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block, param_blocks[0])
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block,
                                          param_blocks[1])  # (1, 2, 2)
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block,
                                          param_blocks[2])  # (1, 5, 5)

        self.middle = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[3])

        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block,
                                          param_blocks[2])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block,
                                          param_blocks[1])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block,
                                          param_blocks[0])

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv_init(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(B * T, C, H, W)  # b,c,t,h,w -> bt,c,h,w
        x_e0 = self.encoder_0(x)
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_m = self.middle(self.pool(x_e2))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_m)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        x_d0 = x_d0.reshape(B, T, x_d0.size(1), x_d0.size(2), x_d0.size(3)).permute(0, 2, 1, 3, 4).contiguous()
        x_d1 = x_d1.reshape(B, T, x_d1.size(1), x_d1.size(2), x_d1.size(3)).permute(0, 2, 1, 3, 4).contiguous()
        x_d2 = x_d2.reshape(B, T, x_d2.size(1), x_d2.size(2), x_d2.size(3)).permute(0, 2, 1, 3, 4).contiguous()
        x_m = x_m.reshape(B, T, x_m.size(1), x_m.size(2), x_m.size(3)).permute(0, 2, 1, 3, 4).contiguous()

        return x_d0, x_d1, x_d2, x_m


class SRFAttentionasoffset(nn.Module):
    def __init__(self, in_channel, out_channel, kernels=[1, 3, 5, 7], reduction=8, L=32, dilation_rates=[1, 6, 12, 24]):
        super().__init__()
        self.d = max(L, in_channel // reduction)
        self.RFs = nn.ModuleList([])
        for rate in dilation_rates:
            self.RFs.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding='same', dilation=rate, groups=2),
                    nn.ReLU(inplace=True)
                )
            )
        self.fc = nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2))
        self.softmax = nn.Softmax(dim=0)

        self.project = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.RFs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c

        S = S.unsqueeze(1)  # bs,1,c
        Z = self.fc(S)  # bs,1,c

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)  # bs,1,c
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weights = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weights = self.softmax(attention_weights)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weights * feats).sum(0)  # b,c,h,w
        return self.project(V)


class DFDAasoffset(nn.Module):
    def __init__(self, num_feat=16, deformable_groups=4, train_mode=False):
        super(DFDAasoffset, self).__init__()
        self.ref_conv_l1 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                                         nn.BatchNorm2d(num_feat),
                                         nn.ReLU(inplace=True))
        self.ref_conv_l2 = nn.Sequential(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
                                         nn.BatchNorm2d(num_feat),
                                         nn.ReLU(inplace=True))
        self.ref_conv_l3 = nn.Sequential(nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1),
                                         nn.BatchNorm2d(num_feat),
                                         nn.ReLU(inplace=True))
        self.key_conv_l1 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                                         nn.BatchNorm2d(num_feat),
                                         nn.ReLU(inplace=True))
        self.key_conv_l2 = nn.Sequential(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
                                         nn.BatchNorm2d(num_feat),
                                         nn.ReLU(inplace=True))
        self.key_conv_l3 = nn.Sequential(nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1),
                                         nn.BatchNorm2d(num_feat),
                                         nn.ReLU(inplace=True))

        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        self.query_mask = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            if i == 1:  # 仅在L1有
                self.query_mask[level] = ObjectGuidedEnhancement(feat_dim=num_feat, query_dim=16)
            """ASPP"""
            self.offset_conv1[level] = nn.Sequential(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
                                                     nn.ReLU(inplace=True))
            if i == 3:
                self.offset_conv2[level] = SRFAttentionasoffset(num_feat, num_feat)
            else:
                self.offset_conv2[level] = SRFAttentionasoffset(num_feat * 2, num_feat)
            self.dcn_pack[level] = \
                DeformConv(num_feat,
                           num_feat,
                           3,
                           padding=1,
                           stride=1,
                           deformable_groups=deformable_groups)  # 每组通道学习独立的偏移

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.ReLU(inplace=False)
        self.training_flag = train_mode

    def estimate_global_shift(self, feat_embed, ref_embed):
        b, c, h, w = feat_embed.size()

        ref_flat = ref_embed.view(b, -1, h * w)  # [B, C/r, H*W]
        curr_flat = feat_embed.view(b, -1, h * w)  # [B, C/r, H*W]
        similarity = torch.bmm(ref_flat.transpose(1, 2), curr_flat)  # [B, H*W, H*W]
        matched_indices = similarity.argmax(dim=2)  # [B, H*W]
        curr_indices = torch.arange(h * w, device=feat_embed.device).expand(b, -1)
        curr_y = (curr_indices / w).floor().long()
        curr_x = (curr_indices % w).long()
        matched_y = (matched_indices / w).floor().long()
        matched_x = (matched_indices % w).long()
        dx = matched_x - curr_x  # [B, H*W]
        dy = matched_y - curr_y  # [B, H*W]
        dx = dx.reshape(b, 1, h, w)  # [B, 1, H, W]
        dy = dy.reshape(b, 1, h, w)  # [B, 1, H, W]

        offset_global = torch.cat([dx, dy], dim=1)  # [B, 2, H, W]
        offset_gen = self.offset_conv(torch.cat([feat_embed, ref_embed], dim=1))  # [B, C/r, H, W]
        offset = self.offset_fuse(torch.cat([offset_global, offset_gen], dim=1))  # [B, 2, H, W]

        return offset  # shape: [B, C, H, W]

    def forward(self, ref_feat, key_feat, cur_query):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        ref_feat_l1, ref_feat_l2, ref_feat_l3 = ref_feat[0], ref_feat[1], ref_feat[2]
        ref_feat_l1 = self.ref_conv_l1(ref_feat_l1)
        ref_feat_l2 = self.ref_conv_l2(ref_feat_l2)
        ref_feat_l3 = self.ref_conv_l3(ref_feat_l3)
        ref_feat_l = [ref_feat_l1, ref_feat_l2, ref_feat_l3]

        key_feat_l1, key_feat_l2, key_feat_l3 = key_feat[0], key_feat[1], key_feat[2]
        key_feat_l1 = self.key_conv_l1(key_feat_l1)
        key_feat_l2 = self.key_conv_l2(key_feat_l2)
        key_feat_l3 = self.key_conv_l3(key_feat_l3)
        key_feat_l = [key_feat_l1, key_feat_l2, key_feat_l3]

        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        att = None
        for i in range(3, 0, -1):  # 3，2，1
            level = f'l{i}'
            offset = torch.cat([ref_feat_l[i - 1], key_feat_l[i - 1]], dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1))

            feat = self.dcn_pack[level](ref_feat_l[i - 1].contiguous(), offset.contiguous())

            if i < 3:
                if i == 1:
                    feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
                    feat, att = self.query_mask[level](feat, cur_query)
                else:
                    feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features x2: when we upsample the offset, we should also enlarge the magnitude.
                upsampled_offset = self.upsample(offset) * 2  # 当offset视为feat时不*2，真为offset才*2
                upsampled_feat = self.upsample(feat)

        return feat, att


class Baseline_2D(nn.Module):
    def __init__(self, input_channels, num_frames=5, block=ResNet, act='relu', train_mode=False,
                 key_mode='last'):
        super().__init__()
        self.key_mode = key_mode
        param_channels = [16, 32, 64, 128]
        self.num_frames = num_frames
        self.feat_extract = feat_extract(input_channels, block=block)  # feat_extract_cycle

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.mask_align = DFDAasoffset(num_feat=param_channels[0], deformable_groups=4,
                                       train_mode=train_mode)
        self.multi_att = Mltiscale_Select_Att(channels=param_channels, reduction=8, patch=True,
                                              patchsize=[32, 16, 8], key_mode='last')  # 32, 16, 8
        self.object_query = ObjectQueryModule(channels=[16, 32, 64], cls_objects=1)  # param_channels[0]
        self.key_enhance_l1 = ObjectGuidedEnhancement(feat_dim=param_channels[0], query_dim=param_channels[0])

        self.output_0 = nn.Sequential(
            nn.Conv2d(param_channels[0], 1, kernel_size=3, padding=1)
        )

        self.output3d_0 = nn.Sequential(
            nn.Conv3d(param_channels[0], param_channels[0], kernel_size=(num_frames, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(param_channels[0]),
            get_activation(act, inplace=True),
            nn.Conv3d(param_channels[0], 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        )
        self.output_1 = nn.Sequential(
            nn.Conv3d(param_channels[1], param_channels[1], kernel_size=(5, 1, 1), padding=(0, 0, 0)),
            nn.GroupNorm(num_groups=8, num_channels=param_channels[1]),
            get_activation(act, inplace=True),
            nn.Conv3d(param_channels[1], 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        )
        self.output_2 = nn.Sequential(
            nn.Conv3d(param_channels[2], param_channels[2], kernel_size=(5, 1, 1), padding=(0, 0, 0)),
            nn.GroupNorm(num_groups=8, num_channels=param_channels[2]),
            get_activation(act, inplace=True),
            nn.Conv3d(param_channels[2], 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        )
        self.final = nn.Conv2d(3, 1, 3, 1, 1)
        self.temporal_attn = TemporalAttention(param_channels[0], reduction=8)

    def forward(self, x, feat_prop, cat_flag, ds_flag):
        out_l1, out_l2, out_l3, out_4 = self.feat_extract(x)  # 输出3D卷积解码的4个尺度特征
        out_l1, out_l2, out_l3 = self.multi_att([out_l1, out_l2, out_l3])
        track_loss, obj_query = self.object_query(out_l1)
        if self.key_mode == 'mid':
            key_feat = [out_l1[:, :, 2, :, :], out_l2[:, :, 2, :, :], out_l3[:, :, 2, :, :]]
        else:
            key_feat = [out_l1[:, :, -1, :, :], out_l2[:, :, -1, :, :], out_l3[:, :, -1, :, :]]
        align_feat_frame = []
        # att_list = []
        for t in range(out_l1.shape[2]):
            if self.key_mode == 'mid' and t == 2:
                key_feat_out, _ = self.key_enhance_l1(key_feat[0], obj_query[:, t])
                continue  # 跳过关键帧
            elif self.key_mode == 'last' and t == out_l1.shape[2] - 1:
                key_feat_out, _ = self.key_enhance_l1(key_feat[0], obj_query[:, t])
                continue
            ref_feat = [out_l1[:, :, t, :, :], out_l2[:, :, t, :, :], out_l3[:, :, t, :, :]]
            cur_query = obj_query[:, t]
            align_feat, _ = self.mask_align(ref_feat, key_feat, cur_query)
            align_feat_frame.append(align_feat)

        align_feat_frame = torch.stack(align_feat_frame, dim=2)
        align_feat_frame = torch.cat([align_feat_frame, key_feat_out.unsqueeze(2)], dim=2)
        align_feat_frame = self.temporal_attn(align_feat_frame)

        if ds_flag:
            mask0 = self.output_0(out_l1)
            mask1 = self.output_1(out_l2)
            mask2 = self.output_2(out_l3)
            output = self.final(
                torch.cat([mask0.squeeze(2), self.up(mask1.squeeze(2)), self.up_4(mask2.squeeze(2))], dim=1))
            return [mask0, mask1, mask2], output, feat_prop

        else:
            output = self.output3d_0(align_feat_frame)
            return [], output, feat_prop, track_loss


class TemporalAttention(nn.Module):
    def __init__(self, channels, reduction=8, use_scale=True):
        super(TemporalAttention, self).__init__()
        self.use_scale = use_scale
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 时间维度保持不变
        # 时间注意力通道
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.size()
        y = self.avg_pool(x)  # [B, C, T, 1, 1]

        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)

        if self.use_scale:
            y = y.view(b, c, t)
            y = F.softmax(y, dim=2)
            y = y.view(b, c, t, 1, 1)
        else:
            y = self.sigmoid(y)

        # 应用注意力权重
        return x * y


if __name__ == '__main__':
    model = Baseline_2D(input_channels=1, num_frames=5, train_mode=True, key_mode='last').cuda()
    dummy_input = torch.randn(1, 1, 5, 512, 512).cuda()
    tag = False
    flops, params = profile(model, (dummy_input, None, 0, False), verbose=False)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
