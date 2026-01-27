import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dcn.modules.deform_conv import DeformConv
import math
from einops import rearrange
import numbers


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


class MotionAttentionEncoding(nn.Module):
    def __init__(self, channels, num_frames=5):
        super().__init__()
        # 运动注意力模块
        self.conv_diff = nn.Conv3d(channels, channels, kernel_size=1)
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        diff = torch.zeros_like(x, device=x.device)  # [b, c, t-1, h, w]
        for i in range(t):
            if i == t - 1:
                diff[:, :, i, :, :] = x[:, :, -1, :, :]
            else:
                diff[:, :, i, :, :] = x[:, :, -1, :, :] - x[:, :, i, :, :]  # [b, c, t-1, h, w]

        motion_feat = self.conv_diff(diff)
        spatial_weights = self.spatial_attn(motion_feat)
        enhanced = x * (1 + spatial_weights)

        return enhanced


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta  # 0.3-0.7

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


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=2):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleFeatureSection(nn.Module):
    def __init__(self, reduction=8, param_channels=[16, 32, 64, 128]):
        super().__init__()
        self.param_c = param_channels
        # self.patch_size = patch_size

        # 降维投影 (用于对齐通道)
        self.dim_conv = nn.ModuleDict()
        for i in range(3):
            level = f'l{i}'
            self.dim_conv[level] = nn.Conv2d(param_channels[0], param_channels[1], kernel_size=1)
            if i == 1:
                self.dim_conv[level] = nn.Conv2d(param_channels[1], param_channels[1], kernel_size=1)
            if i == 2:
                self.dim_conv[level] = nn.Conv2d(param_channels[2], param_channels[1], kernel_size=1)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3 * self.param_c[1], self.param_c[1], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.param_c[1], 3 * self.param_c[1], kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.recover = nn.ModuleDict()
        for i in range(3):
            level = f'l{i}'
            self.recover[level] = nn.Conv3d(param_channels[1], param_channels[0], kernel_size=1)
            if i == 1:
                self.recover[level] = nn.Conv3d(param_channels[1], param_channels[1], kernel_size=1)
            if i == 2:
                self.recover[level] = nn.Conv3d(param_channels[1], param_channels[2], kernel_size=1)
        # self.recover = nn.Conv2d(dim // reduction * 3, dim, kernel_size=1)

    def forward(self, emb_list):
        B, _, T, eH, eW = emb_list[0].shape
        multiscale_feat = emb_list
        reshape_features = []
        for l in range(len(multiscale_feat)):
            level = f'l{l}'
            feat = multiscale_feat[l]
            feat_flat = feat.permute(0, 2, 1, 3, 4).view(B * T, self.param_c[l], eH, eW)
            feat_c = self.dim_conv[level](feat_flat)  # [B*T, C2, eH, eW]
            # feat_reduced = self.dim_reduce(feat_flat)  # [B*T, C/r, 16, 16]
            # feat_reduced = feat_reduced.view(B, C // 8, T, 16, 16)
            reshape_features.append(feat_c)

        # 多尺度通道选择
        # 合并所有尺度特征用于通道注意力计算
        channel_cat = torch.cat(reshape_features, dim=1)  # [B*T, 3*C2, eH, eW]
        c_select = self.channel_attention(channel_cat)
        c_select = c_select.view(B, T, 3, self.param_c[1], 1, 1).permute(0, 2, 3, 1, 4, 5)
        # [B, 3, C2, T, 1, 1] 对同一尺度所有帧通道共享权重

        # 多尺度patch选择
        spatial_maps = []
        for feat in reshape_features:
            # 计算每个尺度的空间特征
            spatial_feat = feat.mean(dim=1, keepdim=True)  # [B*T, 1, 16, 16]
            spatial_maps.append(spatial_feat)

        spatial_cat = torch.cat(spatial_maps, dim=1)  # [B*T, 3, 16, 16]  对所有帧patch共享权重
        spatial_select = self.spatial_attention(spatial_cat).view(B, T, 3, 1, eH, eW).permute(0, 2, 3, 1, 4, 5)
        # [B, 3, 1, T, 16, 16]

        # 多尺度帧选择
        temporal_cat = spatial_cat.mean(dim=(2, 3), keepdim=True)
        temporal_select = self.temporal_attention(temporal_cat).view(B, T, 3, 1, 1, 1).permute(0, 2, 3, 1, 4,
                                                                                               5)  # [B, 3, 1, T, 1, 1]
        # [B, 3, 1, T, 1, 1]

        # 应用注意力权重
        inter_feat = channel_cat.view(B, T, 3, self.param_c[1], eH, eW).permute(0, 2, 3, 1, 4, 5)
        # [B, 3, C2, T, 16, 16]
        enheanced_feat = inter_feat * c_select * spatial_select * temporal_select

        enhanced_emb_out = []
        for l in range(len(multiscale_feat)):
            level = f'l{l}'
            enhanced_emb = enheanced_feat[:, l]  # [B, C2, T, 16, 16]
            enhanced_emb = self.recover[level](enhanced_emb) + multiscale_feat[l]
            enhanced_emb_out.append(enhanced_emb)

        return enhanced_emb_out[0], enhanced_emb_out[1], enhanced_emb_out[2]
        # [B,C1,T,16,16] [B,C2,T,16,16] [B,C3,T,16,16]


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, size=None):
        if x is None:
            return None

        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Patch_Embeddings(nn.Module):
    def __init__(self, patchsize, in_channels, overlap):
        super().__init__()

        self.patch_embeddings = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patchsize,
                      stride=patchsize - overlap, padding=(overlap // 2, overlap // 2), groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # b, c, 16, 16
        return x


class Motion_Mask(nn.Module):
    def __init__(self, mode='sprase'):
        super(Motion_Mask, self).__init__()
        self.mode = mode  # 'sprase' or 'dense'
        if self.mode == 'sprase':
            self.maskgenconv = nn.Conv2d(2, 1, 7, padding=3, bias=True)
        else:
            self.maskgenconv = nn.Conv2d(2, 1, 3, padding=1, bias=True)

    def forward(self, att_feat, topk_size, size=[128, 128]):
        B, _, _ = att_feat.shape  # B,N,1
        if self.mode == 'sprase':
            row_sum = att_feat.sum(dim=2, keepdim=True)  # [B, HW, 1]
            avg_out = row_sum / topk_size
        else:
            avg_out = torch.mean(att_feat, dim=2, keepdim=True)  # [B, HW, 1]
        max_out, _ = torch.max(att_feat, dim=2, keepdim=True)  # [B, HW, 1]
        avg_out = avg_out.permute(0, 2, 1).view(B, 1, size[0], size[1])  # [B, 1, H, W]
        max_out = max_out.permute(0, 2, 1).view(B, 1, size[0], size[1])  # [B, 1, H, W]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.maskgenconv(x)
        mask = x.sigmoid()
        return mask


class GMC_Att(nn.Module):
    def __init__(self, channels, reduction=8, num_frames=5, patch=False, patchsize=16, estimate=True, scale=1.0,
                 org_size=[200, 150], train_mode=False, key_mode='last'):
        super(GMC_Att, self).__init__()
        self.is_patch = patch
        self.key_mode = key_mode
        self.estimate = estimate
        self.scale = scale
        self.embedding = nn.Sequential(nn.Conv2d(channels, channels // reduction, kernel_size=1),
                                       nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3,
                                                 padding='same'))
        self.q_Conv = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1,
                      groups=channels // reduction),
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=1))
        self.k_Conv = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1,
                      groups=channels // reduction),
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=1))
        self.v_Conv = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1,
                      groups=channels // reduction),
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=1))
        self.key_k_Conv = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1,
                      groups=channels // reduction),
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=1))
        self.key_v_Conv = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1,
                      groups=channels // reduction),
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=1))
        self.att_c = channels // reduction

        self.attn_norm1 = LayerNorm3d(channels // reduction, LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channels // reduction, LayerNorm_type='WithBias')
        self.ffn1 = FeedForward(channels // reduction, ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channels // reduction, ffn_expansion_factor=2.66, bias=False)
        self.proj1 = nn.Sequential(LayerNorm3d(channels // reduction, LayerNorm_type='WithBias'),
                                   nn.Conv2d(channels // reduction, channels, kernel_size=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.proj2 = nn.Sequential(LayerNorm3d(channels // reduction, LayerNorm_type='WithBias'),
                                   nn.Conv2d(channels // reduction, channels, kernel_size=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.outconv_ref = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1))
        self.outconv_key = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1))
        self.mask_gen = Motion_Mask(mode='dense')  # 'sprase' or 'dense'
        # 目标大小自适应估计器
        self.size_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )

        if self.scale > 1.0:
            self.est_conv = nn.Sequential(
                nn.Conv3d(channels * int(self.scale), channels * int(self.scale), kernel_size=(1, 3, 3),
                          padding=(0, 1, 1),
                          groups=channels * int(self.scale)),
                nn.BatchNorm3d(channels * int(self.scale)),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels * int(self.scale), channels, kernel_size=1))
            self.fusion_gate = nn.Sequential(
                nn.Conv3d(channels * 2, channels * 2, kernel_size=(1, 3, 3),
                          padding=(0, 1, 1),
                          groups=channels * 2),
                nn.BatchNorm3d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels * 2, channels, kernel_size=1),
                nn.Sigmoid())
            self.final = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                          padding=(0, 1, 1)),
                nn.BatchNorm3d(channels), nn.ReLU())

    def forward(self, x, x_enhance_pre, ref_idx=-1):
        if self.is_patch:
            B, C, T, emb_H, emb_W = x.shape
        else:
            B, C, T, H, W = x.shape
        if self.key_mode == 'mid':
            ref_idx = 2
        if self.estimate:
            attn_list = []
            ref_v_list = []
            x_enhanced = torch.zeros_like(x)
            key_feat = x[:, :, ref_idx]  # [B,C,H,W]未增强的关键帧特征
            if self.is_patch:
                key_embed = self.embedding(key_feat[:, :, :emb_H, :emb_W])  # [B, C/r, s, s]
            else:
                key_embed = self.embedding(key_feat[:, :, :H, :W])
            key_k = self.key_k_Conv(key_embed)  # [B, C/r, H, W]
            key_v = self.key_v_Conv(key_embed)  # [B, C/r, H, W]
            if self.is_patch:
                key_k = key_k.view(B, -1, emb_H * emb_W).permute(0, 2, 1)  # [B, eH*eW, C/r]
                key_v = key_v.view(B, -1, emb_H * emb_W).permute(0, 2, 1)  # [B, eH*eW, C/r]
            else:
                key_k = key_k.view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C/r]
                key_v = key_v.view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C/r]
            for t in range(T):
                if t == T - 1:
                    """key特征增强分块降低复杂度"""
                    chunk_size = 1  # 每次处理的帧数
                    key_enhance_chunks = []

                    for chunk_start in range(0, len(ref_v_list), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(ref_v_list))
                        chunk_v_list = ref_v_list[chunk_start:chunk_end]
                        chunk_attn_list = attn_list[chunk_start:chunk_end]
                        # 处理当前块
                        chunk_ref_v = torch.stack(chunk_v_list, dim=2)  # B C chunk_size H W
                        if self.is_patch:
                            chunk_ref_v = chunk_ref_v.view(B, -1, chunk_size * emb_H * emb_W).permute(0, 2, 1)
                            chunk_attn = torch.stack(chunk_attn_list, dim=1).view(B, chunk_size * emb_H * emb_W,
                                                                                  emb_H * emb_W)
                        else:
                            chunk_ref_v = chunk_ref_v.view(B, -1, chunk_size * H * W).permute(0, 2, 1)
                            chunk_attn = torch.stack(chunk_attn_list, dim=1).view(B, chunk_size * H * W, H * W)
                            # [B, chunk_size*H*W, H*W]
                        chunk_attn = chunk_attn.transpose(-2, -1)  # [B, H*W, chunk_size*H*W]
                        chunk_enhance = torch.bmm(chunk_attn, chunk_ref_v)  # [B, H*W, C/r]
                        key_enhance_chunks.append(chunk_enhance)

                    key_enhance = torch.stack(key_enhance_chunks).mean(dim=0)  # [B, H*W, C/r]
                    if self.is_patch:
                        key_enhance = key_enhance.permute(0, 2, 1).view(B, self.att_c, emb_H, emb_W)  # [B, C/r, H, W]
                    else:
                        key_enhance = key_enhance.permute(0, 2, 1).view(B, self.att_c, H, W)  # [B, C/r, H, W]

                    """不分块"""
                    # ref_v = torch.stack(ref_v_list, dim=2)  # B C T-1 H W
                    # ref_v = ref_v.view(B, -1, (T - 1) * H * W).permute(0, 2, 1)  # [B, (T-1)*H*W, C/r]
                    # attn = torch.stack(attn_list, dim=1).view(B, (T - 1) * H * W,
                    #                                           H * W)  # [B, (T-1)H*W, H*W] 复用相似性增强key
                    # attn = attn.transpose(-2, -1)  # [B, H*W, (T-1)H*W]
                    # key_enhance = torch.bmm(attn, ref_v)  # [B, H*W, C/r]
                    # key_enhance = key_enhance.permute(0, 2, 1).view(B, self.att_c, H, W)  # [B, C/r, H, W]

                    key_att = key_enhance + key_embed  # [B, C/r, H, W]
                    key_att = self.attn_norm2(key_att)  # layernorm
                    key_ffn = self.ffn2(key_att)  # [B, C/r, H, W]  # 轻量级特征增强
                    key_ffn = key_ffn + key_att  # [B, C/r, H, W]
                    key_ffn = self.proj2(key_ffn)  # [B, C, H, W]

                    key_ffn = key_ffn + key_feat  # 残差连接原始输入特征
                    key_ffn = self.outconv_key(key_ffn)  # [B, C, H, W]
                    x_enhanced[:, :, t] = key_ffn  # 关键帧增强引入

                    continue

                cur_feat = x[:, :, t]
                if self.is_patch:
                    ref_embed = self.embedding(cur_feat[:, :, :emb_H, :emb_W])
                else:
                    ref_embed = self.embedding(cur_feat[:, :, :H, :W])  # [B, C/r, s, s]
                ref_q = self.q_Conv(ref_embed)  # [B, C/r, s, s]
                ref_v = self.v_Conv(ref_embed)  # [B, C/r, s, s]
                ref_v_list.append(ref_v)  # T[B,C,H,W]
                """ref特征增强"""
                if self.is_patch:
                    ref_q = ref_q.view(B, -1, emb_H * emb_W).permute(0, 2, 1)  # [B, eH*eW, C/r]
                else:
                    ref_q = ref_q.view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C/r]
                attn1 = torch.bmm(ref_q, key_k.transpose(-2, -1)) / math.sqrt(self.att_c)  # B, HW, HW
                attn1 = F.softmax(attn1, dim=2)  # B, eHeW, eHeW
                attn_list.append(attn1)

                ref_enhance = torch.bmm(attn1, key_v)  # [B, H*W, C/r]  ref_v
                if self.is_patch:
                    ref_enhance = ref_enhance.permute(0, 2, 1).view(B, self.att_c, emb_H, emb_W)  # [B, C/r, H, W]
                else:
                    ref_enhance = ref_enhance.permute(0, 2, 1).view(B, self.att_c, H, W)  # [B, C/r, H, W]
                attn_out = ref_enhance + ref_embed  # [B, C/r, H, W]
                attn_out = self.attn_norm1(attn_out)  # layernorm

                ffn_out = self.ffn1(attn_out)  # [B, C/r, H, W]
                ffn_out = ffn_out + attn_out
                ffn_out = self.proj1(ffn_out)  # [B, C, H, W]

                ffn_out = ffn_out + cur_feat
                ffn_out = self.outconv_ref(ffn_out)  # [B, C, H, W]

                x_enhanced[:, :, t] = ffn_out

        else:
            x_enhanced = nn.Upsample(scale_factor=(1, self.scale, self.scale), mode='trilinear', align_corners=True)(
                x_enhance_pre)  # t
            if self.scale > 1.0:
                x_enhanced = self.est_conv(x_enhanced)  # [B, C, T, H, W]
                gate = self.fusion_gate(torch.cat([x_enhanced, x], dim=1))
                x_enhanced = self.final(x_enhanced * gate + x * (1 - gate))  # [B, C, T, H, W]
        return x_enhanced


class Mltiscale_Select_Att(nn.Module):
    def __init__(self, channels=[16, 32, 64, 128], reduction=8, patch=False, patchsize=[32, 16, 8], key_mode='last'):
        super(Mltiscale_Select_Att, self).__init__()
        self.is_patch = patch
        self.key_mode = key_mode
        self.channels = channels
        if self.is_patch:
            self.patch_embeddings = nn.ModuleList()
            self.reconstructs = nn.ModuleList()
            for l in range(3):
                overlap = patchsize[l] // 2
                self.patch_embeddings.append(Patch_Embeddings(patchsize[l], in_channels=channels[l], overlap=overlap))
                self.reconstructs.append(Reconstruct(channels[l], channels[l], kernel_size=3))

        self.Multiscale_feat_select = MultiScaleFeatureSection(reduction=reduction)
        self.Att_l3 = GMC_Att(channels[2], reduction=8, num_frames=5, patch=patch, patchsize=patchsize[2],
                              estimate=True, scale=1.0,
                              org_size=[200, 150], train_mode=False, key_mode='last')  # CDDC
        self.Att_l2 = GMC_Att(channels[1], reduction=8, num_frames=5, patch=False, patchsize=patchsize[1],
                              estimate=True, scale=2.0,
                              org_size=[200, 150], train_mode=False, key_mode='last')
        self.Att_l1 = GMC_Att(channels[0], reduction=8, num_frames=5, patch=False, patchsize=patchsize[0],
                              estimate=True, scale=4.0,
                              org_size=[200, 150], train_mode=False, key_mode='last')

    def forward(self, x):  # , key_enhanced_l, mask_list
        """输入三个尺度"""
        H1, W1 = x[0].shape[-2], x[0].shape[-1]  # 原始输入层级尺寸
        H2, W2 = x[1].shape[-2], x[1].shape[-1]
        H3, W3 = x[2].shape[-2], x[2].shape[-1]
        emb_l = []
        for l in range(3):
            if self.is_patch:
                B, C, T, H, W = x[l].shape
                feat_l = x[l].permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # [B*T,C,H,W]
                feat_l = self.patch_embeddings[l](feat_l)
                _, _, emb_H, emb_W = feat_l.shape
                feat_l = feat_l.view(B, T, C, emb_H, emb_W).permute(0, 2, 1, 3, 4)  # [B,C,T,emb_H,emb_W]
                emb_l.append(feat_l)

        emb_l_selected = self.Multiscale_feat_select(emb_l)
        emb_l1, emb_l2, emb_l3 = emb_l_selected[0], emb_l_selected[1], emb_l_selected[2]

        emb_l1_att = self.Att_l1(emb_l1, None)  # [B,C1,T,emb_H,emb_W]
        emb_l2_att = self.Att_l2(emb_l2, None)  # [B,C2,T,emb_H,emb_W]
        emb_l3_att = self.Att_l3(emb_l3, None)  # [B,C3,T,emb_H,emb_W]

        emb_l1_att = emb_l1_att.permute(0, 2, 1, 3, 4).contiguous().view(B * T, self.channels[0], emb_H,
                                                                         emb_W)  # [B,T,C1,emb_H,emb_W]
        emb_l2_att = emb_l2_att.permute(0, 2, 1, 3, 4).contiguous().view(B * T, self.channels[1], emb_H,
                                                                         emb_W)  # [B,T,C2,emb_H,emb_W]
        emb_l3_att = emb_l3_att.permute(0, 2, 1, 3, 4).contiguous().view(B * T, self.channels[2], emb_H,
                                                                         emb_W)  # [B,T,C3,emb_H,emb_W]

        emb_l1_re = self.reconstructs[0](emb_l1_att, size=(H1, W1)).view(B, T, self.channels[0], H1, W1).permute(0, 2,
                                                                                                                 1, 3,
                                                                                                                 4)
        emb_l2_re = self.reconstructs[1](emb_l2_att, size=(H2, W2)).view(B, T, self.channels[1], H2, W2).permute(0, 2,
                                                                                                                 1, 3,
                                                                                                                 4)
        emb_l3_re = self.reconstructs[2](emb_l3_att, size=(H3, W3)).view(B, T, self.channels[2], H3, W3).permute(0, 2,
                                                                                                                 1, 3,
                                                                                                                 4)

        out_l1 = emb_l1_re + x[0]  # 残差连接
        out_l2 = emb_l2_re + x[1]
        out_l3 = emb_l3_re + x[2]

        return out_l1, out_l2, out_l3  # B, C, T, H, W


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.layer_norm1 = LayerNorm3d(hidden_features * 2, LayerNorm_type='WithBias')
        self.act = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.layer_norm1(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class CDDC(nn.Module):
    def __init__(self):
        super(CDDC, self).__init__()
