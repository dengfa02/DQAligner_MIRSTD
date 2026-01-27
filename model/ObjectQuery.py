import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# def get_freq_indices(method):
#     assert method in ['top1', 'top2', 'top4', 'top8', 'top9', 'top16', 'top32',
#                       'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
#                       'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
#     num_freq = int(method[3:])
#     if 'top' in method:
#         all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
#                              6, 1]
#         all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
#                              5, 3]
#         mapper_x = all_top_indices_x[:num_freq]  # DCT频率域中的坐标
#         mapper_y = all_top_indices_y[:num_freq]
#     elif 'low' in method:
#         all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
#                              3, 4]
#         all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
#                              4, 3]
#         mapper_x = all_low_indices_x[:num_freq]
#         mapper_y = all_low_indices_y[:num_freq]
#     elif 'bot' in method:
#         all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
#                              3, 6]
#         all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
#                              3, 3]
#         mapper_x = all_bot_indices_x[:num_freq]
#         mapper_y = all_bot_indices_y[:num_freq]
#     else:
#         raise NotImplementedError
#     return mapper_x, mapper_y
#
#
# class MultiSpectralDCTLayer(nn.Module):
#     """
#     Generate dct filters
#     """
#
#     def __init__(self, height, width, mapper_x, mapper_y, channel):
#         super(MultiSpectralDCTLayer, self).__init__()
#
#         assert len(mapper_x) == len(mapper_y)
#         assert channel % len(mapper_x) == 0
#
#         self.num_freq = len(mapper_x)
#
#         # fixed DCT init(org)
#         self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
#
#         # fixed random init
#         # self.register_buffer('weight', torch.rand(channel, height, width))
#
#         # learnable DCT init
#         # self.register_parameter('weight', nn.Parameter(self.get_dct_filter(height, width, mapper_x, mapper_y, channel)))
#
#         # learnable random init
#         # self.register_parameter('weight', torch.rand(channel, height, width))
#
#         # num_freq, h, w
#
#     def forward(self, x):
#         assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
#         # n, c, h, w = x.shape
#
#         x = x * self.weight
#
#         result = torch.sum(x, dim=[2, 3])
#         return result
#
#     def build_filter(self, pos, freq, POS):
#         result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
#         if freq == 0:
#             return result
#         else:
#             return result * math.sqrt(2)
#
#     def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
#         dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
#
#         c_part = channel // len(mapper_x)
#
#         for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
#             for t_x in range(tile_size_x):
#                 for t_y in range(tile_size_y):
#                     dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) \
#                                                                          * self.build_filter(t_y, v_y, tile_size_y)
#                     # 每个c_part内频率成分相同，例如32个channel，选取16个频率，则每2个channel内的频率成分相同,根据索引生成DCT基底
#
#         return dct_filter
#
#
# class FreqQueryGen(torch.nn.Module):
#     def __init__(self, channel, reduction=16, freq_sel_method='top16'):
#         super(FreqQueryGen, self).__init__()
#         c2wh = dict([(16, 56), (32, 56), (64, 56), (128, 28), (256, 14), (512, 7)])
#         dct_h, dct_w = c2wh[channel], c2wh[channel]
#         self.reduction = reduction
#         self.dct_h = dct_h
#         self.dct_w = dct_w
#
#         mapper_x, mapper_y = get_freq_indices(freq_sel_method)  # get the frequency coordinates
#         self.num_split = len(mapper_x)
#         mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
#         mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
#         # make the frequencies in different sizes are identical to a 7x7 frequency space
#         # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
#
#         self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=True),
#             # nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_pooled = x
#         if h != self.dct_h or w != self.dct_w:
#             x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))  # pool到56*56
#             # If you have concerns about one-line-change, don't worry.   :)
#             # In the ImageNet models, this line will never be triggered.
#             # This is for compatibility in instance segmentation and object detection.
#         q = self.dct_layer(x_pooled)
#         q = self.fc(q).view(b, 1, c)
#
#         # q = self.fc(q).view(n, c, 1, 1)
#
#         return q
#         # return x * y.expand_as(x)
#
#
# class FreqModulate(torch.nn.Module):
#     def __init__(self, channel, reduction=16, freq_sel_method='top16'):
#         super(FreqModulate, self).__init__()
#         c2wh = dict([(16, 56), (32, 56), (64, 56), (128, 28), (256, 14), (512, 7)])
#         dct_h, dct_w = c2wh[channel], c2wh[channel]
#         self.reduction = reduction
#         self.dct_h = dct_h
#         self.dct_w = dct_w
#
#         mapper_x, mapper_y = get_freq_indices(freq_sel_method)  # get the frequency coordinates
#         self.num_split = len(mapper_x)
#         mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
#         mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
#         # make the frequencies in different sizes are identical to a 7x7 frequency space
#         # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
#
#         self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, 9, bias=True),
#             # nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape  # 输入key特征
#         x_pooled = x
#         if h != self.dct_h or w != self.dct_w:
#             x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))  # pool到56*56
#             # If you have concerns about one-line-change, don't worry.   :)
#             # In the ImageNet models, this line will never be triggered.
#             # This is for compatibility in instance segmentation and object detection.
#         f = self.dct_layer(x_pooled)  # b, c
#         f = self.fc(f).view(b, 3, 3).unsqueeze(1)  # b, 1, 3, 3
#
#         f_k = f.repeat(b, c, 1, 1)[0].unsqueeze(1)  # c, 1, 3, 3  频率调制核
#
#         # q = self.fc(q).view(n, c, 1, 1)
#
#         return f_k
#         # return x * y.expand_as(x)


class ObjectQueryModule(nn.Module):
    def __init__(self, channels=[16, 32, 64], cls_objects=1):
        super().__init__()
        self.cls_objects = cls_objects
        self.channels = channels
        self.kv_encoder = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1, groups=self.channels[0]),
            nn.Conv2d(self.channels[0], 2 * self.channels[0], kernel_size=1))

        # 目标查询向量 (可学习的目标表示)
        self.object_queries = nn.Parameter(torch.randn(1, cls_objects, self.channels[0]))  # channels

        # 目标注意力
        self.object_attention = nn.MultiheadAttention(self.channels[0], 1, batch_first=True)
        self.object_update = nn.GRUCell(self.channels[0], self.channels[0])

    def motion_consistency_loss(self, object_tracks):
        B, T, num_objects, C = object_tracks.shape

        # 计算相邻时间步的目标表示相似度
        temp_consistency_loss = 0
        for t in range(T - 1):
            curr_objs = object_tracks[:, t]  # [B, num_objects, C]
            next_objs = object_tracks[:, t + 1]  # [B, num_objects, C]

            # 使用余弦相似度衡量目标表示的一致性
            similarity = F.cosine_similarity(curr_objs, next_objs, dim=2)  # [B, num_objects]
            temp_consistency_loss += (1 - similarity).mean()

        return temp_consistency_loss / (T - 1)

    def forward(self, x_enhanced):
        """
        或者三个尺度的特征 3 [B, C, T, H, W]
        x_enhanced: [B, C, T, H, W] - GMC_Att输出的浅层增强特征
        """
        B, C, T, H, W = x_enhanced.shape  # x_enhanced[0]
        object_states = self.object_queries.repeat(B, 1, 1)  # [B, 1, C]  self.object_queries.
        object_tracks = []

        for t in range(T):
            curr_feat = x_enhanced[:, :, t]  # x_enhanced[:, :, t]  # [B, C, H, W]
            feat_kv = self.kv_encoder(curr_feat)  # [B, 2C, H, W]
            feat_k, feat_v = torch.chunk(feat_kv, 2, dim=1)  # 各自形状: [B, C, H, W]

            feat_k = feat_k.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            feat_v = feat_v.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            attn_output, _ = self.object_attention(
                object_states,  # 查询：目标状态 [B, 1, C]
                feat_k,  # 键：特征图 [B, H*W, C]
                feat_v  # 值：特征图 [B, H*W, C]
            )  # [B, num_objects, C]

            object_states_flat = object_states.reshape(-1, C)  # [B*cls_objects, C]
            attn_output_flat = attn_output.reshape(-1, C)  # [B*cls_objects, C]

            updated_states = self.object_update(attn_output_flat, object_states_flat)  # [B*cls_objects, C]
            object_states = updated_states.reshape(B, self.cls_objects, C)  # [B, cls_objects, C]
            object_tracks.append(object_states)

            # 堆叠所有时间步的目标状态
        object_tracks = torch.stack(object_tracks, dim=1)  # [B, T, num_objects, C]
        object_sim_loss = self.motion_consistency_loss(object_tracks)

        return object_sim_loss, object_tracks


class ObjectGuidedEnhancement(nn.Module):
    def __init__(self, feat_dim, query_dim=16):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, feat_dim)
        self.key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.output_proj = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(feat_dim),
                                         nn.ReLU(inplace=True))

    def forward(self, features, object_queries):
        """
        features: [B, C, H, W] - 对齐后单尺度单帧特征
        object_queries: [B, 1, Q] - 目标查询状态
        """
        B, C, H, W = features.shape
        _, N, Q = object_queries.shape

        curr_feat = features  # [B, C, H, W]  [:,:,t]
        curr_queries = object_queries  # [B, N, Q]  [:, t]

        proj_queries = self.query_proj(curr_queries)  # [B, N, C]
        keys = self.key_conv(curr_feat)  # [B, C, H, W]
        keys_flat = keys.flatten(2)  # [B, C, H*W]

        attn = torch.bmm(
            proj_queries,  # [B, N, C]
            keys_flat  # [B, C, H*W]
        )  # [B, N, H*W]

        attn = attn / (C ** 0.5)  # 缩放
        attn = attn.view(B, N, H, W)
        mask = attn.sigmoid()  # [B, N, H, W]

        enhanced = curr_feat * (1.0 + mask)
        enhanced = self.output_proj(enhanced)

        # 堆叠所有增强的帧
        return enhanced, attn
