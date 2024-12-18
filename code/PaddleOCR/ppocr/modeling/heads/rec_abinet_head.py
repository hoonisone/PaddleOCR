# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from: 
https://github.com/FangShancheng/ABINet/tree/main/modules
"""

import math
from tarfile import LNKTYPE
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import LayerList
from ppocr.modeling.heads.rec_nrtr_head import TransformerBlock, PositionalEncoding


class BCNLanguage(nn.Layer):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=2048,
                 dropout=0.,
                 max_length=25,
                 detach=True,
                 num_classes=37):
        super().__init__()

        self.d_model = d_model
        self.detach = detach
        self.max_length = max_length + 1  # additional stop token
        self.proj = nn.Linear(num_classes, d_model, bias_attr=False)
        self.token_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(
            dropout=0, dim=d_model, max_len=self.max_length)

        self.decoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=False,
                with_cross_attn=True) for i in range(num_layers)
        ])

        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (B, N, C) where N is length, B is batch size and C is classes number
            lengths: (B,) where B is batch size, each sample's length is in [0, N]
        """
        if self.detach: tokens = tokens.detach()
        embed = self.proj(tokens)  # (B, N, C)
        embed = self.token_encoder(embed)  # (B, N, C)
        padding_mask = _get_mask(lengths, self.max_length)
        zeros = paddle.zeros_like(embed)  # (B, N, C)
        qeury = self.pos_encoder(zeros)
        for decoder_layer in self.decoder:
            qeury = decoder_layer(qeury, embed, cross_mask=padding_mask)
        output = qeury  # (B, N, C)

        # print("BCN", output.shape)
        logits = self.cls(output)  # (B, N, C)
        # print(self.cls)
        # print(logits.shape)
        # print(output.shape)
        # exit()
        return output, logits


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2D(in_c, out_c, k, s, p), nn.BatchNorm2D(out_c), nn.ReLU())


def decoder_layer(in_c,
                  out_c,
                  k=3,
                  s=1,
                  p=1,
                  mode='nearest',
                  scale_factor=None,
                  size=None):
    align_corners = False if mode == 'nearest' else True
    return nn.Sequential(
        nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners),
        nn.Conv2D(in_c, out_c, k, s, p),
        nn.BatchNorm2D(out_c),
        nn.ReLU())


class PositionAttention(nn.Layer):
    def __init__(self,
                 max_length,
                 in_channels=512,
                 num_channels=64,
                 h=8,
                 w=32,
                 mode='nearest',
                 **kwargs):
        super().__init__()
        self.max_length = max_length
        self.k_encoder = nn.Sequential(
            encoder_layer(
                in_channels, num_channels, s=(1, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)))
        self.k_decoder = nn.Sequential(
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, in_channels, size=(h, w), mode=mode))

        self.pos_encoder = PositionalEncoding(
            dropout=0, dim=in_channels, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        k, v = x, x

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
            # k_encoder를 한 번에 수행해도 되지만, 중간 결과를 저장해두고 나중에 사용하는 방식을 사용
        for i in range(0, len(self.k_decoder) - 1): # 아!! 마지막 꺼 빼는구나
            k = self.k_decoder[i](k)
            # print(k.shape, features[len(self.k_decoder) - 2 - i].shape)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        zeros = paddle.zeros(
            (B, self.max_length, C), dtype=x.dtype)  # (T, N, C)
        q = self.pos_encoder(zeros)  # (B, N, C)
        q = self.project(q)  # (B, N, C)

        # calculate attention
        attn_scores = q @k.flatten(2)  # (B, N, (H*W))
        attn_scores = attn_scores / (C**0.5)
        attn_scores = F.softmax(attn_scores, axis=-1)

        v = v.flatten(2).transpose([0, 2, 1])  # (B, (H*W), C)
        attn_vecs = attn_scores @v  # (B, N, C)

        return attn_vecs, attn_scores.reshape([0, self.max_length, H, W])
    
class PositionAttention_GraphemeLabel_B(nn.Layer):
    def __init__(self,
                 max_length,
                 in_channels=512,
                 num_channels=64,
                 h=8,
                 w=32,
                 mode='nearest',
                 handling_grapheme = None,
                 **kwargs):
        super().__init__()
        self.max_length = max_length
        self.handling_grapheme = handling_grapheme
        self.k_encoder = nn.Sequential(
            encoder_layer(
                in_channels, num_channels, s=(1, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)),
            encoder_layer(
                num_channels, num_channels, s=(2, 2)))
        self.k_decoder = nn.Sequential(
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(
                num_channels, in_channels, size=(h, w), mode=mode))

        self.pos_encoder = PositionalEncoding(
            dropout=0, dim=in_channels, max_len=max_length)
        
        
        self.project = dict()
        for g in self.handling_grapheme:
            self.project[g] = nn.Linear(in_channels, in_channels)
            setattr(self, f"project_{g}", self.project[g])

    def forward(self, x):
        B, C, H, W = x.shape
        k, v = x, x

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
            # k_encoder를 한 번에 수행해도 되지만, 중간 결과를 저장해두고 나중에 사용하는 방식을 사용
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            # print(k.shape, features[len(self.k_decoder) - 2 - i].shape)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        zeros = paddle.zeros(
            (B, self.max_length, C), dtype=x.dtype)  # (T, N, C)
        q = self.pos_encoder(zeros)  # (B, N, C)
        
        
        result = dict()
        for g in self.handling_grapheme:
            q2 = self.project[g](q)  # (B, N, C)

            # calculate attention
            attn_scores = q2 @k.flatten(2)  # (B, N, (H*W))
            attn_scores = attn_scores / (C**0.5)
            attn_scores = F.softmax(attn_scores, axis=-1)

            v2 = v.flatten(2).transpose([0, 2, 1])  # (B, (H*W), C)
            
            attn_vecs = attn_scores @v2  # (B, N, C)
            result[g] = attn_vecs, attn_scores.reshape([0, self.max_length, H, W])
            
            

        return result

class ABINetHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_model=512,  
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 use_lang=False,
                 iter_size=1, **kwargs):
        super().__init__()
        self.max_length = max_length + 1
        self.pos_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=8 * 32)

        self.encoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False) for i in range(num_layers)
        ])
        self.decoder = PositionAttention(
            max_length=max_length + 1,  # additional stop token
            mode='nearest', )
        self.out_channels = out_channels
        self.cls = nn.Linear(d_model, self.out_channels)
        self.use_lang = use_lang
        if use_lang:
            self.iter_size = iter_size
            self.language = BCNLanguage(
                d_model=d_model,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_length=max_length,
                num_classes=self.out_channels)
            # alignment
            self.w_att_align = nn.Linear(2 * d_model, d_model)
            self.cls_align = nn.Linear(d_model, self.out_channels)

    def forward(self, x, targets=None):
        x = x.transpose([0, 2, 3, 1])
        _, H, W, C = x.shape
        feature = x.flatten(1, 2)
        feature = self.pos_encoder(feature) # 미리 계산된 PE 상수 값에 대해 feature 범위 만큼만 추출하여 더함, 그 뒤에 dropout
        # Positional Encoding에서는 [sequence length, batch size, feature dim]으로 입력을 받는다.
        # 근데 지금은 [batch size, sequence length, feature dim]으로 입력을 받는다.
        

        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)# multi-head attention
            # input: [B, S, D] batch size, sequence length, feature dim
            # output: [B, S, D] batch size, sequence length, feature dim
        feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        v_feature, attn_scores = self.decoder(
            feature)  # (B, N, C), (B, C, H, W)
        vis_logits = self.cls(v_feature)  # (B, N, C)
        # logits은 token이 각 class에 속할 확률을 나타낸다.
        # (B, N, C) -> 샘플별, 토큰 별 class에 속할 확률 (확률 보단 적합도에 가깝다. 0~1은 아니고 0~무한대의 값)
        logits = vis_logits
        vis_lengths = _get_length(vis_logits) # 
        if self.use_lang:
            align_logits = vis_logits
            align_lengths = vis_lengths
            all_l_res, all_a_res = [], []
            for i in range(self.iter_size):
                tokens = F.softmax(align_logits, axis=-1)
                lengths = align_lengths
                lengths = paddle.clip(
                    lengths, 2, self.max_length)  # TODO:move to langauge model
                l_feature, l_logits = self.language(tokens, lengths)

                # alignment
                all_l_res.append(l_logits)
                fuse = paddle.concat((l_feature, v_feature), -1)
                f_att = F.sigmoid(self.w_att_align(fuse))
                output = f_att * v_feature + (1 - f_att) * l_feature
                align_logits = self.cls_align(output)  # (B, N, C)

                align_lengths = _get_length(align_logits)
                all_a_res.append(align_logits)
            if self.training:
                return {
                    'align': all_a_res,
                    'lang': all_l_res,
                    'vision': vis_logits
                }
            else:
                logits = align_logits
        if self.training:
            return logits
        else:
            return F.softmax(logits, -1)


class ABINetHead_GraphemeLabel_old(ABINetHead):
    pass
    # def __init__(self,
    #              in_channels,
    #              out_channels,
    #              d_model=512,  
    #              nhead=8,
    #              num_layers=3,
    #              dim_feedforward=2048,
    #              dropout=0.1,
    #              max_length=25,
    #              use_lang=False,
    #              class_num_dict=None, # None이 아니면 grapheme 방식으로 동작
    #              iter_size=1, **kwargs):

    #     super().__init__(in_channels, out_channels, d_model, nhead, num_layers, dim_feedforward, dropout, max_length, use_lang, iter_size, **kwargs)
    #     if class_num_dict is None:
    #         raise Exception("class_num_dict must not be None")
        
    #     self.class_num_dict = class_num_dict
    #     self.main_label = "character" if "character" in class_num_dict.keys() else "initial"   
        
    
    # def split_grapheme_logits(self, x):
    #     self.class_num_dict
        
    #     index_ranges = {}
    #     start_index = 0
    #     for key, length in self.class_num_dict.items():
    #         index_ranges[key] = (start_index, start_index + length)
    #         start_index += length
    
    #     dict_out = {key : x[:, :, start:end] for key, (start, end) in index_ranges.items()}
    #     return dict_out
    #     # _in = self.class_num_dict["initial"]
    #     # _mn = self.class_num_dict["medial"]
    #     # _fn = self.class_num_dict["final"]
        
    #     # f = x[:, :, :_in]
    #     # m = x[:, :, _in:_in+_mn]
    #     # l = x[:, :, _in+_mn:]
    #     # return f, m, l
    
    # def concat_grapheme_logits(self, x, axis=-1):
    #     temp_list = []
    #     for key in ["character", "initial", "medial", "final"]:
    #         if key in x:
    #             temp_list.append(x[key])
                
    #     return paddle.concat(temp_list, axis=axis)
    
    # def softmax_grapheme_logits(self, x, axis=-1):
    #     grapheme_logits = self.split_grapheme_logits(x)
    #     grapheme_logits = {name:F.softmax(logit, -1) for name, logit in grapheme_logits.items()}
    #     return self.concat_grapheme_logits(grapheme_logits, axis=axis)

    # def forward(self, x, targets=None):

    #     x = x.transpose([0, 2, 3, 1])
    #     _, H, W, C = x.shape
    #     feature = x.flatten(1, 2)

    #     feature = self.pos_encoder(feature) # 미리 계산된 PE 상수 값에 대해 feature 범위 만큼만 추출하여 더함, 그 뒤에 dropout
    #     # Positional Encoding에서는 [sequence length, batch size, feature dim]으로 입력을 받는다.
    #     # 근데 지금은 [batch size, sequence length, feature dim]으로 입력을 받는다.    
        
    #     for encoder_layer in self.encoder:
    #         feature = encoder_layer(feature)# multi-head attention
    #         # input: [B, S, D] batch size, sequence length, feature dim
    #         # output: [B, S, D] batch size, sequence length, feature dim
        
    #     feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        
        
    #     v_feature, attn_scores = self.decoder(feature)  # (B, N, C), (B, C, H, W)
    #     vis_logits = self.cls(v_feature)  # (B, N, C)
    #     # logits은 token이 각 class에 속할 확률을 나타낸다.
    #     # (B, N, C) -> 샘플별, 토큰 별 class에 속할 확률 (확률 보단 적합도에 가깝다. 0~1은 아니고 0~무한대의 값)
    #     logits = vis_logits
    #     grapheme_logit_dict = self.split_grapheme_logits(logits)
             
    #     vis_lengths = _get_length(grapheme_logit_dict[self.main_label])
        
    #     if self.use_lang:
    #         align_logits = vis_logits
    #         align_lengths = vis_lengths
    #         all_l_res, all_a_res = [], []
    #         for i in range(self.iter_size):
    #             # tokens = F.softmax(align_logits, axis=-1)
    #             tokens= self.softmax_grapheme_logits(align_logits, axis=-1)
    #             lengths = align_lengths
    #             lengths = paddle.clip(
    #                 lengths, 2, self.max_length)  # TODO:move to langauge model
    #             l_feature, l_logits = self.language(tokens, lengths)

    #             # alignment
    #             all_l_res.append(l_logits)
    #             fuse = paddle.concat((l_feature, v_feature), -1)
    #             f_att = F.sigmoid(self.w_att_align(fuse))
    #             output = f_att * v_feature + (1 - f_att) * l_feature
    #             align_logits = self.cls_align(output)  # (B, N, C)
    #             f_align_logits = self.split_grapheme_logits(align_logits)
                
    #             align_lengths = _get_length(f_align_logits[self.main_label])
    #             all_a_res.append(align_logits)
    #         if self.training:
    #             return {
    #                 'align': all_a_res,
    #                 'lang': all_l_res,
    #                 'vision': vis_logits
    #             }
    #         else:
    #             logits = align_logits
    #     if self.training:
    #         return {
    #             'vision': vis_logits
    #         }
    #         # return logits
    #     else:
    #         # return F.softmax(logits, -1)
    #         return self.softmax_grapheme_logits(logits, axis=-1)

class ABINetHead_GraphemeLabel(ABINetHead):
    # grapheme 들을 한 vecter로 추론하고 각 grapheme 크기로 나눈다.
    
    
    def __init__(self,
                 in_channels,
                 out_channels: dict, # 일반적으로 int이나 grapheme 클래스에서는 dict
                 d_model=512,  
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25, 
                 use_lang=False,
                 iter_size=1, **kwargs):

        total_output = sum(out_channels.values())
        super().__init__(in_channels, total_output, d_model, nhead, num_layers, dim_feedforward, dropout, max_length, use_lang, iter_size, **kwargs)
        
        self.out_channels = out_channels
        self.main_label = "character" if "character" in self.out_channels.keys() else "utf8string" if "utf8string" in self.out_channels.keys() else "initial"   
        
    
    def split_grapheme_logits(self, x):
        
        index_ranges = {}
        start_index = 0
        for key, length in self.out_channels.items():
            index_ranges[key] = (start_index, start_index + length)
            start_index += length
    
        dict_out = {key : x[:, :, start:end] for key, (start, end) in index_ranges.items()}
        return dict_out
    
    def concat_grapheme_logits(self, x, axis=-1):
        temp_list = []
        for key in ["character", "initial", "medial", "final", "utf8string"]:
            if key in x:
                temp_list.append(x[key])
                
        return paddle.concat(temp_list, axis=axis)
    
    def softmax_for_each_grapheme_logits(self, x, axis=-1):
        grapheme_logits = self.split_grapheme_logits(x)
        grapheme_logits = {name:F.softmax(logit, -1) for name, logit in grapheme_logits.items()}
        return self.concat_grapheme_logits(grapheme_logits, axis=axis)

    def forward(self, x, targets=None):

        x = x.transpose([0, 2, 3, 1])
        _, H, W, C = x.shape
        feature = x.flatten(1, 2)

        feature = self.pos_encoder(feature) # 미리 계산된 PE 상수 값에 대해 feature 범위 만큼만 추출하여 더함, 그 뒤에 dropout
        # Positional Encoding에서는 [sequence length, batch size, feature dim]으로 입력을 받는다.
        # 근데 지금은 [batch size, sequence length, feature dim]으로 입력을 받는다.    
        
        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)# multi-head attention
            # input: [B, S, D] batch size, sequence length, feature dim
            # output: [B, S, D] batch size, sequence length, feature dim
        
        feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        
        
        v_feature, attn_scores = self.decoder(feature)  # (B, N, C), (B, C, H, W)
        vis_logits = self.cls(v_feature)  # (B, N, C)
        # logits은 token이 각 class에 속할 확률을 나타낸다.
        # (B, N, C) -> 샘플별, 토큰 별 class에 속할 확률 (확률 보단 적합도에 가깝다. 0~1은 아니고 0~무한대의 값)
        # logits = vis_logits
        vis_grapheme_logit = self.split_grapheme_logits(vis_logits)
             
        vis_lengths = _get_length(vis_grapheme_logit[self.main_label])
        
        report = {'vision': vis_grapheme_logit}
        
        if self.use_lang:
            align_logits = vis_logits # grapheme이 하나의 벡터로 모인 형태
            align_lengths = vis_lengths
            # all_l_res, all_a_res = [], []
            for i in range(self.iter_size):
                # tokens = F.softmax(align_logits, axis=-1)
                tokens= self.softmax_for_each_grapheme_logits(align_logits, axis=-1)
                lengths = align_lengths
                lengths = paddle.clip(
                    lengths, 2, self.max_length)  # TODO:move to langauge model
                l_feature, l_logits = self.language(tokens, lengths)

                # alignment

                lang_grapheme_logits = self.split_grapheme_logits(l_logits)
                report[f"lang{i+1}"] = lang_grapheme_logits
                
                
                # fuse the features of vision and lang
                fuse = paddle.concat((l_feature, v_feature), -1) # fuse는 logits을 구하기 전 상태인 feature 단위에서 수행
                f_att = F.sigmoid(self.w_att_align(fuse))
                output = f_att * v_feature + (1 - f_att) * l_feature
                
                
                align_logits = self.cls_align(output)  # (B, N, C)
                f_align_logits = self.split_grapheme_logits(align_logits)
                
                align_lengths = _get_length(f_align_logits[self.main_label])
                report[f"align{i+1}"] = f_align_logits
            # report["lang"] = all_l_res
            # report["align"] = all_a_res

        return report

class ABINetHead_GraphemeLabel_B(ABINetHead_GraphemeLabel):
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_model=512,  
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 use_lang=False,
                #  class_num_dict=None, # None이 아니면 grapheme 방식으로 동작
                 handling_grapheme = None,
                 iter_size=1, **kwargs):
        
        super().__init__(in_channels, out_channels, d_model, nhead, num_layers, dim_feedforward, dropout, max_length, use_lang, iter_size, **kwargs)
        # if class_num_dict is None:
        #     raise Exception("class_num_dict must not be None")
        
        self.decoder = PositionAttention_GraphemeLabel_B(
            max_length=max_length + 1,  # additional stop token
            mode='nearest', handling_grapheme=handling_grapheme)

        self.handling_grapheme = handling_grapheme
        self.cls = nn.Linear(d_model*len(handling_grapheme), d_model)
        
        for g in handling_grapheme:
            setattr(self, f"cls_{g}", nn.Linear(d_model, self.out_channels[g]))
        
    def forward(self, x, targets=None):

        x = x.transpose([0, 2, 3, 1])
        _, H, W, C = x.shape
        feature = x.flatten(1, 2)

        feature = self.pos_encoder(feature) # 미리 계산된 PE 상수 값에 대해 feature 범위 만큼만 추출하여 더함, 그 뒤에 dropout
        # Positional Encoding에서는 [sequence length, batch size, feature dim]으로 입력을 받는다.
        # 근데 지금은 [batch size, sequence length, feature dim]으로 입력을 받는다.    

        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)# multi-head attention
            # input: [B, S, D] batch size, sequence length, feature dim
            # output: [B, S, D] batch size, sequence length, feature dim
        
        feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        
        v_decode = self.decoder(feature)  # (B, N, C), (B, C, H, W)

        v_feature = paddle.concat([v_decode[g][0] for g in self.handling_grapheme], axis=-1)
        
        # print(v_feature.shape)
        v_feature = self.cls(v_feature)
        # print(v_feature.shape)
        
        vis_logits = [getattr(self, f"cls_{g}")(v_decode[g][0]) for g in self.handling_grapheme]
        vis_logits = paddle.concat(vis_logits, axis=-1)
                
        
        vis_grapheme_logit = self.split_grapheme_logits(vis_logits)
             
        vis_lengths = _get_length(vis_grapheme_logit[self.main_label])
        
        report = {'vision': vis_grapheme_logit}
        
        if self.use_lang:
            align_logits = vis_logits # grapheme이 하나의 벡터로 모인 형태
            align_lengths = vis_lengths
            # all_l_res, all_a_res = [], []
            for i in range(self.iter_size):
                # tokens = F.softmax(align_logits, axis=-1)
                tokens= self.softmax_for_each_grapheme_logits(align_logits, axis=-1)
                lengths = align_lengths
                lengths = paddle.clip(
                    lengths, 2, self.max_length)  # TODO:move to langauge model
                l_feature, l_logits = self.language(tokens, lengths)

                # alignment

                lang_grapheme_logits = self.split_grapheme_logits(l_logits)
                report[f"lang{i+1}"] = lang_grapheme_logits
                
                
                # fuse the features of vision and lang
                fuse = paddle.concat((l_feature, v_feature), -1) # fuse는 logits을 구하기 전 상태인 feature 단위에서 수행
                f_att = F.sigmoid(self.w_att_align(fuse))
                output = f_att * v_feature + (1 - f_att) * l_feature
                
                
                align_logits = self.cls_align(output)  # (B, N, C)
                f_align_logits = self.split_grapheme_logits(align_logits)
                
                align_lengths = _get_length(f_align_logits[self.main_label])
                report[f"align{i+1}"] = f_align_logits
            # report["lang"] = all_l_res
            # report["align"] = all_a_res

        return report
import paddle.nn.initializer as init




class ABINetHead_GraphemeLabel_A2(nn.Layer): # character와 grapheme에 대한 head 파트를 A 방식으로 완전히 독립적으로 수행
    def __init__(self,
                in_channels,
                out_channels,
                d_model=512,  
                nhead=8,
                num_layers=3,
                dim_feedforward=2048,
                dropout=0.1,
                max_length=25,
                use_lang=False,
            #  class_num_dict=None, # None이 아니면 grapheme 방식으로 동작
                handling_grapheme = None,
                iter_size=1, **kwargs):
        super().__init__()
        
        self.character_inner_header = None
        self.grapheme_inner_header = None
        
        if "character" in out_channels:
            inner_out_channels = {"character": out_channels["character"]}
        
            self.character_inner_header = ABINetHead_GraphemeLabel(in_channels, inner_out_channels, d_model, nhead, num_layers, dim_feedforward, dropout, max_length, use_lang, iter_size, **kwargs)
    
    
        graphemes = list(set(out_channels)-set(["character"]))
        if len(graphemes) > 0: 
            inner_out_channels = {g: out_channels[g] for g in graphemes}
            self.grapheme_inner_header = ABINetHead_GraphemeLabel(in_channels, inner_out_channels, d_model, nhead, num_layers, dim_feedforward, dropout, max_length, use_lang, iter_size, **kwargs)
    
        self.max_length = max_length
        
        
        model_names = ["vision"]+[f"lang{i}" for i in range(1, iter_size+1)]+[f"align{i}" for i in range(1, iter_size+1)]
        initializer = paddle.nn.initializer.XavierUniform()
        
        
    def forward(self, x, targets=None):            
            
        result = None
        if self.character_inner_header:
            result = self.character_inner_header(x, targets)
            
        if self.grapheme_inner_header:
            if result is None:
                result = self.grapheme_inner_header(x, targets)
            else:
                sub_result = self.grapheme_inner_header(x, targets)
                for model_name, pred_dict in sub_result.items():
                    result[model_name].update(pred_dict)
        
        return result 
            
        
class ABINetHead_GraphemeLabel_A3(nn.Layer): # 현재  Character + Utf  조합만 사용가능
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_model=512,  
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 use_lang=False,
                 iter_size=1, **kwargs):
        super().__init__()
        self.max_length = max_length + 1
        self.pos_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=8 * 32)


        self.encoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False) for i in range(num_layers)
        ])
        self.decoder = PositionAttention(
            max_length=max_length + 1,  # additional stop token
            mode='nearest', )
        
        
        self.out_channels = out_channels
        
        self.cls = nn.Linear(d_model, self.out_channels["character"])
        self.utf_cls = nn.Linear(d_model, self.out_channels["utf8string"])
        
        self.utf_lang_cls = nn.Linear(d_model, self.out_channels["utf8string"]) # BCNLanguage가 character에 대해서는 자체적으로 해줌, utf만 BNC이 출력한 feature에 한번 젹용하면 됌
        # self.cls = nn.Linear(d_model, self.out_channels)
        
        
        self.use_lang = use_lang
        if use_lang:
            self.iter_size = iter_size
            self.language = BCNLanguage(
                d_model=d_model,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_length=max_length,
                num_classes=self.out_channels["character"])
            # alignment
            self.w_att_align = nn.Linear(2 * d_model, d_model)
            self.cls_align = nn.Linear(d_model, self.out_channels["character"])
            self.utf_cls_align = nn.Linear(d_model, self.out_channels["utf8string"])

            
            # self.cls_align = nn.Linear(d_model, self.out_channels)

    def forward(self, x, targets=None):
        x = x.transpose([0, 2, 3, 1])
        _, H, W, C = x.shape
        feature = x.flatten(1, 2)
        feature = self.pos_encoder(feature) # 미리 계산된 PE 상수 값에 대해 feature 범위 만큼만 추출하여 더함, 그 뒤에 dropout
        # Positional Encoding에서는 [sequence length, batch size, feature dim]으로 입력을 받는다.
        # 근데 지금은 [batch size, sequence length, feature dim]으로 입력을 받는다.
        

        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)# multi-head attention
            # input: [B, S, D] batch size, sequence length, feature dim
            # output: [B, S, D] batch size, sequence length, feature dim
        feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        v_feature, attn_scores = self.decoder(
            feature)  # (B, N, C), (B, C, H, W)
        
        
        # vis_logits = self.cls(v_feature)  # (B, N, C)
        vis_logits = self.cls(v_feature)  # (B, N, C)
        utf_vis_logits = self.utf_cls(v_feature)  # (B, N, C)####### 추가 코드
        
        report = {'vision': {"character": vis_logits, "utf8string": utf_vis_logits}}
        
        
        # logits은 token이 각 class에 속할 확률을 나타낸다.
        # (B, N, C) -> 샘플별, 토큰 별 class에 속할 확률 (확률 보단 적합도에 가깝다. 0~1은 아니고 0~무한대의 값)
        logits = vis_logits
        vis_lengths = _get_length(vis_logits) # 
        if self.use_lang:
            align_logits = vis_logits
            align_lengths = vis_lengths
            all_l_res, all_a_res = [], []
            for i in range(self.iter_size):
                tokens = F.softmax(align_logits, axis=-1)
                lengths = align_lengths
                lengths = paddle.clip(
                    lengths, 2, self.max_length)  # TODO:move to langauge model
                l_feature, l_logits = self.language(tokens, lengths)
               
                # print(feature.shape)
                # print(l_logits.shape)
                # print(self.utf_lang_cls)
                # print(self.d_model)
                # exit()
                
                utf_l_logits = self.utf_lang_cls(l_feature)
                
                report[f"lang{i+1}"] = {"character": l_logits, "utf8string": utf_l_logits}
                
                 
                # alignment
                # all_l_res.append(l_logits)
                fuse = paddle.concat((l_feature, v_feature), -1)
                f_att = F.sigmoid(self.w_att_align(fuse))
                output = f_att * v_feature + (1 - f_att) * l_feature
                align_logits = self.cls_align(output)  # (B, N, C)
                utf_align_logits = self.utf_cls_align(output)  # (B, N, C) ####### 추가 코드
                

                report[f"align{i+1}"] = {"character": align_logits, "utf8string": utf_align_logits}
                align_lengths = _get_length(align_logits)
                
                
                
                all_a_res.append(align_logits)
        return report
        

class ABINetHead_GraphemeLabel_A4(nn.Layer): # 현재  Character + Utf  조합만 사용가능 (A3에서 Character와 Utf의 기능만 교체함 utf가 lang의 주체가 됌)
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_model=512,  
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_length=25,
                 use_lang=False,
                 iter_size=1, **kwargs):
        super().__init__()
        self.max_length = max_length + 1
        self.pos_encoder = PositionalEncoding(
            dropout=0.1, dim=d_model, max_len=8 * 32)


        self.encoder = nn.LayerList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=dropout,
                residual_dropout_rate=dropout,
                with_self_attn=True,
                with_cross_attn=False) for i in range(num_layers)
        ])
        self.decoder = PositionAttention(
            max_length=max_length + 1,  # additional stop token
            mode='nearest', )
        
        
        self.out_channels = out_channels
        
        self.cls = nn.Linear(d_model, self.out_channels["utf8string"])
        self.c_cls = nn.Linear(d_model, self.out_channels["character"])
        
        # self.utf_lang_cls = nn.Linear(d_model, self.out_channels["utf8string"]) # BCNLanguage가 character에 대해서는 자체적으로 해줌, utf만 BNC이 출력한 feature에 한번 젹용하면 됌
        self.c_lang_cls = nn.Linear(d_model, self.out_channels["character"]) # BCNLanguage가 character에 대해서는 자체적으로 해줌, utf만 BNC이 출력한 feature에 한번 젹용하면 됌
        # self.cls = nn.Linear(d_model, self.out_channels)
        
        
        self.use_lang = use_lang
        if use_lang:
            self.iter_size = iter_size
            self.language = BCNLanguage(
                d_model=d_model,
                nhead=nhead,
                num_layers=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_length=max_length,
                num_classes=self.out_channels["utf8string"])
            # alignment
            self.w_att_align = nn.Linear(2 * d_model, d_model)
            self.cls_align = nn.Linear(d_model, self.out_channels["utf8string"])
            self.c_cls_align = nn.Linear(d_model, self.out_channels["character"])

            
            # self.cls_align = nn.Linear(d_model, self.out_channels)

    def forward(self, x, targets=None):
        x = x.transpose([0, 2, 3, 1])
        _, H, W, C = x.shape
        feature = x.flatten(1, 2)
        feature = self.pos_encoder(feature) # 미리 계산된 PE 상수 값에 대해 feature 범위 만큼만 추출하여 더함, 그 뒤에 dropout
        # Positional Encoding에서는 [sequence length, batch size, feature dim]으로 입력을 받는다.
        # 근데 지금은 [batch size, sequence length, feature dim]으로 입력을 받는다.
        

        for encoder_layer in self.encoder:
            feature = encoder_layer(feature)# multi-head attention
            # input: [B, S, D] batch size, sequence length, feature dim
            # output: [B, S, D] batch size, sequence length, feature dim
        feature = feature.reshape([0, H, W, C]).transpose([0, 3, 1, 2])
        v_feature, attn_scores = self.decoder(
            feature)  # (B, N, C), (B, C, H, W)
        
        
        # vis_logits = self.cls(v_feature)  # (B, N, C)
        vis_logits = self.cls(v_feature)  # (B, N, C)
        c_vis_logits = self.c_cls(v_feature)  # (B, N, C)####### 추가 코드
        
        report = {'vision': {"character": c_vis_logits, "utf8string": vis_logits}}
        
        
        # logits은 token이 각 class에 속할 확률을 나타낸다.
        # (B, N, C) -> 샘플별, 토큰 별 class에 속할 확률 (확률 보단 적합도에 가깝다. 0~1은 아니고 0~무한대의 값)
        logits = vis_logits
        vis_lengths = _get_length(vis_logits) # 
        if self.use_lang:
            align_logits = vis_logits
            align_lengths = vis_lengths
            all_l_res, all_a_res = [], []
            for i in range(self.iter_size):
                tokens = F.softmax(align_logits, axis=-1)
                lengths = align_lengths
                lengths = paddle.clip(
                    lengths, 2, self.max_length)  # TODO:move to langauge model
                l_feature, l_logits = self.language(tokens, lengths)
               
                # print(feature.shape)
                # print(l_logits.shape)
                # print(self.utf_lang_cls)
                # print(self.d_model)
                # exit()
                
                c_l_logits = self.c_lang_cls(l_feature)
                
                report[f"lang{i+1}"] = {"character": c_l_logits, "utf8string": l_logits}
                
                 
                # alignment
                # all_l_res.append(l_logits)
                fuse = paddle.concat((l_feature, v_feature), -1)
                f_att = F.sigmoid(self.w_att_align(fuse))
                output = f_att * v_feature + (1 - f_att) * l_feature
                align_logits = self.cls_align(output)  # (B, N, C)
                c_align_logits = self.c_cls_align(output)  # (B, N, C) ####### 추가 코드
                

                report[f"align{i+1}"] = {"character": c_align_logits, "utf8string": align_logits}
                align_lengths = _get_length(align_logits)
                
                
                
                all_a_res.append(align_logits)
        return report
    
        
def _get_length(logit):
    """ 
    Description:
        Greed decoder to obtain length from logit
    Args:
        logit: (B, N, C) where N is length, B is batch size and C is classes number
    Returns:
        out: (B,) where B is batch size
    """
    out = (logit.argmax(-1) == 0) # 0은 stop token class
    abn = out.any(-1) # 샘플에 stop token이 있는가?
    out_int = out.cast('int32')
    out = (out_int.cumsum(-1) == 1) & out
    out = out.cast('int32')
    out = out.argmax(-1)
    out = out + 1 # stop token을 포함한 길이
    len_seq = paddle.zeros_like(out) + logit.shape[1]
    out = paddle.where(abn, out, len_seq) # stop token이 없는 샘플은 max length로 설정
    return out


def _get_mask(length, max_length):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    length = length.unsqueeze(-1)
    B = paddle.shape(length)[0]
    grid = paddle.arange(0, max_length).unsqueeze(0).tile([B, 1])
    zero_mask = paddle.zeros([B, max_length], dtype='float32')
    inf_mask = paddle.full([B, max_length], '-inf', dtype='float32')
    diag_mask = paddle.diag(
        paddle.full(
            [max_length], '-inf', dtype=paddle.float32),
        offset=0,
        name=None)
    mask = paddle.where(grid >= length, inf_mask, zero_mask)
    mask = mask.unsqueeze(1) + diag_mask
    return mask.unsqueeze(1)
