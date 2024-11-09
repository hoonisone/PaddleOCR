# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]

class FirstCTCHead(nn.Layer):
    char_set = """!"#$%&'*+-/0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~©°²½ÁÄÅÇÉÍÎÓÖ×ÜßàáâãäåæçèéêëìíîïðñòóôõöøúûüýāăąćČčđēėęěğīİıŁłńňōřŞşŠšţūźżŽžȘșΑΔαλφГОавлорстя​’“”→∇∼「」アカグニランㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"""

    def __init__(self,
                 in_channels = 64,
                 out_channels = 215,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        
        super(CTCHead, self).__init__()
        if mid_channels is None:
            weight_attr, bias_attr = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        else:
            weight_attr1, bias_attr1 = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1)

            weight_attr2, bias_attr2 = get_para_bias_attr(
                l2_decay=fc_decay, k=mid_channels)
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2)
            
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        
        if self.mid_channels is None: # True
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:    # False
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
            result = predicts

        return result
class CTCHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()

        if mid_channels is None:
            weight_attr, bias_attr = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        else:
            weight_attr1, bias_attr1 = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1)

            weight_attr2, bias_attr2 = get_para_bias_attr(
                l2_decay=fc_decay, k=mid_channels)
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        
        if self.mid_channels is None: # True
            # print("@1")
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:    # False
            # print("@2")
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            # print("@3")
            predicts = F.softmax(predicts, axis=2)
            result = predicts

        return result
import paddle.nn.initializer as init


class CTCHead_Grapheme(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 handling_grapheme=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead_Grapheme, self).__init__()
        
        assert handling_grapheme is not None
        self.handling_grapheme = handling_grapheme
        
        assert mid_channels is None, "Not implemented" # if need, just implement
        assert return_feats is False, "Not implemented"
        
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=fc_decay, k=in_channels) # grapheme 마다 따로 만들 필요 없겠지?  GPT가 그렇데
        
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.fc_dict = {}
        
        for g in self.handling_grapheme:
            fc = nn.Linear(
                in_channels,
                out_channels[g],
                weight_attr=weight_attr,
                bias_attr=bias_attr)
            self.__setattr__(f"{g}_fc", fc)
            self.fc_dict[g] = fc
        
        # self.fc1 = nn.Linear((5)*40*2, 2)
        
        # self.logit_fc_k_3 = nn.Linear(3*40*2, 2)    
        # self.logit_fc_k_3_c = nn.Linear(3*40, 1)
        # self.logit_fc_k_3_g = nn.Linear(3*40, 1)        
        
        # self.logit_fc_k_5 = nn.Linear(5*40*2, 2)
        # self.logit_fc_k_5_c = nn.Linear(5*40, 1)
        # self.logit_fc_k_5_g = nn.Linear(5*40, 1)
                
        # self.logit_fc_k_10 = nn.Linear(10*40*2, 2)
        # self.logit_fc_k_10_c = nn.Linear(10*40, 1)
        # self.logit_fc_k_10_g = nn.Linear(10*40, 1)
        
        # self.logit_fc_k_15 = nn.Linear(15*40*2, 2)
        # self.logit_fc_k_15_c = nn.Linear(15*40, 1)
        # self.logit_fc_k_15_g = nn.Linear(15*40, 1)
        
        
        # self.logit_fc_k_20 = nn.Linear(20*40*2, 2)
        # self.logit_fc_k_20_c = nn.Linear(20*40, 1)
        # self.logit_fc_k_20_g = nn.Linear(20*40, 1)
        
        # self.logit_fc_full = nn.Linear((1850+54)*40, 2)
        
        # self.logit_fc_character = nn.Linear(1850*40, 1)
        # self.logit_fc_grapheme = nn.Linear(54*40, 1)
        
        
        
        # layers_to_initialize = [
        #     self.logit_fc_k_3, self.logit_fc_k_3_c, self.logit_fc_k_3_g,
        #     self.logit_fc_k_5, self.logit_fc_k_5_c, self.logit_fc_k_5_g,
        #     self.logit_fc_k_10, self.logit_fc_k_10_c, self.logit_fc_k_10_g,
        #     self.logit_fc_k_15, self.logit_fc_k_15_c, self.logit_fc_k_15_g,
        #     self.logit_fc_k_20, self.logit_fc_k_20_c, self.logit_fc_k_20_g,
        #     self.logit_fc_full, self.logit_fc_character, self.logit_fc_grapheme
        # ]
    

        # # 각 레이어의 가중치와 편향 초기화
        # for layer in layers_to_initialize:
        #     init.XavierUniform()(layer.weight)  # 가중치를 Xavier Uniform 분포로 초기화
        #     if layer.bias is not None:
        #         init.Constant(0.0)(layer.bias)  # 편향을 0으로 초기화
        
        
        # self.logit_fc_compare_full = nn.Linear((1850+54)*40, 1)
        
        # layer = self.logit_fc_compare_full
        # init.XavierUniform()(layer.weight)  # 가중치를 Xavier Uniform 분포로 초기화
        # if layer.bias is not None:
        #     init.Constant(0.0)(layer.bias)  # 편향을 0으로 초기화
        
        
        
        # self.flag = False
        
    def forward(self, x, targets=None):
        # if self.flag == False:
        #     self.flag = True
        #     # 모델의 가중치를 초기화할 레이어들만 선택하여 초기화
        #     layers_to_initialize = [
        #         self.logit_fc_k_3, self.logit_fc_k_5, self.logit_fc_k_10, 
        #         self.logit_fc_k_10_c, self.logit_fc_k_10_g, 
        #         self.logit_fc_k_15, self.logit_fc_k_20, 
        #         self.logit_fc_full, self.logit_fc_character, self.logit_fc_grapheme
        #     ]

        #     # 가중치와 편향을 초기화하는 루프
        #     for layer in layers_to_initialize:
        #         init.XavierUniform()(layer.weight)  # 가중치를 Xavier Uniform 분포로 초기화
        #         init.Constant(0.0)(layer.bias)
            
            
        predicts = {
            "vision": {g: self.fc_dict[g](x) for g in self.handling_grapheme}
        }

        
        if self.return_feats:    # False
            # print("@2")
            # result = (x, predicts)
            raise NotImplementedError
        else:
            result = predicts
        
        # for model, pred in predicts.items():
        #     # character_logit = pred["character"].flatten(start_axis=1, stop_axis=-1).detach()
        #     # grapheme_logit = pred["utf8string"].flatten(start_axis=1, stop_axis=-1).detach()
        #     # full_logit = paddle.concat([character_logit, grapheme_logit], axis=-1)
        #     # head_confidence = F.sigmoid(self.logit_fc_compare_full(full_logit))
        #     # pred["head_compare_full"] = head_confidence
              
        #     # exit()
            
        #     character_top_logit = paddle.sort(paddle.topk(pred["character"], k=5)[0], axis=-1).reshape([-1, 5*40])
        #     grapheme_top_logit = paddle.sort(paddle.topk(pred["utf8string"], k=5)[0], axis=-1).reshape([-1, 5*40])
        #     top_logit = paddle.concat([character_top_logit, grapheme_top_logit], axis=-1).detach()
        #     head_confidence = F.sigmoid(self.fc1(top_logit))
        #     pred["head_confidence_k_5"] = head_confidence
            
            
            
        #     # character_top_logit = paddle.sort(paddle.topk(pred["character"], k=3)[0], axis=-1).reshape([-1, 3*40])
        #     # grapheme_top_logit = paddle.sort(paddle.topk(pred["utf8string"], k=3)[0], axis=-1).reshape([-1, 3*40])
        #     # top_logit = paddle.concat([character_top_logit, grapheme_top_logit], axis=-1).detach()
        #     # head_confidence = F.sigmoid(self.logit_fc_k_3(top_logit))
        #     # pred["head_confidence_k_3"] = head_confidence
            
        #     # character_conf = self.logit_fc_k_3_c(character_top_logit)
        #     # grapheme_conf = self.logit_fc_k_3_g(grapheme_top_logit)
        #     # conf = paddle.concat([character_conf, grapheme_conf], axis=-1)
        #     # pred["head_confidence2_k_3"] = F.sigmoid(conf)
        
            
            
        #     # character_top_logit = paddle.sort(paddle.topk(pred["character"], k=5)[0], axis=-1).reshape([-1, 5*40])
        #     # grapheme_top_logit = paddle.sort(paddle.topk(pred["utf8string"], k=5)[0], axis=-1).reshape([-1, 5*40])
        #     # top_logit = paddle.concat([character_top_logit, grapheme_top_logit], axis=-1).detach()
        #     # head_confidence = F.sigmoid(self.logit_fc_k_5(top_logit))
        #     # pred["head_confidence_k_5"] = head_confidence
            
        #     # character_conf = self.logit_fc_k_5_c(character_top_logit)
        #     # grapheme_conf = self.logit_fc_k_5_g(grapheme_top_logit)
        #     # conf = paddle.concat([character_conf, grapheme_conf], axis=-1)
        #     # pred["head_confidence2_k_5"] = F.sigmoid(conf)
            
            
            
        #     # character_top_logit = paddle.sort(paddle.topk(pred["character"], k=10)[0], axis=-1).reshape([-1, 10*40])
        #     # grapheme_top_logit = paddle.sort(paddle.topk(pred["utf8string"], k=10)[0], axis=-1).reshape([-1, 10*40])
        #     # top_logit = paddle.concat([character_top_logit, grapheme_top_logit], axis=-1).detach()
        #     # head_confidence = F.sigmoid(self.logit_fc_k_10(top_logit))
        #     # pred["head_confidence_k_10"] = head_confidence
            
        #     # character_conf = self.logit_fc_k_10_c(character_top_logit)
        #     # grapheme_conf = self.logit_fc_k_10_g(grapheme_top_logit)
        #     # conf = paddle.concat([character_conf, grapheme_conf], axis=-1)
        #     # pred["head_confidence2_k_10"] = F.sigmoid(conf)
            
            
            
        #     # character_top_logit = paddle.sort(paddle.topk(pred["character"], k=15)[0], axis=-1).reshape([-1, 15*40])
        #     # grapheme_top_logit = paddle.sort(paddle.topk(pred["utf8string"], k=15)[0], axis=-1).reshape([-1, 15*40])
        #     # top_logit = paddle.concat([character_top_logit, grapheme_top_logit], axis=-1).detach()
        #     # head_confidence = F.sigmoid(self.logit_fc_k_15(top_logit))
        #     # pred["head_confidence_k_15"] = head_confidence
            
        #     # character_conf = self.logit_fc_k_15_c(character_top_logit)
        #     # grapheme_conf = self.logit_fc_k_15_g(grapheme_top_logit)
        #     # conf = paddle.concat([character_conf, grapheme_conf], axis=-1)
        #     # pred["head_confidence2_k_15"] = F.sigmoid(conf)
            
            
            
        #     # character_top_logit = paddle.sort(paddle.topk(pred["character"], k=20)[0], axis=-1).reshape([-1, 20*40])
        #     # grapheme_top_logit = paddle.sort(paddle.topk(pred["utf8string"], k=20)[0], axis=-1).reshape([-1, 20*40])
        #     # top_logit = paddle.concat([character_top_logit, grapheme_top_logit], axis=-1).detach()
        #     # head_confidence = F.sigmoid(self.logit_fc_k_20(top_logit))
        #     # pred["head_confidence_k_20"] = head_confidence

        #     # character_conf = self.logit_fc_k_20_c(character_top_logit)
        #     # grapheme_conf = self.logit_fc_k_20_g(grapheme_top_logit)
        #     # conf = paddle.concat([character_conf, grapheme_conf], axis=-1)
        #     # pred["head_confidence2_k_20"] = F.sigmoid(conf)
            
            
            
        #     # character_logit = pred["character"].flatten(start_axis=1, stop_axis=-1).detach()
        #     # grapheme_logit = pred["utf8string"].flatten(start_axis=1, stop_axis=-1).detach()
            
        #     # full_logit = paddle.concat([character_logit, grapheme_logit], axis=-1)
        #     # head_confidence = F.sigmoid(self.logit_fc_full(full_logit))
        #     # pred["head_confidence_full"] = head_confidence

        #     # character_conf = self.logit_fc_character(character_logit)
        #     # grapheme_conf = self.logit_fc_grapheme(grapheme_logit)
        #     # conf = paddle.concat([character_conf, grapheme_conf], axis=-1)
        #     # pred["head_confidence2_full"] = F.sigmoid(conf)
            

        return result
        
 