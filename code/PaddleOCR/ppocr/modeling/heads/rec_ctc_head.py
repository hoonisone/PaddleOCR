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
from paddle.nn.initializer import KaimingUniform


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    # initializer = nn.initializer.Uniform(-stdv, stdv)
    initializer = nn.initializer.KaimingUniform()
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]

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


        
    def forward(self, x, targets=None):
        predicts = {
            "vision": {g: self.fc_dict[g](x) for g in self.handling_grapheme}
        }

        if self.return_feats:
            raise NotImplementedError
        else:
            result = predicts

        return result
    
    
class CTCHead_Grapheme_B1(nn.Layer):
    """
        자소 3개를 한 글자로 결합하여 글자 추론
        변환시 layer 1층
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 handling_grapheme=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead_Grapheme_B1, self).__init__()
        
        assert handling_grapheme is not None
        

        assert handling_grapheme == ["character", "utf8string"]
        self.handling_grapheme = handling_grapheme
        
        
        assert mid_channels is None, "Not implemented" # if need, just implement
        assert return_feats is False, "Not implemented"
        
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=fc_decay, k=in_channels) # grapheme 마다 따로 만들 필요 없겠지?  GPT가 그렇데
        
        
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.fc_dict = {}
        
        self.utf8string_fc = nn.Linear(
                in_channels,
                out_channels["utf8string"],
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        
        self.character_fc = nn.Linear(
                in_channels*3,
                out_channels["character"],
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        
        self.character_activation = nn.GELU()
        
        
        nn.initializer.KaimingUniform()(self.character_fc.weight)

        
    def forward(self, x, targets=None):
        
        predicts = {"vision": {"utf8string": self.utf8string_fc(x)}}
        
        B, S, C = x.shape
        x = x.reshape([B, S//3, 3*C])
        predicts["vision"]["character"] = self.character_activation(self.character_fc(x))

        if self.return_feats:
            raise NotImplementedError
        else:
            result = predicts
        return result

class CTCHead_Grapheme_B2(nn.Layer):
    """
        자소 3개를 한 글자로 결합하여 글자 추론
        변환시 layer 2층
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 handling_grapheme=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead_Grapheme_B2, self).__init__()
        
        assert handling_grapheme is not None
        

        assert handling_grapheme == ["character", "utf8string"]
        self.handling_grapheme = handling_grapheme
        
        
        assert mid_channels is None, "Not implemented" # if need, just implement
        assert return_feats is False, "Not implemented"
        
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=fc_decay, k=in_channels) # grapheme 마다 따로 만들 필요 없겠지?  GPT가 그렇데
        
        
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.fc_dict = {}
        
        self.utf8string_fc = nn.Linear(
                in_channels,
                out_channels["utf8string"],
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        
        mid_channels = 512
        self.character_fc1 = nn.Linear(
                in_channels*3,
                mid_channels,
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        
        self.character_fc2 = nn.Linear(
                mid_channels,
                out_channels["character"],
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        
        self.character_activation = nn.GELU()


        
    def forward(self, x, targets=None):
        
        predicts = {"vision": {"utf8string": self.utf8string_fc(x)}}
        
        B, S, C = x.shape
        x = x.reshape([B, S//3, 3*C])
        x = self.character_fc1(x)
        x = self.character_activation(x)
        x = self.character_fc2(x)
        x = self.character_activation(x)
        predicts["vision"]["character"] = x

        if self.return_feats:
            raise NotImplementedError
        else:
            result = predicts
        return result
    
      
class CTCHead_Grapheme_Neck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 handling_grapheme=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead_Grapheme_Neck, self).__init__()
        
        assert handling_grapheme is not None
        self.handling_grapheme = handling_grapheme
        
        assert mid_channels is None, "Not implemented" # if need, just implement
        assert return_feats is False, "Not implemented"
        self.return_feats = return_feats
        
        self.character_head = CTCHead(in_channels, out_channels["character"], fc_decay, mid_channels, return_feats)
        self.utf8_head = CTCHead(in_channels, out_channels["utf8string"], fc_decay, mid_channels, return_feats)
        
    def forward(self, x, targets=None):
        

        
        
        
        predicts = {
            "vision": {        "character": self.character_head(x["character"], targets),
                                "utf8string": self.utf8_head(x["utf8string"], targets)}
        }

        if self.return_feats:
            raise NotImplementedError
        else:
            result = predicts
        return result
        
 