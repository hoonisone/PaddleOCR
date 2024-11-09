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

import paddle
from paddle import nn


class CTCLoss(nn.Layer):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        # print("@@@@@@@")
        # # print(f"predicts: {predicts}")
        # print(f"type(predicts): {type(predicts)}")
        # print(f"predicts.shape: {predicts.shape}")
        # # print(f"batch: {batch}")
        # print(f"type(batch): {type(batch)}")
        # print(f"len(batch): {len(batch)}")
        # print(f"batch[0]: {batch[0]}")
        

        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor(
            [N] * B, dtype='int64', place=paddle.CPUPlace())
        
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        # print(predicts, labels, preds_lengths, label_lengths)
        # exit()
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        # preds_lengths = Tensor([128])
        # label_lengths = Tensor([128])
        if self.use_focal_loss:
            weight = paddle.exp(-loss)
            weight = paddle.subtract(paddle.to_tensor([1.0]), weight)
            weight = paddle.square(weight)
            loss = paddle.multiply(loss, weight)
        loss = loss.mean()
        # 그냥 궁금한 건데... CTC Loss가 단일 계산이긴 해도 {'CTC':xxx, 'loss':xxx} 이런식으로 하는것이 더 일관되지 않나?
        # 뒤에서 dict에 update해서 사용하기에 더 편할 것도 같고..
        return {'loss': loss}

from ppocr.postprocess import CTCLabelDecode
# from collections import Counter

class CTCLoss_Grapheme_label(CTCLoss):
    def __init__(self, 
                 use_focal_loss=False, 
                 loss_weight=None,
                 character_dict_path=None,
                 use_space_char=None,
                 **kwargs):
        super(CTCLoss_Grapheme_label, self).__init__(use_focal_loss=use_focal_loss, **kwargs)
        
        assert loss_weight is not None, "loss_weight이 필요함"
        self.loss_weight = loss_weight
        

        self.character_decode = CTCLabelDecode(character_dict_path=character_dict_path["character"], use_space_char=True, **kwargs)
        self.utf8string_decode = CTCLabelDecode(character_dict_path=character_dict_path["utf8string"], use_space_char=True, **kwargs)
        
    def forward(self, predicts, batch):
        loss_dict = {}

         
        for model_name, pred in predicts.items():
            # if any(["head_confidence" in grapheme_name for grapheme_name in pred.keys()]):
            #     char_label = batch["text_label"]["character"]
            #     utf_label = batch["text_label"]["utf8string"]
            #     char_pred = self.character_decode(pred["character"])
            #     utf_pred = self.utf8string_decode(pred["utf8string"])
                
            #     char_acc = sum([pred == label for (pred, _), label in zip(char_pred, char_label)])/len(char_label)
            #     char_acc = min(max(char_acc, 0.00001), 0.99999)
            #     utf_acc = sum([pred == label for (pred, _), label in zip(utf_pred, utf_label)])/len(utf_label)
            #     utf_acc = min(max(utf_acc, 0.00001), 0.99999)
                
                
            #     mean_acc = (char_acc+utf_acc)/2
                
                
            #     char_p_weight = min(1/char_acc, 10)
            #     char_n_weight = min(1/(1-char_acc), 10)
                
            #     utf_p_weight = min(1/utf_acc, 10)
            #     utf_n_weight = min(1/(1-utf_acc), 10)

                
            #     confidence_label = [[char_label==char_pred, utf_label==utf_pred] for char_label, (char_pred, _), utf_label, (utf_pred, _) in zip(char_label, char_pred, utf_label, utf_pred)]
            #     confidence_label = paddle.to_tensor(confidence_label, dtype="float32")############
                
            #     class_loss_weight = [[char_p_weight if char_label==char_pred else char_n_weight, 
            #                     utf_p_weight if utf_label==utf_pred else utf_n_weight] for char_label, (char_pred, _), utf_label, (utf_pred, _) in zip(char_label, char_pred, utf_label, utf_pred)]
            #     class_loss_weight = paddle.to_tensor(class_loss_weight, dtype="float32")############
                
                
                  
            #     # print(char_acc, utf_acc)
            
            for grapheme_name, g_pred in pred.items():
                # if "head_confidence" in grapheme_name:
                #     if mean_acc < 0.5:
                #         continue
                #     confidence_loss = paddle.nn.functional.binary_cross_entropy(g_pred, confidence_label, reduction="none")
                    
                #     pred_acc = g_pred*confidence_label+(1-g_pred)*(1-confidence_label)                    
                #     pred_acc = paddle.clip(pred_acc, min=0.0001, max=0.9999)
                #     # print(pred_acc)
                    
                #     focal_weight = -class_loss_weight*paddle.pow(1-pred_acc, 2.0)*paddle.log(pred_acc+0.00001)


                #     confidence_loss = focal_weight*confidence_loss

                    
                #     # for a, b, c, d, e in zip(confidence_label, g_pred, pred_acc, focal_weight, confidence_loss):
                #     #     if a[0] != a[1]:
                #     #         print(a, b, c, d, e)
                    
                #     loss_dict[f"{model_name}_{grapheme_name}"]=confidence_loss.mean()
                    
                # else:
                inner_batch = [None, batch["label"][grapheme_name], batch["length"][grapheme_name]]
                loss = super().forward(g_pred, inner_batch)["loss"]
                loss_dict[f"{model_name}_{grapheme_name}"]=loss

        loss_dict["loss"] = sum([loss*self.loss_weight[key.split("_")[1]] for key, loss in loss_dict.items()])
        return loss_dict
    

    