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


class CTCLoss_Grapheme_label(CTCLoss):
    def __init__(self, 
                 use_focal_loss=False, 
                 loss_weight=None,
                 **kwargs):
        super(CTCLoss_Grapheme_label, self).__init__(use_focal_loss=use_focal_loss, **kwargs)
        
        assert loss_weight is not None, "loss_weight이 필요함"
        self.loss_weight = loss_weight
        
    def forward(self, predicts, batch):
        loss_dict = {}
        for model_name, pred in predicts.items():
            for grapheme_name, g_pred in pred.items():
                inner_batch = [None, batch["label"][grapheme_name], batch["length"][grapheme_name]]
                loss = super().forward(g_pred, inner_batch)["loss"]
                loss_dict[f"{model_name}_{grapheme_name}"]=loss


        loss_dict["loss"] = sum([loss*self.loss_weight[key.split("_")[1]] for key, loss in loss_dict.items()])
        return loss_dict
        # print(predicts.keys())
        # for model, model_preds in pred_dict.items():
        #     for grapheme in graphemes:
                
        #         loss_dict[f"{model}_{grapheme}"] = super(CELoss_GraphemeLabel, self).forward(model_preds[grapheme], batch_dict[grapheme])["loss"]

        # if isinstance(predicts, (list, tuple)):
        #     predicts = predicts[-1]
        # predicts = predicts.transpose((1, 0, 2))
        # N, B, _ = predicts.shape
        # preds_lengths = paddle.to_tensor(
        #     [N] * B, dtype='int64', place=paddle.CPUPlace())
        
        # labels = batch["label"].astype("int32")
        # label_lengths = batch["length"].astype('int64')
        
        
        # print(predicts, labels, preds_lengths, label_lengths)
        # exit()
        # loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        # # preds_lengths = Tensor([128])
        # # label_lengths = Tensor([128])
        # if self.use_focal_loss:
        #     weight = paddle.exp(-loss)
        #     weight = paddle.subtract(paddle.to_tensor([1.0]), weight)
        #     weight = paddle.square(weight)
        #     loss = paddle.multiply(loss, weight)
        # loss = loss.mean()
        # 그냥 궁금한 건데... CTC Loss가 단일 계산이긴 해도 {'CTC':xxx, 'loss':xxx} 이런식으로 하는것이 더 일관되지 않나?
        # 뒤에서 dict에 update해서 사용하기에 더 편할 것도 같고..
        return {'loss': 1}