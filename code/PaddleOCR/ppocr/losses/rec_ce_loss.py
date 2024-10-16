import paddle
from paddle import nn
import paddle.nn.functional as F


class CELoss(nn.Layer):
    def __init__(self,
                 smoothing=False,
                 with_all=False,
                 ignore_index=-1,
                 **kwargs):
        super(CELoss, self).__init__()
        if ignore_index >= 0:
            self.loss_func = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=ignore_index)
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.smoothing = smoothing
        self.with_all = with_all

    def forward(self, pred, batch):
        flt_logtis = pred.reshape([-1, pred.shape[2]])
        flt_tgt = batch["label"].reshape([-1])
        loss = self.loss_func(flt_logtis, flt_tgt)
        return {'loss': loss}

class CELoss_GraphemeLabel(CELoss):
    def __init__(self,
                 smoothing=False,
                 with_all=False,
                 ignore_index=-1,
                 char_num=None,
                 loss_weight=None,
                 **kwargs):
        super(CELoss_GraphemeLabel, self).__init__(
            smoothing=smoothing,
            with_all=with_all,
            ignore_index=ignore_index,
            **kwargs)

        assert char_num is not None, "class_num_dict should not be None"
        self.class_num_dict = char_num
        assert loss_weight is not None, "loss_weight_dict should not be None"
        self.loss_weight = loss_weight

    def split_grapheme_logits(self, x):
        """ x는 (initial, medial, final) vecter가 concat된 형태
            이를 각 vecter의 길이에 맞게 나누어 dict 형태로 반환
        """
        
        index_ranges = {}
        start_index = 0
        for key, length in self.class_num_dict.items():
            index_ranges[key] = (start_index, start_index + length)
            start_index += length
    
        dict_out = {key : x[:, :, start:end] for key, (start, end) in index_ranges.items()}
        return dict_out
    
    # def concat_grapheme_logits(self, f, m, l, axis=-1):
    #     return paddle.concat([f, m, l], axis=axis)
    
    # def softmax_grapheme_logits(self, x, axis=-1):
    #     grapheme_logits = self.split_grapheme_logits(x)
    #     grapheme_logits = [F.softmax(grapheme_logit, -1) for grapheme_logit in grapheme_logits]
    #     return self.concat_grapheme_logits(*grapheme_logits, axis=axis)
    

    # def split_pred(self, pred):    
    #     pred_dict = {key: self.split_grapheme_logits(value) for key, value in pred.items()}
    #     return pred_dict
    #     """return  = 
    #     {
    #         "Vision": {
    #             "character": Tensor(shape=[N, L, character_C]), 
    #             "initial": Tensor(shape=[N, L, initial_C]),
    #             "medial": Tensor(shape=[N, L, initial_C]),
    #             "final": Tensor(shape=[N, L, initial_C])
    #         },
    #         "Align1": {
    #             ...
    #         },
    #     }
    #     """
    
    
    def forward(self, pred, batch):
        """return  = 
        {
            "Vision": {
                "character": Tensor(shape=[N, L, character_C]), 
                "initial": Tensor(shape=[N, L, initial_C]),
                "medial": Tensor(shape=[N, L, initial_C]),
                "final": Tensor(shape=[N, L, initial_C])
            },
            "Align1": {
                ...
            },
        }
        """
        # pred {}
    
        # pred_dict = self.split_pred(pred)
        
        batch_dict = {grapheme: {"label":value} for grapheme, value in batch["label"].items()}
        
        graphemes = self.class_num_dict.keys()
        
        loss_dict = dict()

        for model, model_preds in pred.items():
            for grapheme in graphemes:
                loss_dict[f"{model}_{grapheme}"] = super(CELoss_GraphemeLabel, self).forward(model_preds[grapheme], batch_dict[grapheme])["loss"]

        
        
        loss_dict["loss"] = sum([loss*self.loss_weight[key.split("_")[1]] for key, loss in loss_dict.items()]) # 가중치 합
        # for k, v in loss_dict.items():
        #     print(k, v.item())
        # print()
        return loss_dict