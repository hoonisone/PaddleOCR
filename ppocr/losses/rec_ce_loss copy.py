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

from ppocr.postprocess import ABINetLabelDecode
from collections import Counter

class CELoss_GraphemeLabel(CELoss):
    def __init__(self,
                 smoothing=False,
                 with_all=False,
                 ignore_index=-1,
                 char_num=None,
                 loss_weight=None,
                 character_dict_path=None,
                 use_space_char=None,
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
        
                
        self.character_decode = ABINetLabelDecode(character_dict_path=character_dict_path["character"], use_space_char=True, **kwargs)
        self.utf8string_decode = ABINetLabelDecode(character_dict_path=character_dict_path["utf8string"], use_space_char=True, **kwargs)
        
        

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


        
        for model_name, model_pred in pred.items():
            if any(["head_confidence" in grapheme_name for grapheme_name in model_pred.keys()]):
                char_label = batch["text_label"]["character"]
                utf_label = batch["text_label"]["utf8string"]
                char_pred = self.character_decode(model_pred["character"])
                utf_pred = self.utf8string_decode(model_pred["utf8string"])
            
                char_acc = Counter([pred == label for (pred, _), label in zip(char_pred, char_label)])[True]/len(char_label)
                utf_acc = Counter([pred == label for (pred, _), label in zip(utf_pred, utf_label)])[True]/len(char_label)


            for grapheme_name, grapheme_pred in model_pred.items():
                if "head_confidence" in grapheme_name:                    
                    
                    confidence_label = [[char_label==char_pred, utf_label==utf_pred] for char_label, (char_pred, _), utf_label, (utf_pred, _) in zip(char_label, char_pred, utf_label, utf_pred)]
                    confidence_label = paddle.to_tensor(confidence_label, dtype="float32")############
                    
                    confidence_loss = paddle.nn.functional.binary_cross_entropy_with_logits(grapheme_pred, confidence_label)
                    loss_dict[f"{model_name}_{grapheme_name}"]=confidence_loss
                    
                else:

                    loss_dict[f"{model_name}_{grapheme_name}"] = super(CELoss_GraphemeLabel, self).forward(grapheme_pred, batch_dict[grapheme_name])["loss"]


        
        # loss_dict["loss"] = sum([loss*self.loss_weight[key.split("_")[1]] for key, loss in loss_dict.items()]) # 가중치 합
        loss_dict["loss"] = sum([loss*self.loss_weight.get(key.split("_")[1], 1) for key, loss in loss_dict.items()])
        # for k, v in loss_dict.items():
        #     print(k, v.item())
        # print()
        return loss_dict