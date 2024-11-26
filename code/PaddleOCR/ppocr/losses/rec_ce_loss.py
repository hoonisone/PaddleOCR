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
        
        if "character" in self.class_num_dict:
            self.character_decode = ABINetLabelDecode(character_dict_path=character_dict_path["character"], use_space_char=True, **kwargs)
        if "utf8string" in self.class_num_dict:
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

    
    
    # def forward(self, predicts, batch):
        
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
    #     # pred {}
    
    #     # pred_dict = self.split_pred(pred)

    #     batch_dict = {grapheme: {"label":value} for grapheme, value in batch["label"].items()}
        
    #     graphemes = self.class_num_dict.keys()
        
    #     loss_dict = dict()
    #     loss_dict["loss"] = {}
    #     loss_dict_others = dict()

        
    #     for model_name, pred in predicts.items():
    #         if any(["head_confidence" in grapheme_name for grapheme_name in pred.keys()]):
    #             char_label = batch["text_label"]["character"]
    #             utf_label = batch["text_label"]["utf8string"]
    #             char_pred = self.character_decode(pred["character"])
    #             utf_pred = self.utf8string_decode(pred["utf8string"])
                
    #             char_acc = sum([pred == label for (pred, _), label in zip(char_pred, char_label)])/len(char_label)
    #             char_acc = min(max(char_acc, 0.00001), 0.99999)
    #             utf_acc = sum([pred == label for (pred, _), label in zip(utf_pred, utf_label)])/len(utf_label)
    #             utf_acc = min(max(utf_acc, 0.00001), 0.99999)
                
    #             mean_acc = (char_acc+utf_acc)/2
                
    #             char_p_weight = min(1/char_acc, 10)
    #             char_n_weight = min(1/(1-char_acc), 10)
                
    #             utf_p_weight = min(1/utf_acc, 10)
    #             utf_n_weight = min(1/(1-utf_acc), 10)

                
    #             confidence_label = [[char_label==char_pred, utf_label==utf_pred] for char_label, (char_pred, _), utf_label, (utf_pred, _) in zip(char_label, char_pred, utf_label, utf_pred)]
    #             confidence_label = paddle.to_tensor(confidence_label, dtype="float32")############
                
    #             class_loss_weight = [[char_p_weight if char_label==char_pred else char_n_weight, 
    #                             utf_p_weight if utf_label==utf_pred else utf_n_weight] for char_label, (char_pred, _), utf_label, (utf_pred, _) in zip(char_label, char_pred, utf_label, utf_pred)]
    #             class_loss_weight = paddle.to_tensor(class_loss_weight, dtype="float32")############
                
                
                  
    #             # print(char_acc, utf_acc)
            
    #         for grapheme_name, g_pred in pred.items():
    #             if "head_confidence" in grapheme_name:
    #                 if mean_acc < 0.9:
    #                     continue
    #                 confidence_loss = paddle.nn.functional.binary_cross_entropy(g_pred, confidence_label, reduction="none")
                    
    #                 pred_acc = g_pred*confidence_label+(1-g_pred)*(1-confidence_label)                    
    #                 pred_acc = paddle.clip(pred_acc, min=0.0001, max=0.9999)
    #                 # print(pred_acc)
                    
    #                 focal_weight = -class_loss_weight*paddle.pow(1-pred_acc, 2.0)*paddle.log(pred_acc+0.00001)
    #                 confidence_loss = focal_weight*confidence_loss
                    

                    
    #                 # for a, b, c, d, e in zip(confidence_label, g_pred, pred_acc, focal_weight, confidence_loss):
    #                 #     if a[0] != a[1]:
    #                 #         print(a, b, c, d, e)

    #                 loss = confidence_loss.mean()
    #                 loss_dict[f"{model_name}_{grapheme_name}"]=loss
    #                 loss_dict_others[f"{model_name}_{grapheme_name}"] = loss*0.0001
                
                    
    #             else:
    #                 loss = super(CELoss_GraphemeLabel, self).forward(g_pred, batch_dict[grapheme_name])["loss"]
    #                 loss_dict[f"{model_name}_{grapheme_name}"] = loss
    #                 loss_dict["loss"][f"{model_name}_{grapheme_name}"] = loss

        
    #     loss_dict["loss"] = sum([loss*self.loss_weight[key.split("_")[1]] for key, loss in loss_dict["loss"].items()]) # 가중치 합
    #     # loss_dict["loss"] = sum(loss_dict["loss"])/len(loss_dict["loss"])

        
    #     # for k, v in loss_dict.items():
    #     #     print(k, v.item())
    #     # print()
    #     return loss_dict, loss_dict_others
    def forward(self, predicts, batch):
        
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

        # loss_dict_others = dict()

        
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
                
                
                  
                # print(char_acc, utf_acc)
            
            for grapheme_name, g_pred in pred.items():
                # if "head_confidence" in grapheme_name:
                #     if mean_acc < 0.9:
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

                #     loss = confidence_loss.mean()
                #     loss_dict[f"{model_name}_{grapheme_name}"]=loss

            
                # else:
                loss = super(CELoss_GraphemeLabel, self).forward(g_pred, batch_dict[grapheme_name])["loss"]
                loss_dict[f"{model_name}_{grapheme_name}"] = loss

        
        loss_dict["loss"] = sum([loss*self.loss_weight[key.split("_")[1]] for key, loss in loss_dict.items()])
        return loss_dict
    
        # return loss_dict, None