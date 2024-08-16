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
        
        if isinstance(pred, dict):  # for ABINet
            loss = {}
            loss_sum = []
            for name, logits in pred.items():
                if isinstance(logits, list):
                    if len(logits) == 0:
                        continue
                    logit_num = len(logits)
                    all_tgt = paddle.concat([batch["label"]] * logit_num, 0)
                    all_logits = paddle.concat(logits, 0)
                    flt_logtis = all_logits.reshape([-1, all_logits.shape[2]])
                    flt_tgt = all_tgt.reshape([-1])
                else:
                    flt_logtis = logits.reshape([-1, logits.shape[2]])
                    flt_tgt = batch["label"].reshape([-1])
                loss[name + '_loss'] = self.loss_func(flt_logtis, flt_tgt)
                loss_sum.append(loss[name + '_loss'])
            loss['loss'] = sum(loss_sum)
            return loss
        else:
            if self.with_all:  # for ViTSTR
                tgt = batch["label"]
                pred = pred.reshape([-1, pred.shape[2]])
                tgt = tgt.reshape([-1])
                loss = self.loss_func(pred, tgt)
                return {'loss': loss}
            else:  # for NRTR
                max_len = batch[2].max()
                tgt = batch["label"][:, 1:2 + max_len]
                pred = pred.reshape([-1, pred.shape[2]])
                tgt = tgt.reshape([-1])
                if self.smoothing:
                    eps = 0.1
                    n_class = pred.shape[1]
                    one_hot = F.one_hot(tgt, pred.shape[1])
                    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (
                        n_class - 1)
                    log_prb = F.log_softmax(pred, axis=1)
                    non_pad_mask = paddle.not_equal(
                        tgt, paddle.zeros(
                            tgt.shape, dtype=tgt.dtype))
                    loss = -(one_hot * log_prb).sum(axis=1)
                    loss = loss.masked_select(non_pad_mask).mean()
                else:
                    loss = self.loss_func(pred, tgt)
                return {'loss': loss}

class CELoss_GraphemeLabel(CELoss):
    def __init__(self,
                 smoothing=False,
                 with_all=False,
                 ignore_index=-1,
                 class_num_dict=None,
                 **kwargs):
        super(CELoss_GraphemeLabel, self).__init__(
            smoothing=smoothing,
            with_all=with_all,
            ignore_index=ignore_index,
            **kwargs)

        assert class_num_dict is not None, "class_num_dict should not be None"
        self.class_num_dict = class_num_dict

    def split_grapheme_logits(self, x):
        self.class_num_dict
        
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
    

    def split_pred(self, pred):
        
        graphemes = self.class_num_dict.keys()
        pred_dict = {grapheme: {"align":[], "lang":[], "vision":None} for grapheme in graphemes}
        
        for key, value in pred.items():
            if isinstance(value, list): # lang, aligh
                for i, v in enumerate(value):
                    logit_dict = self.split_grapheme_logits(v)
                    for grapheme in logit_dict.keys():
                        pred_dict[grapheme][key].append(logit_dict[grapheme])
            else: # vision
                logit_dict = self.split_grapheme_logits(value)
                for grapheme in logit_dict.keys():
                    pred_dict[grapheme][key] = logit_dict[grapheme]                    
        return pred_dict
    
    def forward(self, pred, batch):
        pred_dict = self.split_pred(pred)
        batch_dict = {grapheme: {"label":value} for grapheme, value in batch["label"].items()}
        
        graphemes = self.class_num_dict.keys()
        loss_dict = {grapheme:super(CELoss_GraphemeLabel, self).forward(pred_dict[grapheme], batch_dict[grapheme]) for grapheme in graphemes}
        loss_dict["loss"] = sum([loss["loss"] for loss in loss_dict.values()])
        
        return loss_dict