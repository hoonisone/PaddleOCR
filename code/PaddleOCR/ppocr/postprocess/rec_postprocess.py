# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from ast import Return
from importlib import abc
from matplotlib import use
import numpy as np
import paddle
from paddle.nn import functional as F
import re
from ppocr.utils.korean_compose_by_utf8 import compose_string_by_utf8, char_level_ensemble, word_level_ensemble, word_level_ensemble_by_threshold, char_level_ensemble_by_threshold

def softmax(x, axis=None):
    # Overflow 방지를 위해 입력 값에서 최대값을 뺀 후 지수 계산
    # Keepdims=True는 차원을 유지하여 broadcasting이 가능하도록 함
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False, use_unkown=False):
        self.beg_str = "<BOS>"
        self.end_str = "<EOS>"
        self.unkown_str = "<UNK>"
        self.reverse = False
        self.character_str = []
        self.use_unkown = use_unkown

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        self.char_num = len(self.character)

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            if not isinstance(conf_list, list):
                conf_list = conf_list.tolist()
            
            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)
            result_list.append((text, conf_list))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if preds is not None:
            if isinstance(preds, tuple) or isinstance(preds, list):
                preds = preds[-1]
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()
            
            preds = softmax(preds, axis=2)
                
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)

            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        else:
            text = None
            
        if label is None:
            return text
        
        label = self.decode(label)
    
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

class CTCLabelDecode_TEST(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode_TEST, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if preds is not None:
            if isinstance(preds, tuple) or isinstance(preds, list):
                preds = preds[-1]
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()

            preds = softmax(preds, axis=2)
                
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        else:
            text = None
            
        if label is None:
            return text
        
        if isinstance(label, paddle.Tensor):
            label = label.numpy()
        
        label = self.decode(label)
    
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

class CTCLabelDecode_GraphemeLabel_old(object):
    """ Convert between text-label and text-index """

    def __init__(self, handling_grapheme, character_dict_path=None, use_space_char=False,
                 **kwargs):
        self.grapheme = handling_grapheme
        self.decode_dict = {
            grphame:CTCLabelDecode(character_dict_path = character_dict_path[grphame],
                                           use_space_char = use_space_char,
                                           **kwargs)
            for grphame in self.grapheme
        }

        self.character = {grapheme: self.decode_dict[grapheme].character for grapheme in self.grapheme}
        self.char_num = {grapheme: self.decode_dict[grapheme].char_num for grapheme in self.grapheme}
    
    def compose_character(self, preds, label=None):
        initials = preds["initial"]
        medials = preds["medial"]
        finals = preds["final"]
        


        if label is None: # no label
            composed = list()
            for (i, ip), (m, mp), (f, fp) in zip(initials, medials, finals):                
                print(i, m, f, ip, mp, fp)
                label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)
                composed.append((label, p))
            return composed
                
        else: # with label
            gt_composed = list()
            composed = list()
            for (i, ip), (gt_i, gt_ip), (m, mp), (gt_m, gt_mp), (f, fp), (gt_f, gt_fp) in zip(*initials, *medials, *finals):
                gt_label, gt_p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(gt_i, gt_m, gt_f, gt_ip, gt_mp, gt_fp)
                gt_composed.append((gt_label, gt_p))
                
                label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)                    
                composed.append((label, p))
            return composed, gt_composed
     
    
    def __call__(self, preds, label=None, *args, **kwargs):
        
        # print(preds)
        result = dict()
        

        
        for model_name, pred in preds.items():
            model_result = dict()
            for grapheme in self.grapheme:
                if label != None:
                    arg_label=label[grapheme].numpy()
                else:
                    arg_label = None
                model_result[grapheme] = self.decode_dict[grapheme](pred[grapheme], label=arg_label, *args, **kwargs)
            
            if all([name in self.decode_dict.keys() for name in ["initial", "medial", "final"]]):
                
                model_result["composed"] = self.compose_character(model_result, label)
            result[model_name] = model_result
            

        return result


    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

class CTCLabelDecode_GraphemeLabel(object):
    """ Convert between text-label and text-index """

    def __init__(self, handling_grapheme, character_dict_path=None, use_space_char=False,
                 **kwargs):
        self.grapheme = handling_grapheme
        self.decode_dict = {
            grphame:CTCLabelDecode(character_dict_path = character_dict_path[grphame],
                                           use_space_char = use_space_char,
                                           **kwargs)
            for grphame in self.grapheme
        }

        self.character = {grapheme: self.decode_dict[grapheme].character for grapheme in self.grapheme}
        self.char_num = {grapheme: self.decode_dict[grapheme].char_num for grapheme in self.grapheme}
        self.c_th_list = [0.01, 0.03, 0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        self.g_th_list = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
        
        
    def compose_character(self, preds, label=None):
        initials = preds["initial"]
        medials = preds["medial"]
        finals = preds["final"]
        


        if label is None: # no label
            composed = list()
            for (i, ip), (m, mp), (f, fp) in zip(initials, medials, finals):                
                print(i, m, f, ip, mp, fp)
                label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)
                composed.append((label, p))
            return composed
                
        else: # with label
            gt_composed = list()
            composed = list()
            for (i, ip), (gt_i, gt_ip), (m, mp), (gt_m, gt_mp), (f, fp), (gt_f, gt_fp) in zip(*initials, *medials, *finals):
                gt_label, gt_p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(gt_i, gt_m, gt_f, gt_ip, gt_mp, gt_fp)
                gt_composed.append((gt_label, gt_p))
                
                label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)                    
                composed.append((label, p))
            return composed, gt_composed
     
    
    def __call__(self, preds, label=None, *args, **kwargs):
        
        # print(preds)
        result = dict()
        
        for model_name, pred in preds.items():
            # if model_name == "head_confidence":
            #     continue
            model_result = dict()
            for grapheme in self.grapheme:
                if label != None:
                    arg_label=label[grapheme].numpy()
                else:
                    arg_label = None
                model_result[grapheme] = self.decode_dict[grapheme](pred[grapheme], label=arg_label, *args, **kwargs)
            
            ### 이 부분만 추가함 !!!!!!!!!!!!!!!!!!!!!!!!!!!
            if "utf8string" in model_result:
                model_result["utf8composed"] = self.composed_utf8_string(model_result, label)
            
            if all([name in self.decode_dict.keys() for name in ["initial", "medial", "final"]]):                
                model_result["composed"] = self.compose_character(model_result, label)
            
            
            if label == None:
                ### Ensemble
                if "utf8composed" in model_result and "character" in model_result:
                    
                    char_pred = model_result["character"]
                    utf8_pred = model_result["utf8composed"]
                    #########
                    ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
                    model_result["ensemble(c+g_utf8)_by_char"] = ensemble
                    #########

                    ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
                    model_result["ensemble(c+g_utf8)_by_word"] = ensemble
                                        
                    for threshold in self.c_th_list:
                        ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_char_on_char_({threshold})"] = ensemble
                        
                        ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_word_on_char_({threshold})"] = ensemble
                        
                    
                    for threshold in self.g_th_list:
                        ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="right" ) for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_char_on_utf_({threshold})"] = ensemble
                        
                        ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="right") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_word_on_utf_({threshold})"] = ensemble

                    
                    
                if "composed" in model_result and "character" in model_result:
                    char_pred = model_result["character"]
                    composed_pred = model_result["composed"]

                    #########
                    ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
                    model_result["ensemble(c+g)_by_char"] = ensemble
                    #########
                    ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
                    model_result["ensemble(c+g)_by_word"] = ensemble

            else:
                ### Ensemble
                if "utf8composed" in model_result and "character" in model_result:
                    
                    char_pred = model_result["character"][0]
                    utf8_pred = model_result["utf8composed"][0]
                    gt = model_result["character"][1]
                    
                    #########
                    ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
                    model_result["ensemble(c+g_utf8)_by_char"] = [ensemble, gt]
                    #########
                    ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
                    model_result["ensemble(c+g_utf8)_by_word"] = [ensemble, gt]
                    
                    
                    for threshold in self.c_th_list:
                        ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_char_on_char_({threshold})"] = [ensemble, gt]
                        
                        ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_word_on_char_({threshold})"] = [ensemble, gt]
                        
                    

                    
                    for threshold in self.g_th_list:
                        ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="right" ) for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_char_on_utf_({threshold})"] = [ensemble, gt]
                        
                        ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="right") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g_utf8)_by_word_on_utf_({threshold})"] = [ensemble, gt]
                        


                    # compare_label = [pred.replace(" ", "") == gt.replace(" ", "") for (pred, _), (gt, _) in zip(utf8_pred, gt)]
                    # compare_pred = [g_conf >= 0.5 for c_conf, g_conf in pred[f"head_confidence_k_5"]]
                    
                    # print(f"min= {paddle.min()}")
                    
                    # for a, (c, g) in zip(compare_label, pred[f"head_confidence_k_5"]):
                    #     if a == False:
                    #         print(float(g))
                    
                    # answer = 0
                    # for a, b in zip(compare_label, compare_pred):
                        
                    #     if a == b:
                    #         answer += 1
                    # print(f"#######{answer/len(compare_label)}")

                    # ensemble = [c if compare >= 0.45 else utf8 for c, utf8, (compare) in zip(char_pred, utf8_pred, pred[f"head_compare_full"])]
                    # model_result[f"ensemble(c+g_utf8)_by_compare_full"] = [ensemble, gt]
                    
                    # compare = pred["head_compare_full"]
                    
                    # comapre_gt = []
                    # for char_pred, utf8_pred, gt in zip(char_pred, utf8_pred, gt):
                    #     char_pred = char_pred.replace(" ", "")
                    #     utf8_pred = utf8_pred.replace(" ", "")
                    #     gt = gt.replace(" ", "")
                        
                    
                    # model_result["head_compare_full"] = [compare > 0.45 for compare in pred["head_compare_full"]]
                    # print(compare.shape)
                    # print(compare>0.45)
                    # for a in compare:
                    #     print(a)
                    # exit()
                    # for k in [3, 5, 10, 15, 20]:                
                    #     ensemble = [c if char_conf >= grapheme_conf else utf8 for c, utf8, (char_conf, grapheme_conf) in zip(char_pred, utf8_pred, pred[f"head_confidence_k_{k}"])]
                    #     model_result[f"ensemble(c+g_utf8)_by_logit_k_{k}"] = [ensemble, gt]
                        
                    #     ensemble = [c if char_conf >= grapheme_conf else utf8 for c, utf8, (char_conf, grapheme_conf) in zip(char_pred, utf8_pred, pred[f"head_confidence2_k_{k}"])]
                    #     model_result[f"ensemble2(c+g_utf8)_by_logit_k_{k}"] = [ensemble, gt]                        
                    #     # if k==10:
                    #     #     ensemble = [c if char_conf >= grapheme_conf else utf8 for c, utf8, (char_conf, grapheme_conf) in zip(char_pred, utf8_pred, pred[f"head_confidence2_k_{k}"])]
                    #     #     model_result[f"ensemble2(c+g_utf8)_by_logit_k_{k}"] = [ensemble, gt]
                
                
                
                
                    # ensemble = [utf8 if grapheme_conf >= 0.7 else c for c, utf8, (char_conf, grapheme_conf) in zip(char_pred, utf8_pred, pred[f"head_confidence_k_5"])]
                    # model_result[f"ensemble(c+g_utf8)_by_logit_full"] = [ensemble, gt]
                    
                    # ensemble = [c if char_conf >= grapheme_conf else utf8 for c, utf8, (char_conf, grapheme_conf) in zip(char_pred, utf8_pred, pred[f"head_confidence2_full"])]
                    # model_result[f"ensemble2(c+g_utf8)_by_logit_full"] = [ensemble, gt]                
                    
                    
                if "composed" in model_result and "character" in model_result:
                    raise NotImplementedError
                    char_pred = model_result["character"][0]
                    composed_pred = model_result["composed"][0]
                    gt = model_result["character"][1]
                    #########
                    ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
                    model_result["ensemble(c+g)_by_char"] = [ensemble, gt]
                    #########
                    ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
                    model_result["ensemble(c+g)_by_word"] = [ensemble, gt]
                    

                    for threshold in self.c_th_list:
                        ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g)_by_char_on_char_({threshold})"] = [ensemble, gt]
                        
                        ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g)_by_word_on_char_({threshold})"] = [ensemble, gt]
                        
                    

                    
                    for threshold in self.g_th_list:
                        ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="right" ) for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g)_by_char_on_utf_({threshold})"] = [ensemble, gt]
                        
                        ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="right") for c, utf8 in zip(char_pred, utf8_pred)]
                        model_result[f"ensemble(c+g)_by_word_on_utf_({threshold})"] = [ensemble, gt]
                        
                    

            result[model_name] = model_result
            
        return result

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def composed_utf8_string(self, preds, label=None):
        utf8string = preds["utf8string"]
        
        
        if label is None: # no label
            composed = [compose_string_by_utf8(pred, pred_p) for pred, pred_p in utf8string]
            return composed
            # raise NotImplementedError
            # composed = list()
            # for pred, prob in utf8string:
              
            #     print(i, m, f, ip, mp, fp)
            #     label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)
            #     composed.append((label, p))
            # return composed
                
        else: # with label
            composed = [compose_string_by_utf8(pred, pred_p) for pred, pred_p in utf8string[0]]
            gt_composed = [compose_string_by_utf8(gt, gt_p) for gt, gt_p in utf8string[1]]

            return composed, gt_composed


# class CTCLabelDecode_GraphemeLabel_utf8(object):
#     """ Convert between text-label and text-index """

#     def __init__(self, handling_grapheme, character_dict_path=None, use_space_char=False,
#                  **kwargs):
#         self.grapheme = handling_grapheme
#         self.decode_dict = {
#             grphame:CTCLabelDecode(character_dict_path = character_dict_path[grphame],
#                                            use_space_char = use_space_char,
#                                            **kwargs)
#             for grphame in self.grapheme
#         }

#         self.character = {grapheme: self.decode_dict[grapheme].character for grapheme in self.grapheme}
#         self.char_num = {grapheme: self.decode_dict[grapheme].char_num for grapheme in self.grapheme}
    
#     def compose_character(self, preds, label=None):
#         initials = preds["initial"]
#         medials = preds["medial"]
#         finals = preds["final"]
        

#         if label is None: # no label
#             composed = list()
#             for (i, ip), (m, mp), (f, fp) in zip(initials, medials, finals):                
#                 print(i, m, f, ip, mp, fp)
#                 label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)
#                 composed.append((label, p))
#             return composed
                
#         else: # with label
#             gt_composed = list()
#             composed = list()
#             for (i, ip), (gt_i, gt_ip), (m, mp), (gt_m, gt_mp), (f, fp), (gt_f, gt_fp) in zip(*initials, *medials, *finals):
#                 gt_label, gt_p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(gt_i, gt_m, gt_f, gt_ip, gt_mp, gt_fp)
#                 gt_composed.append((gt_label, gt_p))
                
#                 label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)                    
#                 composed.append((label, p))
#             return composed, gt_composed
     
    
#     def __call__(self, preds, label=None, *args, **kwargs):
        
#         # print(preds)
#         result = dict()
        
#         for model_name, pred in preds.items():
#             model_result = dict()
#             for grapheme in self.grapheme:
#                 if label != None:
#                     arg_label=label[grapheme].numpy()
#                 else:
#                     arg_label = None
#                 model_result[grapheme] = self.decode_dict[grapheme](pred[grapheme], label=arg_label, *args, **kwargs)
            
#             ### 이 부분만 추가함 !!!!!!!!!!!!!!!!!!!!!!!!!!!
#             if "utf8string" in model_result:
#                 model_result["utf8composed"] = self.composed_utf8_string(model_result, label)
            
#             if all([name in self.decode_dict.keys() for name in ["initial", "medial", "final"]]):                
#                 model_result["composed"] = self.compose_character(model_result, label)
            
#             ### Ensemble
#             if "utf8composed" in model_result and "character" in model_result:
                
#                 char_pred = model_result["character"][0]
#                 utf8_pred = model_result["utf8composed"][0]
#                 gt = model_result["character"][1]
#                 #########
#                 ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
#                 model_result["ensemble(c+g_utf8)_by_char"] = [ensemble, gt]
#                 #########
#                 ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
#                 model_result["ensemble(c+g_utf8)_by_word"] = [ensemble, gt]
                
#             if "composed" in model_result and "character" in model_result:
#                 char_pred = model_result["character"][0]
#                 composed_pred = model_result["composed"][0]
#                 gt = model_result["character"][1]
#                 #########
#                 ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
#                 model_result["ensemble(c+g)_by_char"] = [ensemble, gt]
#                 #########
#                 ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
#                 model_result["ensemble(c+g)_by_word"] = [ensemble, gt]

#             result[model_name] = model_result
    

    
#         return result
    
    
#     def composed_utf8_string(self, preds, label=None):
#         utf8string = preds["utf8string"]
        
        
#         if label is None: # no label
#             raise NotImplementedError
#             # composed = list()
#             # for pred, prob in utf8string:
              
#             #     print(i, m, f, ip, mp, fp)
#             #     label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)
#             #     composed.append((label, p))
#             # return composed
                
#         else: # with label
#             composed = [compose_string_by_utf8(pred, pred_p) for pred, pred_p in utf8string[0]]
#             gt_composed = [compose_string_by_utf8(gt, gt_p) for gt, gt_p in utf8string[1]]

#             return composed, gt_composed
        
    

#     def add_special_char(self, dict_character):
#         dict_character = ['blank'] + dict_character
#         return dict_character

class CTCLabelDecode_Grapheme(object):               ## 옛날 CTC 버전
    pass
    # """ Convert between text-label and text-index """

    # def __init__(self, handling_grapheme, character_dict_path=None, use_space_char=False,
    #              **kwargs):
    #     self.grapheme = handling_grapheme
    #     self.decode_dict = {
    #         grphame:CTCLabelDecode(character_dict_path = character_dict_path[grphame],
    #                                        use_space_char = use_space_char,
    #                                        **kwargs)
    #         for grphame in self.grapheme
    #     }
    #     # self.first_decode = CTCLabelDecode(character_dict_path = character_dict_path[0],
    #     #                                    use_space_char = use_space_char,
    #     #                                    **kwargs)
    #     # self.second_decode = CTCLabelDecode(character_dict_path = character_dict_path[1],
    #     #                             use_space_char = use_space_char,
    #     #                             **kwargs)
    #     # self.third_decode = CTCLabelDecode(character_dict_path = character_dict_path[2],
    #     #                     use_space_char = use_space_char,
    #     #                     **kwargs)
    #     # self.origin_decode = CTCLabelDecode(character_dict_path = character_dict_path[3],
    #     #                     use_space_char = use_space_char,
    #     #                     **kwargs)
    #     self.character = {grapheme: self.decode_dict[grapheme].character for grapheme in self.grapheme}
    #     self.char_num = {grapheme: self.decode_dict[grapheme].char_num for grapheme in self.grapheme}
    #     # self.character = [self.first_decode.character, self.second_decode.character, self.third_decode.character, self.origin_decode.character] # character 속성이 있는지를 통해 조건을 체크하는 부분이 있어서 CTCLabelDecode와 동일하게 생성해줌
    
    # def compose_character(self, preds, label=None):
    #     initials = preds["initial"]
    #     medials = preds["medial"]
    #     finals = preds["final"]
        
        
    #     if label is None: # no label
    #         composed = list()
    #         for (i, ip), (m, mp), (f, fp) in zip(initials, medials, finals):                
    #             print(i, m, f, ip, mp, fp)
    #             label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)
    #             composed.append((label, p))
    #         return composed
                
    #     else: # with label
    #         gt_composed = list()
    #         composed = list()
    #         for (i, ip), (gt_i, gt_ip), (m, mp), (gt_m, gt_mp), (f, fp), (gt_f, gt_fp) in zip(*initials, *medials, *finals):
    #             gt_label, gt_p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(gt_i, gt_m, gt_f, gt_ip, gt_mp, gt_fp)
    #             gt_composed.append((gt_label, gt_p))
                
    #             label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)                    
    #             composed.append((label, p))
    #         return composed, gt_composed
     
    
    # def __call__(self, preds, label=None, *args, **kwargs):
    #     result = dict()
    #     for grapheme in self.grapheme:
    #         if label != None:
    #             arg_label=label[f"{grapheme}_label"].numpy()
    #         else:
    #             arg_label = None
    #         result[grapheme] = self.decode_dict[grapheme](preds[grapheme], label=arg_label, *args, **kwargs)
        
    #     if all([name in self.decode_dict.keys() for name in ["initial", "medial", "final"]]):
            
    #         result["composed"] = self.compose_character(result, label)
            

        
        
    #     # first = self.first_decode(preds[0], label=label[0], *args, **kwargs)
    #     # second = self.second_decode(preds[1], label=label[1], *args, **kwargs)
    #     # third = self.third_decode(preds[2], label=label[2], *args, **kwargs)
        
        
    #     # texts = {"first":first[0], "second":second[0], "third":third[0]}
    #     # labels = {"first":first[1], "second":second[1], "third":third[1]}
    #     # print(result)
    #     # exit()
    #     # texts = {grapheme: result[grapheme][0] for grapheme in self.grapheme}
    #     # print(result["character"])
    #     # exit()
    #     # try: 
    #     #     labels = {grapheme: result[grapheme][1] for grapheme in self.grapheme}
    #     #     a = {grapheme: (result[grapheme][1]) for grapheme in self.grapheme}
            
            
    #     # except: # 추론형인 레이블 없음
    #     #     labels = None
    #     return {"vision": result}

    # def add_special_char(self, dict_character):
    #     dict_character = ['blank'] + dict_character
    #     return dict_character


class DistillationCTCLabelDecode(CTCLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 model_name=["student"],
                 key=None,
                 multi_head=False,
                 **kwargs):
        super(DistillationCTCLabelDecode, self).__init__(character_dict_path,
                                                         use_space_char)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            if self.multi_head and isinstance(pred, dict):
                pred = pred['ctc']
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class AttnLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(AttnLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class RFLLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(RFLLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        # if seq_outputs is not None:
        if isinstance(preds, tuple) or isinstance(preds, list):
            cnt_outputs, seq_outputs = preds
            if isinstance(seq_outputs, paddle.Tensor):
                seq_outputs = seq_outputs.numpy()
            preds_idx = seq_outputs.argmax(axis=2)
            preds_prob = seq_outputs.max(axis=2)
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

            if label is None:
                return text
            label = self.decode(label, is_remove_duplicate=False)
            return text, label

        else:
            cnt_outputs = preds
            if isinstance(cnt_outputs, paddle.Tensor):
                cnt_outputs = cnt_outputs.numpy()
            cnt_length = []
            for lens in cnt_outputs:
                length = round(np.sum(lens))
                cnt_length.append(length)
            if label is None:
                return cnt_length
            label = self.decode(label, is_remove_duplicate=False)
            length = [len(res[0]) for res in label]
            return cnt_length, length

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SEEDLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SEEDLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        self.padding_str = "padding"
        self.end_str = "eos"
        self.unknown = "unknown"
        dict_character = dict_character + [
            self.end_str, self.padding_str, self.unknown
        ]
        return dict_character

    def get_ignored_tokens(self):
        end_idx = self.get_beg_end_flag_idx("eos")
        return [end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "sos":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "eos":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" % beg_or_end
        return idx

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        [end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        preds_idx = preds["rec_pred"]
        if isinstance(preds_idx, paddle.Tensor):
            preds_idx = preds_idx.numpy()
        if "rec_pred_scores" in preds:
            preds_idx = preds["rec_pred"]
            preds_prob = preds["rec_pred_scores"]
        else:
            preds_idx = preds["rec_pred"].argmax(axis=2)
            preds_prob = preds["rec_pred"].max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label


class SRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SRNLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)
        self.max_text_length = kwargs.get('max_text_length', 25)

    def __call__(self, preds, label=None, *args, **kwargs):
        pred = preds['predict']
        char_num = len(self.character_str) + 2
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = np.reshape(pred, [-1, char_num])

        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)

        preds_idx = np.reshape(preds_idx, [-1, self.max_text_length])

        preds_prob = np.reshape(preds_prob, [-1, self.max_text_length])

        text = self.decode(preds_idx, preds_prob)

        if label is None:
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            return text
        label = self.decode(label)
        return text, label

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SARLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SARLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

        self.rm_symbol = kwargs.get('rm_symbol', False)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()

        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            if self.rm_symbol:
                comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
                text = text.lower()
                text = comp.sub('', text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        return [self.padding_idx]


class SATRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SATRNLabelDecode, self).__init__(character_dict_path,
                                               use_space_char)

        self.rm_symbol = kwargs.get('rm_symbol', False)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()

        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            if self.rm_symbol:
                comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
                text = text.lower()
                text = comp.sub('', text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        return [self.padding_idx]


class DistillationSARLabelDecode(SARLabelDecode):
    """
    Convert 
    Convert between text-label and text-index
    """

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 model_name=["student"],
                 key=None,
                 multi_head=False,
                 **kwargs):
        super(DistillationSARLabelDecode, self).__init__(character_dict_path,
                                                         use_space_char)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name

        self.key = key
        self.multi_head = multi_head

    def __call__(self, preds, label=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            if self.multi_head and isinstance(pred, dict):
                pred = pred['sar']
            output[name] = super().__call__(pred, label=label, *args, **kwargs)
        return output


class PRENLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(PRENLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        padding_str = '<PAD>'  # 0 
        end_str = '<EOS>'  # 1
        unknown_str = '<UNK>'  # 2

        dict_character = [padding_str, end_str, unknown_str] + dict_character
        self.padding_idx = 0
        self.end_idx = 1
        self.unknown_idx = 2

        return dict_character

    def decode(self, text_index, text_prob=None):
        """ convert text-index into text-label. """
        result_list = []
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] == self.end_idx:
                    break
                if text_index[batch_idx][idx] in \
                    [self.padding_idx, self.unknown_idx]:
                    continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = ''.join(char_list)
            if len(text) > 0:
                result_list.append((text, np.mean(conf_list).tolist()))
            else:
                # here confidence of empty recog result is 1
                result_list.append(('', 1))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class NRTRLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=True, use_unkown=False, **kwargs):
        super(NRTRLabelDecode, self).__init__(character_dict_path,
                                              use_space_char, use_unkown = use_unkown)

    def __call__(self, preds, label=None, *args, **kwargs):

        if len(preds) == 2:
            preds_id = preds[0]
            preds_prob = preds[1]
            if isinstance(preds_id, paddle.Tensor):
                preds_id = preds_id.numpy()
            if isinstance(preds_prob, paddle.Tensor):
                preds_prob = preds_prob.numpy()
            if preds_id[0][0] == 2:
                preds_idx = preds_id[:, 1:]
                preds_prob = preds_prob[:, 1:]
            else:
                preds_idx = preds_id
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            if label is None:
                return text
            label = self.decode(label[:, 1:])
        else:
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
            if label is None:
                return text
            label = self.decode(label[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank', '<unk>', '<s>', '</s>'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = self.character[int(text_index[batch_idx][idx])]
                except:
                    continue
                if char_idx == '</s>':  # end
                    break
                char_list.append(char_idx)
                if text_prob is not None:
                    conf_list.append(float(text_prob[batch_idx][idx]))
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, conf_list))
        return result_list


class ViTSTRLabelDecode(NRTRLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(ViTSTRLabelDecode, self).__init__(character_dict_path,
                                                use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds[:, 1:].numpy()
        else:
            preds = preds[:, 1:]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label[:, 1:])
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['<s>', '</s>'] + dict_character
        return dict_character


class ABINetLabelDecode(NRTRLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False, use_unkown=False,
                 **kwargs):
        super(ABINetLabelDecode, self).__init__(character_dict_path,
                                                use_space_char, use_unkown=use_unkown)

    def __call__(self, preds, label=None, *args, **kwargs):
        # 다양한 형태에 유연하게 대처
        if isinstance(preds, dict):
            preds = preds['align'][-1].numpy()
        elif isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        else:
            preds = preds

        

        
        preds = softmax(preds, axis=2)
        
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['</s>'] + dict_character
        if self.use_unkown:
            dict_character = dict_character+[self.unkown_str]
        return dict_character

class ABINetLabelDecode_GraphemeLabel(object):
    pass

class ABINetLabelDecode_GraphemeLabel_All(object):
    """ Convert between text-label and text-index """
    from ppocr.utils.korean_grapheme_label import compose_korean_char
    
    def __init__(self, character_dict_path=None, use_space_char=False, use_unkown=False, handling_grapheme = None,
                 **kwargs):
        
        if handling_grapheme == None:
            raise ValueError("handling_grpaheme must not be None")
        
        self.decode_dict = dict()
        if "character" in handling_grapheme:
            self.decode_dict["character"] = ABINetLabelDecode(character_dict_path=character_dict_path["character"], use_space_char=use_space_char, use_unkown=use_unkown)
        if "initial" in handling_grapheme:
            self.decode_dict["initial"] = ABINetLabelDecode(character_dict_path=character_dict_path["initial"], use_space_char=use_space_char, use_unkown=use_unkown)
        if "medial" in handling_grapheme:
            self.decode_dict["medial"] = ABINetLabelDecode(character_dict_path=character_dict_path["medial"], use_space_char=use_space_char, use_unkown=True)
        if "final" in handling_grapheme:
            self.decode_dict["final"] = ABINetLabelDecode(character_dict_path=character_dict_path["final"], use_space_char=use_space_char, use_unkown=True)
        
        if "utf8string" in handling_grapheme:
            self.decode_dict["utf8string"] = ABINetLabelDecode(character_dict_path=character_dict_path["utf8string"], use_space_char=use_space_char, use_unkown=True)
            
        self.char_num = {name: decode.char_num for name, decode in self.decode_dict.items()}
        
        self.character = {name: decode.character for name, decode in self.decode_dict.items()}
        

        self.handling_grapheme = handling_grapheme
        
        self.c_th_list = [0.01, 0.03, 0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        self.g_th_list = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
        
        # self.
        return
    
    
    def split_grapheme_logits(self, x):
        
        index_ranges = {}
        start_index = 0
        for key, length in self.char_num.items():
            index_ranges[key] = (start_index, start_index + length)
            start_index += length
    
        dict_out = {key : x[:, :, start:end] for key, (start, end) in index_ranges.items()}
        return dict_out
    
    def compose(self, decode_dict, label):
        
        initials = decode_dict["initial"]
        medials = decode_dict["medial"]
        finals = decode_dict["final"] 
        
        
        labels = []
        gt_lables = []
        if label is None: # no label
            for initial, medial, final in zip(initials, medials, finals):
                label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(initial[0], medial[0], final[0], initial[1], medial[1], final[1])
                
                labels.append((label, p))
            return labels
              
        else: # with label
            for initial, gt_initial, medial, gt_medial, final, gt_final in zip(*initials, *medials, *finals):
                gt_label, gt_p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(gt_initial[0], gt_medial[0], gt_final[0], gt_initial[1], gt_medial[1], gt_final[1])
                gt_lables.append((gt_label, 1))
                
                label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(initial[0], medial[0], final[0], initial[1], medial[1], final[1])                    
                labels.append((label, p))
        
        return labels, gt_lables
    
    def __call__(self, preds, label=None, *args, **kwargs):
        # 기존 클래스(ABINetLabelDecode)에 grapheme 단위로 적용할 수 있도록 형태 변환
        """_summary_

        Args:
            preds (dict): {
                vision: Tensor(shape[3, 26, 2041]),
                lang1: Tensor(shape[3, 26, 2041])
                align1: Tensor(shape[3, 26, 2041])
            }
            label (_type_, optional): _description_. Defaults to None. label이 주어지면 함게 디코드, 없으면 pred만 디코딩

        Returns:
            _type_: _description_
        """
        
        result_dict = dict()
        
        for model_name, pred in preds.items():
            if model_name != "align3":
                continue
            # logit_dict = self.split_grapheme_logits(pred)
            logit_dict = pred
            pred_dict = {key: {"align":[logit_dict[key]]} for key in self.decode_dict.keys()} 
            label_dict = {key: label[key].numpy() if label else None for key in self.decode_dict.keys()}
            
            
            # 각 grpaheme 별로 기존 클래스 적용
            decode_dict = {key: self.decode_dict[key](pred_dict[key], label=label_dict[key], *args, **kwargs) for key in self.decode_dict.keys()}
            if all([name in self.decode_dict.keys() for name in ["initial", "medial", "final"]]):
                decode_dict["composed"] = self.compose(decode_dict, label)
        

        
            ### 이 부분만 추가함 !!!!!!!!!!!!!!!!!!!!!!!!!!!
            if "utf8string" in self.handling_grapheme:
                decode_dict["utf8composed"] = self.composed_utf8_string(decode_dict, label)
                
                        ### Ensemble
            if "utf8composed" in decode_dict and "character" in decode_dict:
                
                char_pred = decode_dict["character"][0]
                utf8_pred = decode_dict["utf8composed"][0]
                gt = decode_dict["character"][1]
                
                #########
                ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
                decode_dict["ensemble(c+g_utf8)_by_char"] = [ensemble, gt]
                #########
                ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, utf8_pred)]
                decode_dict["ensemble(c+g_utf8)_by_word"] = [ensemble, gt]
                

                for threshold in self.c_th_list:
                    ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                    decode_dict[f"ensemble(c+g_utf8)_by_char_on_char_({threshold})"] = [ensemble, gt]
                    
                    ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="left") for c, utf8 in zip(char_pred, utf8_pred)]
                    decode_dict[f"ensemble(c+g_utf8)_by_word_on_char_({threshold})"] = [ensemble, gt]
                    
                
                
                
                for threshold in self.g_th_list:
                    ensemble = [char_level_ensemble_by_threshold(c, utf8, threshold=threshold, on="right" ) for c, utf8 in zip(char_pred, utf8_pred)]
                    decode_dict[f"ensemble(c+g_utf8)_by_char_on_utf_({threshold})"] = [ensemble, gt]
                    
                    ensemble = [word_level_ensemble_by_threshold(c, utf8, threshold = threshold, on="right") for c, utf8 in zip(char_pred, utf8_pred)]
                    decode_dict[f"ensemble(c+g_utf8)_by_word_on_utf_({threshold})"] = [ensemble, gt]
                
                
                

                    
                    
                        
            if "composed" in decode_dict and "character" in decode_dict:
                raise NotImplementedError
                char_pred = decode_dict["character"][0]
                composed_pred = decode_dict["composed"][0]
                gt = decode_dict["character"][1]
                #########
                ensemble = [char_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
                decode_dict["ensemble(c+g)_by_char"] = [ensemble, gt]
                #########
                ensemble = [word_level_ensemble(c, utf8) for c, utf8 in zip(char_pred, composed_pred)]
                decode_dict["ensemble(c+g)_by_word"] = [ensemble, gt]
                
   
        
            result_dict[model_name] = decode_dict
            # print(decode_dict)
            # exit()
        
        return result_dict

    def composed_utf8_string(self, preds, label=None):
        utf8string = preds["utf8string"]
        
        
        if label is None: # no label

            composed = [compose_string_by_utf8(pred, pred_p) for pred, pred_p in utf8string]
            return composed
            # raise NotImplementedError
            # composed = list()
            # for pred, prob in utf8string:
              
            #     print(i, m, f, ip, mp, fp)
            #     label, p = ABINetLabelDecode_GraphemeLabel_All.compose_korean_char(i, m, f, ip, mp, fp)
            #     composed.append((label, p))
            # return composed
                
        else: # with label
            composed = [compose_string_by_utf8(pred, pred_p) for pred, pred_p in utf8string[0]]
            gt_composed = [compose_string_by_utf8(gt, gt_p) for gt, gt_p in utf8string[1]]

            return composed, gt_composed
        
        
        
class SPINLabelDecode(AttnLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(SPINLabelDecode, self).__init__(character_dict_path,
                                              use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + [self.end_str] + dict_character
        return dict_character


class VLLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(VLLabelDecode, self).__init__(character_dict_path, use_space_char)
        self.max_text_length = kwargs.get('max_text_length', 25)
        self.nclass = len(self.character) + 1

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id - 1]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def __call__(self, preds, label=None, length=None, *args, **kwargs):
        if len(preds) == 2:  # eval mode
            text_pre, x = preds
            b = text_pre.shape[1]
            lenText = self.max_text_length
            nsteps = self.max_text_length

            if not isinstance(text_pre, paddle.Tensor):
                text_pre = paddle.to_tensor(text_pre, dtype='float32')

            out_res = paddle.zeros(
                shape=[lenText, b, self.nclass], dtype=x.dtype)
            out_length = paddle.zeros(shape=[b], dtype=x.dtype)
            now_step = 0
            for _ in range(nsteps):
                if 0 in out_length and now_step < nsteps:
                    tmp_result = text_pre[now_step, :, :]
                    out_res[now_step] = tmp_result
                    tmp_result = tmp_result.topk(1)[1].squeeze(axis=1)
                    for j in range(b):
                        if out_length[j] == 0 and tmp_result[j] == 0:
                            out_length[j] = now_step + 1
                    now_step += 1
            for j in range(0, b):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            output = paddle.zeros(
                shape=[int(out_length.sum()), self.nclass], dtype=x.dtype)
            for i in range(0, b):
                cur_length = int(out_length[i])
                output[start:start + cur_length] = out_res[0:cur_length, i, :]
                start += cur_length
            net_out = output
            length = out_length

        else:  # train mode
            net_out = preds[0]
            length = length
            net_out = paddle.concat([t[:l] for t, l in zip(net_out, length)])
        text = []
        if not isinstance(net_out, paddle.Tensor):
            net_out = paddle.to_tensor(net_out, dtype='float32')
        net_out = F.softmax(net_out, axis=1)
        for i in range(0, length.shape[0]):
            preds_idx = net_out[int(length[:i].sum()):int(length[:i].sum(
            ) + length[i])].topk(1)[1][:, 0].tolist()
            preds_text = ''.join([
                self.character[idx - 1]
                if idx > 0 and idx <= len(self.character) else ''
                for idx in preds_idx
            ])
            preds_prob = net_out[int(length[:i].sum()):int(length[:i].sum(
            ) + length[i])].topk(1)[0][:, 0]
            preds_prob = paddle.exp(
                paddle.log(preds_prob).sum() / (preds_prob.shape[0] + 1e-6))
            text.append((preds_text, float(preds_prob)))
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class CANLabelDecode(BaseRecLabelDecode):
    """ Convert between latex-symbol and symbol-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CANLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def decode(self, text_index, preds_prob=None):
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            seq_end = text_index[batch_idx].argmin(0)
            idx_list = text_index[batch_idx][:seq_end].tolist()
            symbol_list = [self.character[idx] for idx in idx_list]
            probs = []
            if preds_prob is not None:
                probs = preds_prob[batch_idx][:len(symbol_list)].tolist()

            result_list.append([' '.join(symbol_list), probs])
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        pred_prob, _, _, _ = preds
        preds_idx = pred_prob.argmax(axis=2)

        text = self.decode(preds_idx)
        if label is None:
            return text
        label = self.decode(label)
        return text, label
