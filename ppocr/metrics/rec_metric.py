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

from http.cookiejar import CookiePolicy
from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

import numpy as np
import string
from tomlkit import item

# from PaddleOCR.ppstructure import predict_system

# exit()
# from PaddleOCR.tools import infer_kie_token_ser_re
from ppocr.utils.korean_grapheme_label import compose_korean_char, decompose_korean_char, grapheme_edit_dis

def hirschberg_lcs(X, Y):
    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = curr[:]
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j], prev[j])
        return curr

    def hirschberg(X, Y):
        if len(X) == 0:
            return ""
        elif len(Y) == 0:
            return ""
        elif len(X) == 1 or len(Y) == 1:
            for x in X:
                if x in Y:
                    return x
            return ""
        else:
            i = len(X) // 2
            L1 = lcs_length(X[:i], Y)
            L2 = lcs_length(X[i:][::-1], Y[::-1])
            k = max(range(len(Y) + 1), key=lambda j: L1[j] + L2[len(Y) - j])
            return hirschberg(X[:i], Y[:k]) + hirschberg(X[i:], Y[k:])

    return hirschberg(X, Y)

class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 test_print = False,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

        self.test_print = test_print
        
        self.label_char = dict()
        self.pred_char = dict()
        self.answer_char = dict()
        
        self.is_training = kwargs["is_training"]
        
        self.char_set_path_dict = {
            # "full": "/home/labelsets/aihub_rec_full_horizontal_clean_80:10:10/char_set_full.txt",
            # "few":"/home/labelsets/aihub_rec_full_horizontal_clean_80:10:10/char_set_few.txt",
            # "medium":"/home/labelsets/aihub_rec_full_horizontal_clean_80:10:10/char_set_medium.txt",
            # "many":"/home/labelsets/aihub_rec_full_horizontal_clean_80:10:10/char_set_many.txt",
            
            # "zero":"/home/code/PaddleOCR/ppocr/metrics/remove_30_zero_char_set.txt"
        }

        self.char_set_dict = {
            
        }
        for name, path in self.char_set_path_dict.items():
            with open(path) as f:
               self.char_set_dict[name] = [line.strip() for line in f.readlines()]

        
        # print(self.char_set_dict)
        # exit()
        
    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def eval_core(self, preds, labels):
        
        correct_num = 0
        all_num = 0
        
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")

            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
                    
            if pred == target:
                correct_num += 1

            all_num += 1
            
        self.correct_num += correct_num
        self.all_num += all_num

        report =  {
            'acc': correct_num / (all_num + self.eps),
        }    
        
        return report

    def eval_full(self, preds, labels):
    
        # preds: [(test, acc), ...]
        # labels: [(test, acc), ...]

        label_char = dict()
        pred_char = dict()
        answer_char = dict()

        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        grapheme_norm_edit_dis = 0.0
        # print(preds, labels)
        # exit()

        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")

            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
        
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            grapheme_norm_edit_dis += grapheme_edit_dis(pred, target)
            if pred == target:
                correct_num += 1



            for c in pred:
                if c not in pred_char:
                    pred_char[c] = 0
                    self.pred_char[c] = 0
                pred_char[c] += 1
                self.pred_char[c] += 1
            for c in target:
                if c not in label_char:
                    label_char[c] = 0
                    self.label_char[c] = 0
                label_char[c] += 1
                self.label_char[c] += 1
            for c in hirschberg_lcs(pred, target):
                if c not in answer_char:
                    answer_char[c] = 0
                    self.answer_char[c] = 0       
                answer_char[c] += 1
                self.answer_char[c] += 1
            
            all_num += 1
            
            
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        self.grapheme_norm_edit_dis += grapheme_norm_edit_dis
        report =  {
            'acc': correct_num / (all_num + self.eps),
            'c_ned': 1 - norm_edit_dis / (all_num + self.eps),
            'g_ned': 1 - grapheme_norm_edit_dis / (all_num + self.eps)
        }
        

        for name, char_set in self.char_set_dict.items():
            precisions = []
            recalls = []
            f1_scores = []
            
            total_label = 0
            total_answer = 0
            total_pred = 0
            for char in char_set:
                if char in list(label_char.keys()): # 레이블이 있으면 recall 계산 가능
                    recall = answer_char.get(char, 0) / (label_char.get(char, 0) + self.eps)
                    recalls.append(recall)
                else:
                    recall = 0
                    
                    
                if char in list(pred_char.keys()): # 예측된 정답이 있으면 precision 계산 가능
                    precision = answer_char.get(char, 0) / (pred_char.get(char, 0) + self.eps)
                    precisions.append(precision)
                else:
                    precision = 0    
                
                
                if char in list(label_char.keys()) or char in list(pred_char.keys()): # 정답 또는 레이블 하나라도 있어야 f1-score 게산 가능
                    
                    f1_score = 2 * recall * precision / (recall + precision + self.eps)
                    f1_scores.append(f1_score)
                    
                    
                total_label += label_char.get(char, 0)
                total_answer += answer_char.get(char, 0)
                total_pred += pred_char.get(char, 0)
        
            mean_f1_score = sum(f1_scores) / (len(f1_scores) + self.eps)
            mean_precision = sum(precisions) / (len(precisions) + self.eps)
            mean_recall = sum(recalls) / (len(recalls) + self.eps)
            
            overall_precision = total_answer/(total_pred+self.eps)
            overall_recall = total_answer/(total_label+self.eps)
            overall_f1_score = 2 * overall_recall * overall_precision / (overall_recall + overall_precision + self.eps)
            
            report[f"mean_precision_{name}"] = mean_precision
            report[f"mean_recall_{name}"] = mean_recall
            report[f"mean_f1_score_{name}"] = mean_f1_score
            report[f"overall_precision_{name}"] = overall_precision
            report[f"overall_recall{name}"] = overall_recall
            report[f"overall_f1_score{name}"] = overall_f1_score
            
        return report
    
    def __call__(self, preds, labels, *args, **kwargs):
    
        if self.is_training:
            return self.eval_core(preds, labels)
        else:
            return self.eval_full(preds, labels) 

    def get_metric_core(self): # traning 시점
        acc = 1.0 * self.correct_num / (self.all_num + self.eps) if self.all_num > 0 else 0.0
        report = {'acc': acc}        
        self.reset()
        return report
    
    def get_metric_full(self): #
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        # print(self.correct_num, self.all_num)
        acc = 1.0 * self.correct_num / (self.all_num + self.eps) if self.all_num > 0 else 0.0
        norm_edit_dis = 1.0 - self.norm_edit_dis / (self.all_num + self.eps) if self.all_num > 0 else 0.0
        grapheme_norm_edit_dis = 1.0 - self.grapheme_norm_edit_dis / (self.all_num + self.eps) if self.all_num > 0 else 0.0

        report = {'acc': acc, 'C_NED': norm_edit_dis, "G_NED":grapheme_norm_edit_dis}

        for name, char_set in self.char_set_dict.items():
            precisions = []
            recalls = []
            f1_scores = []
            
            total_label = 0
            total_answer = 0
            total_pred = 0
            for char in char_set:
                if char in list(self.label_char.keys()): # 레이블이 있으면 recall 계산 가능
                    recall = self.answer_char.get(char, 0) / (self.label_char.get(char, 0) + self.eps)
                    recalls.append(recall)
                else:
                    recall = 0
                    
                    
                if char in list(self.pred_char.keys()): # 예측된 정답이 있으면 precision 계산 가능
                    precision = self.answer_char.get(char, 0) / (self.pred_char.get(char, 0) + self.eps)
                    precisions.append(precision)
                else:
                    precision = 0    
                
                
                if char in list(self.label_char.keys()) or char in list(self.pred_char.keys()): # 정답 또는 레이블 하나라도 있어야 f1-score 게산 가능
                    
                    f1_score = 2 * recall * precision / (recall + precision + self.eps)
                    f1_scores.append(f1_score)
                    
                    
                total_label += self.label_char.get(char, 0)
                total_answer += self.answer_char.get(char, 0)
                total_pred += self.pred_char.get(char, 0)
        
            mean_f1_score = sum(f1_scores) / (len(f1_scores) + self.eps)
            mean_precision = sum(precisions) / (len(precisions) + self.eps)
            mean_recall = sum(recalls) / (len(recalls) + self.eps)
            
            overall_precision = total_answer/(total_pred+self.eps)
            overall_recall = total_answer/(total_label+self.eps)
            overall_f1_score = 2 * overall_recall * overall_precision / (overall_recall + overall_precision + self.eps)
            
            report[f"mean_precision_{name}"] = mean_precision
            report[f"mean_recall_{name}"] = mean_recall
            report[f"mean_f1_score_{name}"] = mean_f1_score
            report[f"overall_precision_{name}"] = overall_precision
            report[f"overall_recall{name}"] = overall_recall
            report[f"overall_f1_score{name}"] = overall_f1_score
        
        self.reset()
        return report


    def get_metric(self):
        if self.is_training:
            return self.get_metric_core()
        else:
            return self.get_metric_full()

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.grapheme_norm_edit_dis = 0
        self.label_char = dict()    
        self.pred_char = dict()
        self.answer_char = dict()


# class RecMetric_GraphemeLabel(object):
#     def __init__(self,
#                  main_indicator='acc',
#                  is_filter=False,
#                  ignore_space=True,
#                  handling_grapheme = None,
#                  print_test = False,
#                  **kwargs):
#         self.is_filter = is_filter
#         self.ignore_space = ignore_space
#         self.eps = 1e-5
#         self.reset()    
        
#         self.handling_grapheme = handling_grapheme
#         self._main_indicator = main_indicator
        
#         self.inner_recmetric = {"o" : {"direct":{}, "composed":{}, "utf8composed":{}}, "x" : {"direct":{}, "composed":{}, "utf8composed":{}}}

#         # self.c_th_list = [0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
#         # self.g_th_list = [0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]
        
#         self.c_th_list = kwargs.get("c_th_list", []) 
#         self.g_th_list = kwargs.get("g_th_list", [])
        
#         # self.c_th_list = [0.30, 0.40, 0.60, 0.70, 0.95]
#         # self.g_th_list = [0.30, 0.40, 0.60, 0.70, 0.95]
        
#         if "character" in self.handling_grapheme:
#             for g in ["character", "initial", "medial", "final"]:
#                 self.inner_recmetric["o"]["direct"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
#                 self.inner_recmetric["x"]["direct"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
        
#         pure_grapheme = list(set(["initial", "medial", "final"]) & set(self.handling_grapheme))
#         if 0 < len(pure_grapheme):
#             for g in self.handling_grapheme:
#                 self.inner_recmetric["o"]["composed"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
#                 self.inner_recmetric["x"]["composed"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
        
#             if len(pure_grapheme) == 3:
#                 self.inner_recmetric["o"]["composed"]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
#                 self.inner_recmetric["x"]["composed"]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
        
#         if "utf8string" in self.handling_grapheme:
#             for g in ["character", "utf8string", "initial", "medial", "final"]:
#                 self.inner_recmetric["o"]["utf8composed"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
#                 self.inner_recmetric["x"]["utf8composed"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)

#         if "utf8string" in self.handling_grapheme and "character" in self.handling_grapheme: # Utf8 & Character Ensemble
#             for name in ["ensemble(c+g_utf8)_by_char", "ensemble(c+g_utf8)_by_word"]+[
#                 f"ensemble(c+g_utf8)_by_char_on_char_({threshold})" for threshold in self.c_th_list]+[
#                 f"ensemble(c+g_utf8)_by_char_on_utf_({threshold})" for threshold in self.g_th_list]+[
#                 f"ensemble(c+g_utf8)_by_word_on_char_({threshold})" for threshold in self.c_th_list]+[
#                 f"ensemble(c+g_utf8)_by_word_on_utf_({threshold})" for threshold in self.g_th_list]:
#                 self.inner_recmetric["o"].setdefault(name, {})
#                 self.inner_recmetric["x"].setdefault(name, {})
#                 self.inner_recmetric["o"][name]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
#                 self.inner_recmetric["x"][name]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
            
#         # if len(pure_grapheme) == 3 and "character" in self.handling_grapheme: # Character & Composed Ensemble
#         #     for name in ["ensemble(c+g)_by_char", "ensemble(c+g)_by_word",
#         #         "ensemble(c+g)_by_logit_k_3", "ensemble(c+g)_by_logit_k_5", "ensemble(c+g)_by_logit_k_10", "ensemble(c+g)_by_logit_k_15", "ensemble(c+g)_by_logit_k_20", "ensemble(c+g)_by_logit_full",
#         #         "ensemble2(c+g)_by_logit_k_3", "ensemble2(c+g)_by_logit_k_5", "ensemble2(c+g)_by_logit_k_10", "ensemble2(c+g)_by_logit_k_15", "ensemble2(c+g)_by_logit_k_20", "ensemble2(c+g)_by_logit_full"
#         #         ]:
#         #         self.inner_recmetric["o"].setdefault(name, {})
#         #         self.inner_recmetric["x"].setdefault(name, {})
                
#         #         self.inner_recmetric["o"][name]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
#         #         self.inner_recmetric["x"][name]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
            

#         self.ideal_ensemble_correct_num = 0
#         self.total_num = 0

#     @property
#     def main_indicator(self):
#         return self._main_indicator

#     def __call__(self, pred_label, batch=None, *args, **kwargs):
        
#         # label dict 준비 | 모델, 방식 등에 관계없이 추론 형태 마다 하나씩만 존재
#         label_dict = {}
#         for grapheme, labels in batch["text_label"].items():
#             label_dict[grapheme] = [(label, [1]) for label in labels]


        
#         ## organize pred
#         pred_dict = {"direct":{}, "composed":{}, "utf8composed":{}}
#         if "character" in pred_label.keys():
#             # pred_dict["direct"] = {x:[] for x in ["initial", "medial", "final"]}
#             pred_dict["direct"]["character"] = pred_label["character"][0]
            
#             # for pred_text in pred_label["character"][0]:
#             #     pred_text, probability = pred_text
#             #     decomposed = decompose_korean_char(pred_text)
#             #     pred_dict["direct"]["initial"].append([decomposed["initial"], probability])
#             #     pred_dict["direct"]["medial"].append([decomposed["medial"], probability])
#             #     pred_dict["direct"]["final"].append([decomposed["final"], probability])
    
            
#         for g in ["initial", "medial", "final"]:  # 문자 방식 그래핌 추론
#             if g in pred_label:
#                 pred_dict["composed"]["character" if g == "composed" else g] = pred_label[g][0]
    
                
#         if "composed" in pred_label:
#             pred_dict["composed"]["character"] = pred_label["composed"][0]
    
    

#         if "utf8string" in pred_label:
#             # pred_dict["utf8composed"] = {x:[] for x in ["initial", "medial", "final"]}
#             pred_dict["utf8composed"]["utf8string"] = pred_label["utf8string"][0]
#             pred_dict["utf8composed"]["character"] = pred_label["utf8composed"][0]
            
#             # for pred_text in pred_dict["utf8composed"]["character"]:
#             #     pred_text, probability = pred_text
#             #     decomposed = decompose_korean_char(pred_text)
#             #     # print(decomposed)
#             #     pred_dict["utf8composed"]["initial"].append([decomposed["initial"], probability])
#             #     pred_dict["utf8composed"]["medial"].append([decomposed["medial"], probability])
#             #     pred_dict["utf8composed"]["final"].append([decomposed["final"], probability])
                

#         # for ensemble in ["ensemble(c+g_utf8)_by_char", "ensemble(c+g_utf8)_by_word"]+[
#         #         f"ensemble(c+g_utf8)_by_char_on_char_({threshold})" for threshold in self.c_th_list]+[
#         #         f"ensemble(c+g_utf8)_by_char_on_utf_({threshold})" for threshold in self.g_th_list]+[
#         #         f"ensemble(c+g_utf8)_by_word_on_char_({threshold})" for threshold in self.c_th_list]+[
#         #         f"ensemble(c+g_utf8)_by_word_on_utf_({threshold})" for threshold in self.g_th_list]:
#         #     if ensemble in pred_label:
#         #         # print(ensemble)
#         #         pred_dict.setdefault(ensemble, {})
#         #         pred_dict[ensemble]["character"] = pred_label[ensemble][0]
            
#         # exit()

#         metric_report = {}
#         for pred_type, value in pred_dict.items(): # Direct, Composed
#             for g, pred in value.items(): # Character, Initial, Medial, Final
#                 # print(pred_type, g, len(pred), pred[0][:10])
#                 # if g in self.handling_grapheme:            
#                 label = label_dict[g]
#                 for ignore_f in ["o", "x"]: 
#                     # print(pred_type, ignore_f)
#                     # print(self.inner_recmetric[ignore_f][pred_type].keys())
                    
#                     metric = self.inner_recmetric[ignore_f][pred_type][g]([pred, label])
#                     for metric_type, value in metric.items():
#                         metric_report[f"{self.capital(pred_type)}|{self.capital(g)}|{self.capital(metric_type)}|{self.capital(ignore_f)}"] = value     
                    
#         self.metric = metric_report
        
#         if "character" in pred_label and "utf8string" in pred_label:
            
#             for (c, _), (g, _), (label, _) in zip(pred_dict["direct"]["character"], pred_dict["utf8composed"]["character"], label_dict['character']):
#                 c = c.replace(" ", "")
#                 g = g.replace(" ", "")
#                 label = label.replace(" ", "")
#                 if c == label or g == label:
#                     self.ideal_ensemble_correct_num += 1
            
#             self.total_num += len(label_dict['character'])


#     def capital(self, x):
#         return x[0].upper()+x[1:]
        
#     def get_metric(self):

#         metric_report = {}
        
#         for ignore_f, x in self.inner_recmetric.items():
#             for pred_type, y in x.items():
#                 for g, z in y.items():
#                     metric = z.get_metric()
#                     for metric_type, value in metric.items():
#                         metric_report[f"{self.capital(pred_type)}|{self.capital(g)}|{self.capital(metric_type)}|{self.capital(ignore_f)}"] = value
        
#         if self.total_num > 0:
#             metric_report["Ideal_Ensemble_Acc"] = self.ideal_ensemble_correct_num/self.total_num
#         return metric_report
        
#     def reset(self):
#         self.ideal_ensemble_correct_num = 0
#         self.total_num = 0
#         self.metric = None     

class RecMetric_GraphemeLabel_All(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 handling_grapheme = None,
                 **kwargs):
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        
        self.handling_grapheme = handling_grapheme
        self._main_indicator = main_indicator

        self.inner_metric = {}
        
        self.kwargs = kwargs
        

    @property
    def main_indicator(self):
        return self._main_indicator

    def eval(self, pred_dict, label_dict, metric_keys):
        """
            dict를 재귀적으로 탐색하여 내부에 각 추론 결과에 대해 평가를 수행
        """
        result = dict()
        for k, v in pred_dict.items():
            if isinstance(v, dict):
                result[k] = self.eval(v, label_dict, metric_keys+[k])
            else:
                metric = self.inner_metric
                for key in metric_keys[:-1]:
                    metric = metric.setdefault(key, {})
                metric = metric.setdefault(metric_keys[-1], RecMetric(main_indicator=self._main_indicator, is_filter=self.is_filter, ignore_space=self.ignore_space, handling_grapheme=self.handling_grapheme, **self.kwargs))
                return metric(v, label_dict[k])

    def __call__(self, pred_dict, batch=None):
        """
        pred = Models x Methods x Strings
        labels = Strings (batch로 부터 이렇게 만들어야 함)
        """
        
        label_dict = {}
        for grapheme, labels in batch["text_label"].items():
            label_dict[grapheme] = [(label, [1]) for label in labels]
        
        
        
        result = self.eval(pred_dict, label_dict, [])
                
        return result
        
    
    def _get_metric(self, metric, result_dict, key):
        if isinstance(metric, dict):
            for k, v in metric.items():
                self._get_metric(v, result_dict, f"{key}|{k}")
        else:
            result = metric.get_metric()
            for k, v in result.items():
                result_dict[f"{key}|{k}"] = v
            # result_dict[key] = 
    
    def get_metric(self):
        result_dict = dict()
        self._get_metric(self.inner_metric, result_dict, "")
        return result_dict

        
class CNTMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):

        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        for pred, target in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {'acc': correct_num / (all_num + self.eps), }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0


class CANMetric(object):
    def __init__(self, main_indicator='exp_rate', **kwargs):
        self.main_indicator = main_indicator
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
        self.word_rate = 0
        self.exp_rate = 0
        self.reset()
        self.epoch_reset()

    def __call__(self, preds, batch, **kwargs):
        for k, v in kwargs.items():
            epoch_reset = v
            if epoch_reset:
                self.epoch_reset()
        word_probs = preds
        word_label, word_label_mask = batch
        line_right = 0
        if word_probs is not None:
            word_pred = word_probs.argmax(2)
        word_pred = word_pred.cpu().detach().numpy()
        word_scores = [
            SequenceMatcher(
                None,
                s1[:int(np.sum(s3))],
                s2[:int(np.sum(s3))],
                autojunk=False).ratio() * (
                    len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) /
            len(s1[:int(np.sum(s3))]) / 2
            for s1, s2, s3 in zip(word_label, word_pred, word_label_mask)
        ]
        batch_size = len(word_scores)
        for i in range(batch_size):
            if word_scores[i] == 1:
                line_right += 1
        self.word_rate = np.mean(word_scores)  #float
        self.exp_rate = line_right / batch_size  #float
        exp_length, word_length = word_label.shape[:2]
        self.word_right.append(self.word_rate * word_length)
        self.exp_right.append(self.exp_rate * exp_length)
        self.word_total_length = self.word_total_length + word_length
        self.exp_total_num = self.exp_total_num + exp_length

    def get_metric(self):
        """
        return {
            'word_rate': 0,
            "exp_rate": 0,
        }
        """
        cur_word_rate = sum(self.word_right) / self.word_total_length
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        self.reset()
        return {'word_rate': cur_word_rate, "exp_rate": cur_exp_rate}

    def reset(self):
        self.word_rate = 0
        self.exp_rate = 0

    def epoch_reset(self):
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
