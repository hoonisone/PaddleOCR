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
from ppocr.utils.korean_compose import compose_korean_char, decompose_korean_char
def grapheme_edit_dis(x, y):
    _x, _y = x, y
    if len(x) == 0 or len(y) == 0:
        if len(x) == len(y):
            return 1
        else:
            return 0
        
    x = decompose_korean_char(x)
    y = decompose_korean_char(y)
    
    # x = "".join(["".join(v) if len(set(v)) > 1 else v[0] for v in x])
    # y = "".join(["".join(v) if len(set(v)) > 1 else v[0] for v in y])
    
    x = "".join(["".join(v) for v in x])
    y = "".join(["".join(v) for v in y])
    
    # print(_x, _y, 1 - Levenshtein.normalized_distance(x, y))
    return Levenshtein.normalized_distance(x, y)

class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        
        
        preds, labels = pred_label
        # preds: [(test, acc), ...]
        # labels: [(test, acc), ...]



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
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        self.grapheme_norm_edit_dis = grapheme_norm_edit_dis
        return {
            'acc': correct_num / (all_num + self.eps),
            'C_NED': 1 - norm_edit_dis / (all_num + self.eps),
            'G_NED': 1 - grapheme_norm_edit_dis / (all_num + self.eps)
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        grapheme_norm_edit_dis = 1 - self.grapheme_norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc, 'C_NED': norm_edit_dis, "G_NED":grapheme_norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.grapheme_norm_edit_dis = 0

# class RecMetric_GraphemeLabel_old(object):
#     def __init__(self,
#                  main_indicator='acc',
#                  is_filter=False,
#                  ignore_space=True,
#                  handling_grapheme = None,
#                  **kwargs):
#         self.is_filter = is_filter
#         self.ignore_space = ignore_space
#         self.eps = 1e-5
#         self.reset()    
#         self.inner_recmetric = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=ignore_space, **kwargs)
#         self.handling_grapheme = handling_grapheme

#     @property
#     def main_indicator(self):
#         return self.inner_recmetric.main_indicator

#     def __call__(self, pred_label, batch=None, *args, **kwargs):
#         # print(preds.keys())
#         # print(batch.keys())
        

#         total_metric = dict()
    

    
#         if "character" in pred_label.keys():
#             character_grapheme = {x:[] for x in ["initial", "medial", "final"]}
#             character_grapheme_label = {x:[] for x in ["initial", "medial", "final"]}
            
#             for (pred_text, label) in zip(pred_label["character"][0], pred_label["character"][1]):
#                 pred_text, probability = pred_text
#                 decomposed = decompose_korean_char(pred_text)
#                 first = "".join([x[0] for x in  decomposed])
#                 second = "".join([x[1] for x in  decomposed])
#                 third = "".join([x[2] for x in  decomposed])
#                 character_grapheme["initial"].append([first, probability])
#                 character_grapheme["medial"].append([second, probability])
#                 character_grapheme["final"].append([third, probability])
                
#                 decomposed = decompose_korean_char(label[0])
#                 first = "".join([x[0] for x in  decomposed])
#                 second = "".join([x[1] for x in  decomposed])
#                 third = "".join([x[2] for x in  decomposed])
#                 character_grapheme_label["initial"].append([first, 1])
#                 character_grapheme_label["medial"].append([second, 1])
#                 character_grapheme_label["final"].append([third, 1])
                

#             for g in ["initial", "medial", "final"]:  # 문자 방식 그래핌 추론
#                 # print(pred_label.keys())
#                 # print(pred_label["character"])
#                 # exit()
#                 # if g in self.handling_grapheme:
#                 g_name = g[0].upper()+g[1:]
#                 self.inner_recmetric.ignore_space = True
#                 metric = self.inner_recmetric([character_grapheme[g], character_grapheme_label[g]])    
#                 total_metric[f"C|{g_name}|Acc|X"] = metric["acc"]
#                 total_metric[f"C|{g_name}|C_NED|X"] = metric["C_NED"]
#                 # total_metric[f"C|{g_nameg}|G_NED|X"] = metric["G_NED"]
            
#                 self.inner_recmetric.ignore_space = False
#                 metric = self.inner_recmetric([character_grapheme[g], character_grapheme_label[g]])    
#                 total_metric[f"C|{g_name}|Acc|O"] = metric["acc"]
#                 total_metric[f"C|{g_name}|C_NED|O"] = metric["C_NED"]
#                 # total_metric[f"C|{g_name}|G_NED|O"] = metric["G_NED"]                    



#             g = "character"
#             g_name = g[0].upper()+g[1:]
#             self.inner_recmetric.ignore_space = True
#             metric = self.inner_recmetric([pred_label[g][0], pred_label[g][1]])    
#             total_metric[f"C|{g_name}|Acc|X"] = metric["acc"]
#             total_metric[f"C|{g_name}|C_NED|X"] = metric["C_NED"]
#             total_metric[f"C|{g_name}|G_NED|X"] = metric["G_NED"]
        
#             self.inner_recmetric.ignore_space = False
#             metric = self.inner_recmetric([pred_label[g][0], pred_label[g][1]])    
#             total_metric[f"C|{g_name}|Acc|O"] = metric["acc"]
#             total_metric[f"C|{g_name}|C_NED|O"] = metric["C_NED"]
#             total_metric[f"C|{g_name}|G_NED|O"] = metric["G_NED"]
            
#         ########################################################################################################################################################################    
#         for g in ["initial", "medial", "final"]:  # 그래핌 방식 그래핌 추론
#             if g in pred_label.keys():
#                 g_name = g[0].upper()+g[1:]
#                 self.inner_recmetric.ignore_space = True
#                 metric = self.inner_recmetric([pred_label[g][0], pred_label[g][1]])    
#                 total_metric[f"G|{g_name}|Acc|X"] = metric["acc"]
#                 total_metric[f"G|{g_name}|C_NED|X"] = metric["C_NED"]
#                 # total_metric[f"G|{g_name}|G_NED|X"] = metric["G_NED"]
            
#                 self.inner_recmetric.ignore_space = False
#                 metric = self.inner_recmetric([pred_label[g][0], pred_label[g][1]])    
#                 total_metric[f"G|{g_name}|Acc|O"] = metric["acc"]
#                 total_metric[f"G|{g_name}|C_NED|O"] = metric["C_NED"]
#                 # total_metric[f"G|{g_name}|G_NED|O"] = metric["G_NED"]
                

        
#         pure_grapheme = list(set(["initial", "medial", "final"]) & set(pred_label.keys()))
#         if len(pure_grapheme) == 3: # 그래핌 방식 문자 추론
#             g = "composed"
            
#             self.inner_recmetric.ignore_space = True
#             metric = self.inner_recmetric([pred_label[g][0], pred_label[g][1]])
#             total_metric["G|Character|Acc|X"] = metric["acc"]
#             total_metric["G|Character|C_NED|X"] = metric["C_NED"]
#             total_metric["G|Character|G_NED|X"] = metric["G_NED"]
            
#             self.inner_recmetric.ignore_space = False
#             metric = self.inner_recmetric([pred_label[g][0], pred_label[g][1]])
#             total_metric["G|Character|Acc|O"] = metric["acc"]
#             total_metric["G|Character|C_NED|O"] = metric["C_NED"]
#             total_metric["G|Character|G_NED|O"] = metric["G_NED"]
            
#             # for (f, fp), (s, sp), (th, thp) in zip(preds["first"], preds["second"], preds["third"]):
#             #     pass
            
#         # if len(pure_grapheme) > 0:
#         #     total_metric["grapheme_acc"] = sum([total_metric[f"{g}_acc"] for g in pure_grapheme])/len(pure_grapheme)
#         #     total_metric["grapheme_C_NED"] = sum([total_metric[f"{g}_C_NED"] for g in pure_grapheme])/len(pure_grapheme)
#         #     total_metric["grapheme_G_NED"] = sum([total_metric[f"{g}_G_NED"] for g in pure_grapheme])/len(pure_grapheme)

#         # if len(self.handling_grapheme) == 4: # ensemble
            
#         #     # work level ensenble
#         #     ensemble_preds = []
#         #     for (f, fp), (s, sp), (th, thp), (c, cp) in zip(preds["first"], preds["second"], preds["third"], preds["character"]):
#         #         composed_pred, composed_conf = compose_korean_char(f, s, th, fp, sp, thp, self.first_main)
#         #         gp = [np.mean(np.array(x)) for x in [fp, sp, thp]]
#         #         gp = np.mean(np.array(gp))
#         #         cp = np.mean(np.array(cp))

#         #         if gp <= cp:
#         #             ensemble_preds.append((c, cp))
#         #         else:
#         #             ensemble_preds.append((composed_pred, gp))
            
#         #     self.inner_recmetric.ignore_space = True
#         #     metric = self.inner_recmetric([ensemble_preds, origin_label])
#         #     total_metric["E|Character|Acc|X"] = metric["acc"]
#         #     total_metric["E|Character|C_NED|X"] = metric["C_NED"]
#         #     total_metric["E|Character|G_NED|X"] = metric["G_NED"]
            
#         #     self.inner_recmetric.ignore_space = False
#         #     metric = self.inner_recmetric([ensemble_preds, origin_label])
#         #     total_metric["E|Character|Acc|O"] = metric["acc"]
#         #     total_metric["E|Character|C_NED|O"] = metric["C_NED"]
#         #     total_metric["E|Character|G_NED|O"] = metric["G_NED"]

#         self.metric = total_metric
    
#         # for key, v in self.metric.items():
#         #     print(key, v)
#         # exit()
        
#     def get_metric(self):
#         self.inner_recmetric.get_metric()
#         metric = self.metric
#         self.reset()
#         return metric
        
#     def reset(self):
#         self.metric = None

class RecMetric_GraphemeLabel(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 handling_grapheme = None,
                 **kwargs):
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()    
        
        self.handling_grapheme = handling_grapheme
        self._main_indicator = main_indicator
        
        self.inner_recmetric = {"o" : {"direct":{}, "composed":{}}, "x" : {"direct":{}, "composed":{}}}
        
        if "character" in self.handling_grapheme:
            for g in ["character", "initial", "medial", "final"]:
                self.inner_recmetric["o"]["direct"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
                self.inner_recmetric["x"]["direct"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
        
        pure_grapheme = list(set(["initial", "medial", "final"]) & set(self.handling_grapheme))
        if 0 < len(pure_grapheme):
            for g in self.handling_grapheme:
                self.inner_recmetric["o"]["composed"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
                self.inner_recmetric["x"]["composed"][g] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
        
            if len(pure_grapheme) == 3:
                self.inner_recmetric["o"]["composed"]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=True, **kwargs)
                self.inner_recmetric["x"]["composed"]["character"] = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=False, **kwargs)
        

    @property
    def main_indicator(self):
        return self._main_indicator

    def __call__(self, pred_label, batch=None, *args, **kwargs):

        
        ## organize label
        if "character" in pred_label.keys():
            label_dict = {"character":pred_label["character"][1], "initial":[], "medial":[], "final":[]}
            for label in pred_label["character"][1]:
                decomposed = decompose_korean_char(label[0])
                first = "".join([x[0] for x in  decomposed])
                second = "".join([x[1] for x in  decomposed])
                third = "".join([x[2] for x in  decomposed])
                label_dict["initial"].append([first, 1])
                label_dict["medial"].append([second, 1])
                label_dict["final"].append([third, 1])
        else:
            label_dict = {k: v[1] for k, v in pred_label.items()}
            if "composed" in label_dict:
                label_dict["character"] = label_dict["composed"]
                del label_dict["composed"]

        ## organize pred
        pred_dict = {"direct":{}, "composed":{}}
        if "character" in pred_label.keys():
            pred_dict["direct"] = {x:[] for x in ["initial", "medial", "final"]}
            pred_dict["direct"]["character"] = pred_label["character"][0]
            
            for pred_text in pred_label["character"][0]:
                pred_text, probability = pred_text
                decomposed = decompose_korean_char(pred_text)
                first = "".join([x[0] for x in  decomposed])
                second = "".join([x[1] for x in  decomposed])
                third = "".join([x[2] for x in  decomposed])
                pred_dict["direct"]["initial"].append([first, probability])
                pred_dict["direct"]["medial"].append([second, probability])
                pred_dict["direct"]["final"].append([third, probability])
            
        for g in ["initial", "medial", "final"]:  # 문자 방식 그래핌 추론
            if g in pred_label:
                pred_dict["composed"]["character" if g == "composed" else g] = pred_label[g][0]
                
        if "composed" in pred_label:
            pred_dict["composed"]["character"] = pred_label["composed"][0]
    
    
        metric_report = {}
        for pred_type, value in pred_dict.items(): # Direct, Composed
            for g, pred in value.items(): # Character, Initial, Medial, Final
                # print(pred_type, g, len(pred), pred[0][:10])
                # if g in self.handling_grapheme:            
                label = label_dict[g]
                for ignore_f in ["o", "x"]: 
                    metric = self.inner_recmetric[ignore_f][pred_type][g]([pred, label])
                    
                    for metric_type, value in metric.items():
                        metric_report[f"{self.capital(pred_type)}|{self.capital(g)}|{self.capital(metric_type)}|{self.capital(ignore_f)}"] = value     
                    
        self.metric = metric_report

    def capital(self, x):
        return x[0].upper()+x[1:]
        
    def get_metric(self):

        metric_report = {}
        
        for ignore_f, x in self.inner_recmetric.items():
            for pred_type, y in x.items():
                for g, z in y.items():
                    metric = z.get_metric()
                    for metric_type, value in metric.items():
                        metric_report[f"{self.capital(pred_type)}|{self.capital(g)}|{self.capital(metric_type)}|{self.capital(ignore_f)}"] = value
        
        return metric_report
        
    def reset(self):
        self.metric = None     

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
        self.reset()    
        
        self.handling_grapheme = handling_grapheme
        self._main_indicator = main_indicator

        self.inner_metric = {}
        
        

    @property
    def main_indicator(self):
        return self._main_indicator

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        metric_report = {}
        for k, v in pred_label.items():
            if k not in self.inner_metric.keys():
                self.inner_metric[k] = RecMetric_GraphemeLabel(main_indicator=self._main_indicator, is_filter=self.is_filter, ignore_space=self.ignore_space, handling_grapheme=self.handling_grapheme)
            metric_report[self.capital(k)] = self.inner_metric[k](v)

    def capital(self, x):
        return x[0].upper()+x[1:]
        
    def get_metric(self):
        metric_report = dict()
        for vision_or_lang, v in self.inner_metric.items():
            metric = v.get_metric()
            for k, v in metric.items():
                metric_report[f"{self.capital(vision_or_lang)}|{k}"] = v

        return metric_report
        
    def reset(self):
        self.metric = None     
        
class RecMetric_Grapheme(object):
    def __init__(self,
                 handling_grapheme,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 first_main = False,
                 **kwargs):
        self.first_main = first_main
        self.handling_grapheme = handling_grapheme 
        self.inner_recmetric = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=ignore_space, **kwargs)
    
    @property
    def main_indicator(self):
        return self.inner_recmetric.main_indicator

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        preds, labels = pred_label
        

        total_metric = dict()
        
        for g in self.handling_grapheme:
            metric = self.inner_recmetric([preds[g], labels[g]])    
            total_metric[f"{g}_acc"] = metric["acc"]
            total_metric[f"{g}_C_NED"] = metric["C_NED"]
            total_metric[f"{g}_G_NED"] = metric["G_NED"]
        
        pure_grapheme = list(set(["first", "second", "third"]) & set(self.handling_grapheme))
        if len(pure_grapheme) == 3:
            composed_preds = []
            
            for (f, fp), (s, sp), (th, thp) in zip(preds["first"], preds["second"], preds["third"]):
                composed_pred, composed_conf = compose_korean_char(f, s, th, fp, sp, thp, self.first_main)
                # p = (fp+sp+thp)/3
                composed_preds.append((composed_pred, composed_conf))
            origin_label = [(x, 1.0) for x in batch["origin_label"]]
            
            metric = self.inner_recmetric([composed_preds, origin_label])
            total_metric["composed_acc"] = metric["acc"]
            total_metric["composed_C_NED"] = metric["C_NED"]
            total_metric["composed_G_NED"] = metric["G_NED"]
            
            for (f, fp), (s, sp), (th, thp) in zip(preds["first"], preds["second"], preds["third"]):
                pass
            
        if len(pure_grapheme) > 0:
            total_metric["grapheme_acc"] = sum([total_metric[f"{g}_acc"] for g in pure_grapheme])/len(pure_grapheme)
            total_metric["grapheme_C_NED"] = sum([total_metric[f"{g}_C_NED"] for g in pure_grapheme])/len(pure_grapheme)
            total_metric["grapheme_G_NED"] = sum([total_metric[f"{g}_G_NED"] for g in pure_grapheme])/len(pure_grapheme)

        if len(self.handling_grapheme) == 4: # ensemble
            
            # work level ensenble
            ensemble_preds = []
            for (f, fp), (s, sp), (th, thp), (c, cp) in zip(preds["first"], preds["second"], preds["third"], preds["character"]):
                composed_pred, composed_conf = compose_korean_char(f, s, th, fp, sp, thp, self.first_main)
                gp = [np.mean(np.array(x)) for x in [fp, sp, thp]]
                gp = np.mean(np.array(gp))
                cp = np.mean(np.array(cp))

                if gp <= cp:
                    ensemble_preds.append((c, cp))
                else:
                    ensemble_preds.append((composed_pred, gp))
                    
            metric = self.inner_recmetric([ensemble_preds, origin_label])
            total_metric["word_ensemble_acc"] = metric["acc"]
            total_metric["word_ensemble_C_NED"] = metric["C_NED"]
            total_metric["word_ensemble_G_NED"] = metric["G_NED"]
            
            # character level ensenble
            ensemble_preds = []
            
            for (f, fp), (s, sp), (th, thp), (c, cp) in zip(preds["first"], preds["second"], preds["third"], preds["character"]):
                composed_pred, composed_conf = compose_korean_char(f, s, th, fp, sp, thp, self.first_main)
                gp = [np.mean(np.array(x)) for x in [fp, sp, thp]]
                gp = np.mean(np.array(gp))
                cp = np.mean(np.array(cp))

                if gp <= cp:
                    ensemble_preds.append((c, cp))
                else:
                    ensemble_preds.append((composed_pred, gp))
                    
            metric = self.inner_recmetric([ensemble_preds, origin_label])
            total_metric["word_ensemble_acc"] = metric["acc"]
            total_metric["word_ensemble_C_NED"] = metric["C_NED"]
            total_metric["word_ensemble_G_NED"] = metric["G_NED"]

        self.metric = total_metric
        
    def get_metric(self):
        self.inner_recmetric.get_metric()
        metric = self.metric
        self.reset()
        return metric
        
    def reset(self):
        self.metric = None
        
class RecMetric_Grapheme_v2(object):
    def __init__(self,
                 handling_grapheme,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 first_main = False,
                 **kwargs):
        self.first_main = first_main
        self.handling_grapheme = handling_grapheme 
        self.inner_recmetric = RecMetric(main_indicator=main_indicator, is_filter=is_filter, ignore_space=ignore_space, **kwargs)
    
    @property
    def main_indicator(self):
        return self.inner_recmetric.main_indicator

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        preds, labels = pred_label
        # print(preds.keys())
        # print(batch.keys())
        

        total_metric = dict()
        
    
        if "character" in self.handling_grapheme:
            character_grapheme = {x:[] for x in ["first", "second", "third"]}
            for text, probability in preds["character"]:
                decomposed = decompose_korean_char(text)
                first = "".join([x[0] for x in  decomposed])
                second = "".join([x[1] for x in  decomposed])
                third = "".join([x[2] for x in  decomposed])
                character_grapheme["first"].append([first, probability])
                character_grapheme["second"].append([second, probability])
                character_grapheme["third"].append([third, probability])

            for g in ["first", "second", "third"]:  # 문자 방식 그래핌 추론
                # if g in self.handling_grapheme:
                g_name = g[0].upper()+g[1:]
                self.inner_recmetric.ignore_space = True
                metric = self.inner_recmetric([character_grapheme[g], labels[g]])    
                total_metric[f"C|{g_name}|Acc|X"] = metric["acc"]
                total_metric[f"C|{g_name}|C_NED|X"] = metric["C_NED"]
                # total_metric[f"C|{g_nameg}|G_NED|X"] = metric["G_NED"]
            
                self.inner_recmetric.ignore_space = False
                metric = self.inner_recmetric([character_grapheme[g], labels[g]])    
                total_metric[f"C|{g_name}|Acc|O"] = metric["acc"]
                total_metric[f"C|{g_name}|C_NED|O"] = metric["C_NED"]
                # total_metric[f"C|{g_name}|G_NED|O"] = metric["G_NED"]                    



                g = "character"
                g_name = g[0].upper()+g[1:]
                self.inner_recmetric.ignore_space = True
                metric = self.inner_recmetric([preds[g], labels[g]])    
                total_metric[f"C|{g_name}|Acc|X"] = metric["acc"]
                total_metric[f"C|{g_name}|C_NED|X"] = metric["C_NED"]
                total_metric[f"C|{g_name}|G_NED|X"] = metric["G_NED"]
            
                self.inner_recmetric.ignore_space = False
                metric = self.inner_recmetric([preds[g], labels[g]])    
                total_metric[f"C|{g_name}|Acc|O"] = metric["acc"]
                total_metric[f"C|{g_name}|C_NED|O"] = metric["C_NED"]
                total_metric[f"C|{g_name}|G_NED|O"] = metric["G_NED"]
            
            
        for g in ["first", "second", "third"]:  # 그래핌 방식 그래핌 추론
            if g in self.handling_grapheme:
                g_name = g[0].upper()+g[1:]
                self.inner_recmetric.ignore_space = True
                metric = self.inner_recmetric([preds[g], labels[g]])    
                total_metric[f"G|{g_name}|Acc|X"] = metric["acc"]
                total_metric[f"G|{g_name}|C_NED|X"] = metric["C_NED"]
                # total_metric[f"G|{g_name}|G_NED|X"] = metric["G_NED"]
            
                self.inner_recmetric.ignore_space = False
                metric = self.inner_recmetric([preds[g], labels[g]])    
                total_metric[f"G|{g_name}|Acc|O"] = metric["acc"]
                total_metric[f"G|{g_name}|C_NED|O"] = metric["C_NED"]
                # total_metric[f"G|{g_name}|G_NED|O"] = metric["G_NED"]
                

        
        pure_grapheme = list(set(["first", "second", "third"]) & set(self.handling_grapheme))
        if len(pure_grapheme) == 3: # 그래핌 방식 문자 추론
            composed_preds = []
            
            for (f, fp), (s, sp), (th, thp) in zip(preds["first"], preds["second"], preds["third"]):
                composed_pred, composed_conf = compose_korean_char(f, s, th, fp, sp, thp, self.first_main)
                # p = (fp+sp+thp)/3
                composed_preds.append((composed_pred, composed_conf))
            origin_label = [(x, 1.0) for x in batch["origin_label"]]
            
            self.inner_recmetric.ignore_space = True
            metric = self.inner_recmetric([composed_preds, origin_label])
            total_metric["G|Character|Acc|X"] = metric["acc"]
            total_metric["G|Character|C_NED|X"] = metric["C_NED"]
            total_metric["G|Character|G_NED|X"] = metric["G_NED"]
            
            self.inner_recmetric.ignore_space = False
            metric = self.inner_recmetric([composed_preds, origin_label])
            total_metric["G|Character|Acc|O"] = metric["acc"]
            total_metric["G|Character|C_NED|O"] = metric["C_NED"]
            total_metric["G|Character|G_NED|O"] = metric["G_NED"]
            
            for (f, fp), (s, sp), (th, thp) in zip(preds["first"], preds["second"], preds["third"]):
                pass
            
        # if len(pure_grapheme) > 0:
        #     total_metric["grapheme_acc"] = sum([total_metric[f"{g}_acc"] for g in pure_grapheme])/len(pure_grapheme)
        #     total_metric["grapheme_C_NED"] = sum([total_metric[f"{g}_C_NED"] for g in pure_grapheme])/len(pure_grapheme)
        #     total_metric["grapheme_G_NED"] = sum([total_metric[f"{g}_G_NED"] for g in pure_grapheme])/len(pure_grapheme)

        if len(self.handling_grapheme) == 4: # ensemble
            
            # work level ensenble
            ensemble_preds = []
            for (f, fp), (s, sp), (th, thp), (c, cp) in zip(preds["first"], preds["second"], preds["third"], preds["character"]):
                composed_pred, composed_conf = compose_korean_char(f, s, th, fp, sp, thp, self.first_main)
                gp = [np.mean(np.array(x)) for x in [fp, sp, thp]]
                gp = np.mean(np.array(gp))
                cp = np.mean(np.array(cp))

                if gp <= cp:
                    ensemble_preds.append((c, cp))
                else:
                    ensemble_preds.append((composed_pred, gp))
            
            self.inner_recmetric.ignore_space = True
            metric = self.inner_recmetric([ensemble_preds, origin_label])
            total_metric["E|Character|Acc|X"] = metric["acc"]
            total_metric["E|Character|C_NED|X"] = metric["C_NED"]
            total_metric["E|Character|G_NED|X"] = metric["G_NED"]
            
            self.inner_recmetric.ignore_space = False
            metric = self.inner_recmetric([ensemble_preds, origin_label])
            total_metric["E|Character|Acc|O"] = metric["acc"]
            total_metric["E|Character|C_NED|O"] = metric["C_NED"]
            total_metric["E|Character|G_NED|O"] = metric["G_NED"]

        self.metric = total_metric
    
        # for key, v in self.metric.items():
        #     print(key, v)
        # exit()
        
    def get_metric(self):
        self.inner_recmetric.get_metric()
        metric = self.metric
        self.reset()
        return metric
        
    def reset(self):
        self.metric = None
        
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
