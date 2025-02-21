from functools import reduce
from operator import mul
import math

import numpy as np
from typing import List, Union
from pydantic import validate_call

from rapidfuzz.distance import Levenshtein

class GraphemeComposer:
    INITIALS = ['가', '까', '나', '다', '따', '라', '마', '바', '빠', '사', '싸', '아', '자', '짜', '차', '카', '타', '파', '하']
    MEDIALS = ['아', '애', '야', '얘', '어', '에', '여', '예', '오', '와', '왜', '외', '요', '우', '워', '웨', '위', '유', '으', '의', '이']
    FINALS = ['으', '윽', '윾', '윿', '은', '읁', '읂', '읃', '을', '읅', '읆', '읇', '읈', '읉', '읊', '읋', '음', '읍', '읎', '읏', '읐', '응', '읒', '읓', '읔', '읕', '읖', '읗']
    
    @classmethod
    def decompose_char(cls, char):
        # 한글 유니코드 시작: 44032, 끝: 55199
        if not 44032 <= ord(char) <= 55199:
            return char, char, char
        
        # 한글 자모 시작 유니코드
        initial = 44032  # '가'의 유니코드
        
        # 초성, 중성, 종성 계산
        code = ord(char) - initial
        initial_index = code // 588
        medial_index = (code - initial_index * 588) // 28
        final_index = code % 28
        
        return [cls.INITIALS[initial_index], cls.MEDIALS[medial_index], cls.FINALS[final_index]]
    
    @classmethod
    def decompose_string(cls, text):
        decomposed = [cls.decompose_char(c) for c in text]
        return {
            "initial":"".join([x[0] for x in decomposed]),
            "medial":"".join([x[1] for x in decomposed]),
            "final":"".join([x[2] for x in decomposed])
        }

    @classmethod
    def compose_char(cls, initial, medial, final, initial_p=None, medial_p=None, final_p=None):
        if (final is None) or (medial is None) or (initial is None):
            return [" ", 0]
        
        initial_p = 0 if initial_p is None else initial_p
        medial_p = 0 if medial_p is None else medial_p
        final_p = 0 if final_p is None else final_p
        
        # try:
        initial_index = cls.INITIALS.index(initial) if (initial is not None) and (initial in cls.INITIALS) else None
        medial_index = cls.MEDIALS.index(medial) if (medial is not None) and (medial in cls.MEDIALS) else None
        final_index = cls.FINALS.index(final) if (final is not None) and (final in cls.FINALS) else None
        
        # 자소가 1개 이하인 경우 => 그나마 가장 괜찮을 걸로 대체
        if [initial_index, medial_index, final_index].count(None) >= 2:
            grapheme = [initial, medial, final]
            p = [initial_p, medial_p, final_p]
            idx = np.array(p).argmax()
            return [grapheme[idx], p[idx]]
        
        # 자소가 2개 이상인 경우 => 자소가 아닌 녀석만 기본 자소로 사용
        else:
            initial_index = initial_index if initial_index is not None else 11
            medial_index = medial_index if medial_index is not None else 0
            final_index = final_index if final_index is not None else 0
            
        char_code = 44032 + (initial_index * 21 + medial_index) * 28 + final_index
        return [chr(char_code), sum([initial_p, medial_p, final_p])/3]

    @classmethod
    @validate_call
    def compose_string(cls,
                            initial: Union[str, List[str]],  
                            medial: Union[str, List[str]], 
                            final: Union[str, List[str]], 
                            initial_p: Union[List[float], float, None] = None, 
                            medial_p: Union[List[float], float, None] = None, 
                    
                            final_p: Union[List[float], float, None] = None):
        
        
        def preprocessing_of_prob(text, p):
            """_summary_

            Args:
                p [None, int, list]: character에 대한 확률 값 (문자열 전체애 대한 확률인 경우 int, 각 글자에 대한 확률인 경우 [int, ...])
                l int: 문자열 길이 
            Returns:
                [int, ...]: 각 글자에 대한 확률 값으로 변환
            """
            
            if p is None:
                return [None for _ in range(len(text))] # 각 글자마다 확률 None
            elif isinstance(p, int):
                return [p for _ in range(len(text))] # 각 글자에 동일한 확률값 배정
            else:
                return p
            
        # 확률값을 글자 당 확률 형태로 통일        
        initial_p = preprocessing_of_prob(initial, initial_p)    
        medial_p = preprocessing_of_prob(medial, medial_p)
        final_p = preprocessing_of_prob(final, final_p)
        
        text_list, conf_list = [], []
        for i, m, f, ip, mp, fp in list(zip(initial, medial, final, initial_p, medial_p, final_p)):
            text, conf = cls.compose_char(i, m, f, ip, mp, fp)
            text_list.append(text)
            conf_list.append(conf)
        
        if len(text_list) == 0:
            return " ", 0
        return ["".join(text_list), conf_list]

    @classmethod
    def grapheme_edit_dis(cls, x, y):
        _x, _y = x, y
        if len(x) == 0 or len(y) == 0:
            if len(x) == len(y):
                return 1
            else:
                return 0
            
        x = cls.decompose_korean_char(x)
        y = cls.decompose_korean_char(y)
        # x = "".join(["".join(v) if len(set(v)) > 1 else v[0] for v in x])
        # y = "".join(["".join(v) if len(set(v)) > 1 else v[0] for v in y])
        
        x = "".join(["".join(v) for v in x])
        y = "".join(["".join(v) for v in y])
        # print(_x, _y, 1 - Levenshtein.normalized_distance(x, y))
        return Levenshtein.normalized_distance(x, y)


class UTF8Composer:
    HANGUL_BASE = 0xAC00
    CHO_BASE = 0x1100
    JUNG_BASE = 0x1161
    JONG_BASE = 0x11A7

    # 초성, 중성, 종성 테이블
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSUNG_LIST = ['@', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    @classmethod
    def decompose_hangul_by_utf8(cls, text):
        s = ""
        for char in text:
            if not (cls.HANGUL_BASE <= ord(char) <= cls.HANGUL_BASE + 11171):
                s += char
                continue
            
            # 한글 문자 코드
            char_code = ord(char) - cls.HANGUL_BASE
            jong = char_code % 28
            jung = ((char_code - jong) // 28) % 21
            cho = ((char_code - jong) // 28) // 21
            
            s +=  cls.CHOSUNG_LIST[cho]+cls.JUNGSUNG_LIST[jung]+cls.JONGSUNG_LIST[jong]
        return s



    @classmethod
    def compose_hangul_by_utf8(cls, cho, jung, jong=''):
        # 초성, 중성, 종성을 결합하여 유니코드 한글 문자로 변환
        cho_index = cls.CHOSUNG_LIST.index(cho)
        jung_index = cls.JUNGSUNG_LIST.index(jung)
        jong_index = cls.JONGSUNG_LIST.index(jong) if jong else 0
        
        char_code = cls.HANGUL_BASE + (cho_index * 21 + jung_index) * 28 + jong_index
        return chr(char_code)

# def compose_string_by_utf8_old(decomposed_list):
#     composed = []
#     i = 0
#     while i < len(decomposed_list):
#         # 초성, 중성, 종성이 이어진 한글 문자인지 확인
#         if decomposed_list[i] in CHOSUNG_LIST:
#             if i + 1 < len(decomposed_list) and decomposed_list[i+1] in JUNGSUNG_LIST:
#                 cho = decomposed_list[i]
#                 jung = decomposed_list[i+1]
#                 jong = decomposed_list[i+2] if i + 2 < len(decomposed_list) and decomposed_list[i+2] in JONGSUNG_LIST else ''
#                 composed.append(compose_hangul_by_utf8(cho, jung, jong))
#                 i += 3 if jong else 2  # 종성이 있으면 3칸, 없으면 2칸 이동
#             else:
#                 composed.append(decomposed_list[i])  # 중성이 없으면 그대로 추가
#                 i += 1
#         else:
#             composed.append(decomposed_list[i])  # 한글이 아니면 그대로 추가
#             i += 1
#     return ''.join(composed)
    @classmethod
    def compose_string_by_utf8(cls, decomposed_list, p_list):
        if len(decomposed_list) != len(p_list):
            min_len = min(len(decomposed_list), len(p_list))
            decomposed_list = decomposed_list[:min_len]
            p_list = p_list[:min_len]
            
        """
        decomposed: 자소 시퀀스
        p_list: 각 자소 별 추론 확률(confidence) 시퀀스
        """ 
        composed = [] # 결합된 글자 시퀀스
        p_composed = [] # composed 된 시퀀스의 각 글자에 대한 확률 시퀀스
        i = 0
        while i < len(decomposed_list):
            # 초성, 중성, 종성이 이어진 한글 문자인지 확인
            if decomposed_list[i] in cls.CHOSUNG_LIST:
                if i + 1 < len(decomposed_list) and decomposed_list[i+1] in cls.JUNGSUNG_LIST:
                    cho = decomposed_list[i]
                    jung = decomposed_list[i+1]
                    jong = decomposed_list[i+2] if i + 2 < len(decomposed_list) and decomposed_list[i+2] in cls.JONGSUNG_LIST else ''
                    
                    avg_p = sum(p_list[i:i+3]) / 3 if jong != '' else sum(p_list[i:i+2]) / 2
                    composed.append(cls.compose_hangul_by_utf8(cho, jung, jong))
                    p_composed.append(avg_p)
                    i += 3 if jong else 2  # 종성이 있으면 3칸, 없으면 2칸 이동
                else:
                    composed.append(decomposed_list[i])  # 중성이 없으면 그대로 추가
                    p_composed.append(p_list[i])
                    i += 1
            else:
                composed.append(decomposed_list[i])  # 한글이 아니면 그대로 추가
                p_composed.append(p_list[i])
                i += 1
        return [''.join(composed), p_composed]

class Ensembler:

    @classmethod    
    def char_level_ensemble(cls, pred1, pred2):
        c = []
        p = []
        if len(pred1[0]) == 0:
            return pred1
        elif len(pred2[0]) == 0:
            return pred2

        try:
            for c1, p1, c2, p2 in zip(*pred1, *pred2):
                if p1 >= p2:
                    p.append(p1)
                    c.append(c1)
                else:
                    p.append(p2)
                    c.append(c2)
            return "".join(c), p
        except Exception as e:
            # print("char_level_ensemble error", e, pred1, pred2)
            return pred1
    
    @classmethod
    def char_level_ensemble_by_threshold(cls, pred1, pred2, threshold=0.5, on = "left"):
        c = []
        p = []
        if len(pred1[0]) == 0:
            return pred1
        elif len(pred2[0]) == 0:
            return pred2

        try:
            for c1, p1, c2, p2 in zip(*pred1, *pred2):
                if on == "left":
                    if p1 >= threshold:
                        p.append(p1)
                        c.append(c1)
                    else:
                        p.append(p2)
                        c.append(c2)
                elif on == "right":
                    if p2 >= threshold:
                        p.append(p2)
                        c.append(c2)
                    else:
                        p.append(p1)
                        c.append(c1)
                else:
                    raise ValueError("on should be either 'left' or 'right'")
            return "".join(c), p
        except Exception as e:
            print("char_level_ensemble error", e, pred1, pred2)
            return pred1
    @classmethod
    def word_level_ensemble(cls, pred1, pred2):
        try:
            p1 = cls.mul_prob(pred1[1])
            p2 = cls.mul_prob(pred2[1])
            return cls.conditional_select(pred1, pred2, p1 >= p2)
        except Exception as e:
            print("work_level_ensemble error", e, pred1, pred2)
            return pred1


    @classmethod
    def threshold_based_word_level_ensemble_with_mul_prob(cls, pred1, pred2, threshold=0.5, on = "left"):
            p1 = cls.mul_prob(pred1[1])
            p2 = cls.mul_prob(pred2[1])
            return cls.__word_level_ensemble_by_threshold(pred1, pred2, p1, p2, threshold, on)
            
    @classmethod
    def threshold_based_word_level_ensemble_with_log_avg_prob(cls, pred1, pred2, threshold=0.5, on = "left"):
            p1 = cls.log_avg_prob(pred1[1])
            p2 = cls.log_avg_prob(pred2[1])

            return cls.__word_level_ensemble_by_threshold(pred1, pred2, p1, p2, threshold, on)
    
    @classmethod
    def __word_level_ensemble_by_threshold(cls, pred1, pred2, prob1, prob2, threshold=0.5, on = "left"):
        try:        
            if on == "left" :
                return cls.conditional_select(pred1, pred2, prob1 >= threshold)        
            elif on == "right":
                return cls.conditional_select(pred2, pred1, prob2 >= threshold)
            else:
                raise ValueError("on should be either 'left' or 'right'")

        except Exception as e:
            print("work_level_ensemble error", e, pred1, pred2)
            return pred1

    @classmethod
    def log_avg_prob(cls, prob_list):
        try:
            if len(prob_list) == 0:
                return 0
            return sum([math.log(max(p, 0.000001)) for p in prob_list]) / len(prob_list)
        except Exception as e:
            print("log_avg_prob error", e, prob_list)
            return 0

    @classmethod
    def mul_prob(cls, prob_list):
        if len(prob_list) == 0:
            return 0
        return reduce(mul, prob_list)

    @classmethod
    def conditional_select(cls, a, b, condition):
        # condition is true => select a else select b
        if condition:
            return a
        else:
            return b