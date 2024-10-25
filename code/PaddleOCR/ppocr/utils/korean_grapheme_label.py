import numpy as np
from typing import List, Optional, Union
from pydantic import BaseModel, validate_call

from rapidfuzz.distance import Levenshtein

initial_list = ['가', '까', '나', '다', '따', '라', '마', '바', '빠', '사', '싸', '아', '자', '짜', '차', '카', '타', '파', '하']
medial_list = ['아', '애', '야', '얘', '어', '에', '여', '예', '오', '와', '왜', '외', '요', '우', '워', '웨', '위', '유', '으', '의', '이']
final_list = ['으', '윽', '윾', '윿', '은', '읁', '읂', '읃', '을', '읅', '읆', '읇', '읈', '읉', '읊', '읋', '음', '읍', '읎', '읏', '읐', '응', '읒', '읓', '읔', '읕', '읖', '읗']

def _decompose_korean_char(char):
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
    
    return [initial_list[initial_index], medial_list[medial_index], final_list[final_index]]

def decompose_korean_char(text):
    decomposed = [_decompose_korean_char(c) for c in text]
    return {
        "initial":"".join([x[0] for x in decomposed]),
        "medial":"".join([x[1] for x in decomposed]),
        "final":"".join([x[2] for x in decomposed])
    }
    
def _compose_korean_char(initial, medial, final, initial_p=None, medial_p=None, final_p=None):
    if (final is None) or (medial is None) or (initial is None):
        return [" ", 0]
    
    initial_p = 0 if initial_p is None else initial_p
    medial_p = 0 if medial_p is None else medial_p
    final_p = 0 if final_p is None else final_p
    
    # try:
    initial_index = initial_list.index(initial) if (initial is not None) and (initial in initial_list) else None
    medial_index = medial_list.index(medial) if (medial is not None) and (medial in medial_list) else None
    final_index = final_list.index(final) if (final is not None) and (final in final_list) else None
    
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

@validate_call
def compose_korean_char(initial: Union[str, List[str]],  
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
        text, conf = _compose_korean_char(i, m, f, ip, mp, fp)
        text_list.append(text)
        conf_list.append(conf)
    
    if len(text_list) == 0:
        return " ", 0
    return ["".join(text_list), conf_list]

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




def test_addition():
    assert _decompose_korean_char("한") == ["하", "아", "은"]
    assert decompose_korean_char("한명훈") == {"initial":"하마하", "medial":"아여우", "final":"은응은"}
    assert decompose_korean_char("a") == {"initial":"a", "medial":"a", "final":"a"}
    
    assert _compose_korean_char("하", "아", "은", 1, 1, 1) == ["한", 1.0]
    assert compose_korean_char(["하"], ["아"], ["은"], [1.0], [1.0], [1.0]) == ["한", 1.0]
    assert compose_korean_char(["하", "마", "하"], ["아", "여", "우"], ["은", "응", "은"], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == ["한명훈", 1.0]
    


if __name__ == "__main__":
    print(grapheme_edit_dis("abc", "abd"))
    
# Character group:      [Character elements, ...]                                       composition arguments of character group
# Element group:        [Initials, Medials, Finals, Initials_p, Medials_p, Finals_p]    composition arguments of element group
# Character elements:   [Initial, Medial, Final, Initial_p, Medial_p, Final_p]                                                         ex) ["하", "아", "은", 1, 1, 1]
# Initials:             [Initial, ...]                                                                                                 ex) ["하", "마", "하"]
# Medials:              [Medial, ...]                                                                                                  ex) ["아", "여", "우"]
# Finals:               [Final, ...]                                                                                                   ex) ["은", "응", "은"]
# Initial:              char                                                            Initial
# Medial:               char                                                            Medial
# Final:                char                                                            Final
# Initial_p:            int                                                             Initial probability
# Medial_p:             int                                                             Medial probability
# Final_p:              int                                                             Final probability
