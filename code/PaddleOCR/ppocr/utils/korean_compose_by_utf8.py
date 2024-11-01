HANGUL_BASE = 0xAC00
CHO_BASE = 0x1100
JUNG_BASE = 0x1161
JONG_BASE = 0x11A7

# 초성, 중성, 종성 테이블
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['@', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def decompose_hangul_by_utf8(text):
    s = ""
    for char in text:
        if not (HANGUL_BASE <= ord(char) <= HANGUL_BASE + 11171):
            s += char
            continue
        
        # 한글 문자 코드
        char_code = ord(char) - HANGUL_BASE
        jong = char_code % 28
        jung = ((char_code - jong) // 28) % 21
        cho = ((char_code - jong) // 28) // 21
        
        s +=  CHOSUNG_LIST[cho]+JUNGSUNG_LIST[jung]+JONGSUNG_LIST[jong]
    return s


def compose_hangul_by_utf8(cho, jung, jong=''):
    # 초성, 중성, 종성을 결합하여 유니코드 한글 문자로 변환
    cho_index = CHOSUNG_LIST.index(cho)
    jung_index = JUNGSUNG_LIST.index(jung)
    jong_index = JONGSUNG_LIST.index(jong) if jong else 0
    
    char_code = HANGUL_BASE + (cho_index * 21 + jung_index) * 28 + jong_index
    return chr(char_code)

def compose_string_by_utf8_old(decomposed_list):
    composed = []
    i = 0
    while i < len(decomposed_list):
        # 초성, 중성, 종성이 이어진 한글 문자인지 확인
        if decomposed_list[i] in CHOSUNG_LIST:
            if i + 1 < len(decomposed_list) and decomposed_list[i+1] in JUNGSUNG_LIST:
                cho = decomposed_list[i]
                jung = decomposed_list[i+1]
                jong = decomposed_list[i+2] if i + 2 < len(decomposed_list) and decomposed_list[i+2] in JONGSUNG_LIST else ''
                composed.append(compose_hangul_by_utf8(cho, jung, jong))
                i += 3 if jong else 2  # 종성이 있으면 3칸, 없으면 2칸 이동
            else:
                composed.append(decomposed_list[i])  # 중성이 없으면 그대로 추가
                i += 1
        else:
            composed.append(decomposed_list[i])  # 한글이 아니면 그대로 추가
            i += 1
    return ''.join(composed)

def compose_string_by_utf8(decomposed_list, p_list):
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
        if decomposed_list[i] in CHOSUNG_LIST:
            if i + 1 < len(decomposed_list) and decomposed_list[i+1] in JUNGSUNG_LIST:
                cho = decomposed_list[i]
                jung = decomposed_list[i+1]
                jong = decomposed_list[i+2] if i + 2 < len(decomposed_list) and decomposed_list[i+2] in JONGSUNG_LIST else ''
                
                avg_p = sum(p_list[i:i+3]) / 3 if jong != '' else sum(p_list[i:i+2]) / 2
                composed.append(compose_hangul_by_utf8(cho, jung, jong))
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

def char_level_ensemble(pred1, pred2):
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

def word_level_ensemble(pred1, pred2):
    try:
        p1 = sum(pred1[1])/len(pred1[1])
        p2 = sum(pred2[1])/len(pred2[1])
        if p1 >= p2:
            return pred1
        else:
            return pred2
    except Exception as e:
        # print("work_level_ensemble error", e, pred1, pred2)
        return pred1