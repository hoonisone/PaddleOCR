from rapidfuzz.distance import Levenshtein
import string

from termios import VKILL
import numpy as np
import random
from PIL import Image


def norm_edit_dis(pred, target):
    ignore_space = True
    is_filter = False

    def _normalize_text(text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()
        
    if ignore_space:
        pred = pred.replace(" ", "")
        target = target.replace(" ", "")
    if is_filter:
        pred = _normalize_text(pred)
        target = _normalize_text(target)
    # return
    return Levenshtein.normalized_distance(pred, target)


def get_shape(path):
    image = Image.open(path)
    w, h = image.size
    if h > w*2:
        return "Vertical"
    elif h*2 < w:
        return "Horizontal"
    else:
        return "Square"

import string


alphabet = list(string.ascii_lowercase)+list(string.ascii_uppercase)
def get_language(text):
    for char in text:
        if char in alphabet:
            return "English"
    return "Korean"

def get_count(data_list):

    # 범위를 나누어 개수를 세기 위한 리스트 생성
    count_list = []

    # 0~100 범위를 5개 구간으로 나누고, 각 구간의 개수를 세서 count_list에 추가
    for i in range(0, 101, 5):
        count = sum(1 for num in data_list if i <= num < i + 5)
        count_list.append(count)

    return np.array(count_list)
    