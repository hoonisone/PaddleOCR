from PIL import Image
import numpy as np

def rotate_image(image, angle):
    # 이미지를 원하는 각도로 회전
    rotated_image = image.rotate(angle, expand=True)
    
    # 회전 후 이미지의 크기 계산
    width, height = rotated_image.size
    
    # 새로운 이미지를 생성하여 회전된 이미지를 중앙에 배치
    new_image = Image.new("RGB", (width, height), (255, 255, 255))
    x_offset = (width - image.width) // 2
    y_offset = (height - image.height) // 2
    
    # new_image.paste(rotated_image, (x_offset, y_offset))
    new_image.paste(rotated_image, (0, 0))
    
    return new_image
class RotateVertical(object):
    def __init__(self, **kwargs):
        self.aspect_ratio_thresh = kwargs["aspect_ratio_thresh"]
        self.angle = kwargs["angle"]

    def __call__(self, data):
        img = data['image']
        h, w, c = img.shape
        if h/w > self.aspect_ratio_thresh:
            img = rotate_image(Image.fromarray(img), self.angle)
            img = np.array(img)
            data["image"] = img
            
        return data

# jamo_to_choseong = {
#             'ᄀ': '가', 'ᄁ': '까', 'ᄂ': '나', 'ᄃ': '다', 'ᄄ': '따', 'ᄅ': '라',
#             'ᄆ': '마', 'ᄇ': '바', 'ᄈ': '빠', 'ᄉ': '사', 'ᄊ': '싸', 'ᄋ': '아',
#             'ᄌ': '자', 'ᄍ': '짜', 'ᄎ': '차', 'ᄏ': '카', 'ᄐ': '타', 'ᄑ': '파', 'ᄒ': '하'
#         }

# def extract_first_grapheme(text):
#     result = ''
#     for char in text:
#         # 한글 유니코드 범위인지 확인
#         if 44032 <= ord(char) <= 55203:
#             # 초성 구하기
#             choseong_index = (ord(char) - 44032) // 588
#             # 초성을 문자로 변환하여 결과에 추가
#             choseong = chr(choseong_index + 0x1100)  # 초성 유니코드 시작값은 0x1100입니다.
#             result += jamo_to_choseong[choseong]
#         else: # 한글이 아니면 그대로 결과에 추가
#             result += char
#     return result


# # 예시 사용
# text = "안녕하세요, Hello!"
# result = convert_to_choseong(text)
# print(result)
  
# class GetFirstGrapheme(object):
#     def __init__(self, **kwargs):
#         pass

#     def __call__(self, data):       
#         # print(data["label"])
#         data["label"] = convert_to_choseong(data["label"])
#         # print(data["label"])     
#         return data
    
    
import functools
from ppocr.utils.korean_compose import UTF8Composer, GraphemeComposer


@functools.lru_cache()
def extract_grapheme(text):
    decomposed = GraphemeComposer.decompose_string(text)
    decomposed["character"] = text
    # return {
    #     "character":text,
    #     "initial":"".join([x[0] for x in decomposed_test]),
    #     "medial":"".join([x[1] for x in decomposed_test]),
    #     "final":"".join([x[2] for x in decomposed_test])
    # }
    return decomposed
import copy

class ExtractGrapheme(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):     
        try:  
            origin_data = data
            data["text_label"] = extract_grapheme(data["label"]) # encode 되지 않고 평가 시 사용되는 레이블 (자칭 train_label)
            data["label"] = copy.copy((data["text_label"])) # 추후 encode 되어 loss를 계산할 수 있도록 하는 레이블 (자칭 train_label)
            
            data["text_label"]["utf8string"] = UTF8Composer.decompose_hangul_by_utf8(data["text_label"]["character"])
            data["label"]["utf8string"] = copy.copy(data["text_label"]["utf8string"])
            return data
        except Exception as e:
            print(e)
            return origin_data
        
def test_ExtractGrapheme():
    data = {"label":"안녕하세요"}
    eg = ExtractGrapheme()
    result = eg(data)
    assert result["text_label"] == {'initial': '아나하사아', 'medial': '아여아에요', 'final': '은응으으으', 'character': '안녕하세요'}


# class ExtractGrapheme2(object):
#     pass
    # # ExtractGrapheme 에서 utf 8 extract까지 추가한 버전
    # def __init__(self, **kwargs):
    #     self.inner_extractor = ExtractGrapheme()
    
    # def __call__(self, data):     
    #     try:  
    #         data = self.inner_extractor(data)
    #         data["text_label"]["utf8string"] = copy.copy(data["text_label"]["character"])
    #         data["label"]["utf8string"] = decompose_hangul_by_utf8(data["text_label"]["character"])
    #         return data
    #     except Exception as e:
    #         print(e)
    #         return data