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