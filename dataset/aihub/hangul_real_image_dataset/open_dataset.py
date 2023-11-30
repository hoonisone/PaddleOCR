from abc import *
from PIL import Image
import bbox_visualizer as bbv

class OpenDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.sample_list = self.get_all_sample_list()

    @abstractmethod
    def load_image(self, path):
        pass
    
    @abstractmethod
    def load_label(self, path):
        pass                
    
    def get_image_path(self, index):
        return self.sample_list[index][0]
    
    def get_label_path(self, index):
        return self.sample_list[index][1]
    
    def get_image(self, index):
        return self.load_image(self.get_image_path(index))
    
    def get_label(self, index):
        return self.load_label(self.get_label_path(index))
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        img_path, label_path = self.get_sample_path(index)
        img = self.load_image(img_path)
        label = self.load_label(label_path)
        return img, label
    
    def get_sample_path(self, index):
        return self.sample_list[index]
    
    @abstractmethod
    def get_all_sample_list(self):
        pass

    def get_box_detection_dataset():
        return None
    
    
class BoxDetectionDataset:
    def __init__(self, dataset):
        self.__dataset = dataset
    
    
    @abstractmethod
    def to_box_detextion_x(self, x):
        # numpy image [3, w, h]
        pass
    
    @abstractmethod
    def to_box_detextion_y(self, x):
        # return: label list
        # label: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  
        # order: (top-left, top-right, bottom-right, bottom-left)
        pass

    def to_xywh(self, y):
        results = []
        for v in y:
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = v
            results.append([x1, y1, (x2-x1), (y3-y2)])
        return results
    
    def to_poligon(self, y):
        results = []
        for x, y, w, h in y:
            results.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            
    def show_x(self, x):
        Image.fromarray(x).show()
    
    def show_y(self, y):
        for v in y:
            label = v["label"]
            bbox = v["bbox"]
            print(f"{label:-10s}: {bbox}")
    
    def show_xy(self, x, y):
        labels = [v["label"] for v in y]
        poligon_boxes = [v["bbox"] for v in y]

        xywh_bboxes = self.to_xywh(poligon_boxes)
        min_max_xy_bboxes = [[x, y, x+w, y+h]for x, y, w, h in xywh_bboxes]
        x = bbv.draw_multiple_rectangles(x, min_max_xy_bboxes, thickness=2, bbox_color=(0,0,0))
        self.show_x(x)
        
    def __len__(self):
        return len(self.__dataset)
    
    def __getitem__(self, index):
        x, y = self.__dataset[index]
        x, y = self.to_box_detextion_x(x), self.to_box_detextion_y(y)
        return x, y
        
    
    