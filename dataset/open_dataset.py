from abc import *
from PIL import Image
import bbox_visualizer as bbv

class OpenDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.sample_list = self.get_all_sample_list()

    @abstractmethod
    def load_x(self, path):
        pass
    
    @abstractmethod
    def load_y(self, path):
        pass                
    
    def get_x_path(self, index):
        return self.sample_list[index][0]
    
    def get_y_path(self, index):
        return self.sample_list[index][1]
    
    def get_x(self, index):
        return self.load_x(self.get_x_path(index))
    
    def get_y(self, index):
        return self.load_y(self.get_y_path(index))
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        return [self.get_x(index), self.get_y(index)]
    
    def get_sample_path(self, index):
        return self.sample_list[index]
    
    @abstractmethod
    def get_all_sample_list(self):
        pass

    def get_box_detection_dataset():
        return None
    
class OpenDatasetDecorator:
    def __init__(self, dataset):
        self.__dataset = dataset
        
    def __len__(self):
        return len(self.__dataset)

    def convert_x(self, x):
        return x
    
    def convert_y(self, y):
        return y
    
    def get_x(self, index):
        x = self.__dataset.get_x(index)
        return self.convert_x(x)
    
    def get_y(self, index):
        y = self.__dataset.get_y(index)
        return self.convert_y(y)
    
    def __getitem__(self, index):
        return [self.get_x(index), self.get_y(index)]
    
    def get_x_path(self, index):
        return self.__dataset.get_x_path(index)
    
    def get_y_path(self, index):
        return self.__dataset.get_y_path(index)
        
            
class BoxDetectionDataset(OpenDatasetDecorator):
    @abstractmethod
    def convert_x(self, x):
        # numpy image [3, w, h]
        pass
    
    @abstractmethod
    def convert_y(self, x):
        # return: label list
        # label: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  
        # order: (top-left, top-right, bottom-right, bottom-left)
        pass

    def poligon_to_xywh(self, y):
        results = []
        for v in y:
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = v
            results.append([x1, y1, (x2-x1), (y3-y2)])
        return results
    
    def xywh_to_poligon(self, y):
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

        xywh_bboxes = self.poligon_to_xywh(poligon_boxes)
        min_max_xy_bboxes = [[x, y, x+w, y+h]for x, y, w, h in xywh_bboxes]
        x = bbv.draw_multiple_rectangles(x, min_max_xy_bboxes, thickness=2, bbox_color=(0,0,0))
        self.show_x(x)
        

        
    
    