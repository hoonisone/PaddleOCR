from abc import *

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