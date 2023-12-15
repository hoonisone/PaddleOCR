
from pathlib import Path
from dataset.open_dataset import *
import json
from pathlib import Path
from PIL import Image
import os
import numpy as np
from config_loader import YamlConfigLoader
    
with open(Path(os.path.realpath(__file__)).parent/"preprocess/organize.json") as f:
    ALL_PATH_PAIR = json.load(f)
    

DEFALT_CONFIG_PATH = "/home/code/dataset/aihub/hangul_real_image_dataset/config.yml"

class DirChecklist:
    # @staticmethod
    # def get_checklist(path):
    #     with open(path) as f:
    #         return yaml.load(f, Loader=yaml.FullLoader)    
    
    @staticmethod
    def get_checked_dir_list(checklist):
        return DirChecklist._get_checked_dir_list(checklist, "")  
    
    @staticmethod
    def _get_checked_dir_list(checklist, path):
        # checklist = dict(checklist)
        path = Path(path)
        if isinstance(checklist, dict):
            if checklist["all"] == True:
                return [path]
            elif checklist["all"] == False:
                return []
            else:
                del checklist["all"]
                return sum([DirChecklist._get_checked_dir_list(v, path/k) for k, v in checklist.items()], [])

        if checklist == True:
            return [path]
        else:
            return []
        
class HangulRealImageDataset(OpenDataset):
    """
        AI HUB에서 제공하는 '야외 실제 촬영 한글 이미지'의 데이터를 관리하는 클래스
        데이터 정리, 로드 등 기능 제공 
        데이터를 직접 다운해야 함
        
    """
    def __init__(self, args=None):
        args = args if args is not None else YamlConfigLoader.load_config(DEFALT_CONFIG_PATH)
        self.args = args
        
        checklist = args.checklist

        self.checked_dir_list = self.get_checked_dir_list(checklist)
        
        print("*******Checked Dir List*******")
        for dir in self.checked_dir_list:
            print(dir)
        print("******************************")
        super().__init__(args)

    def get_checked_dir_list(self, checklist_dir_list):
        checked_dir_list = DirChecklist.get_checked_dir_list(checklist_dir_list)
        return [Path(self.args.dataset_dir)/x for x in checked_dir_list]
    
    def load_x(self, path):
        return Image.open(path)
    
    def load_y(self, path):
        with open(path) as f:
            return json.load(f)
        
    def get_all_sample_list(self):
        img_ext = self.args.x_ext
        label_ext = self.args.y_ext
        sample_list = []
                
        for checked_dir in self.checked_dir_list:
            for img_path in sorted(Path(checked_dir).rglob(f"*.{img_ext}")):
                label_path = img_path.parent/f"{img_path.stem}.{label_ext}"
                label_path = Path(str(label_path).replace("source", "label"))
                sample_list.append([img_path, label_path])
        return sample_list

    def get_box_detection_dataset(self):
        return HangulRealImage_BoxDetectionDataset(self)
    
class HangulRealImage_BoxDetectionDataset(BoxDetectionDataset):
    # @override
    def convert_x(self, x):
        return np.array(x)
    
    # @override
    def convert_y(self, y):
        label = y
        result = []
        for annotation in label["annotations"]:
            try:
                
                x, y, w, h = annotation["bbox"]
                upper_left = [x, y]
                upper_right = [x+w, y]
                bottom_right = [x+w, y+h]
                bottom_left = [x, y+h]
            except:
                continue
            result.append({"bbox":[upper_left, upper_right, bottom_right, bottom_left], "label":annotation["text"]})
        return result


def show_data_example_function():
    import sys
    sys.path.append("/home/code")


    DEFALT_CONFIG_PATH = "/home/code/dataset/aihub/hangul_real_image_dataset/config.yml"
    config = YamlConfigLoader.load_config(DEFALT_CONFIG_PATH)  
    dataset = HangulRealImageDataset(config)
    box_det_dataset = dataset.get_box_detection_dataset()

    sample_idx = 10
    x, y = box_det_dataset[sample_idx]
    box_det_dataset.show_xy(x, y) # 레이블링된 이미지 출력
    box_det_dataset.show_y(y) # 레이블 출력

# class AiHubShell:
#     def __init__(self, id=None, pw=None):
#         self._id = id if id else os.environ["AIHUB_ID"]
#         self._pw = pw if pw else os.environ["AIHUB_PW"]
        
#         os.system("curl -o 'aihubshell' https://api.aihub.or.kr/api/aihubshell.do")
#         os.system("chmod +x aihubshell")
#         os.system("cp aihubshell /usr/bin")
        
#     @property
#     def id(self):
#         if self._id == None:
#             self._id = input("Enter aihub id:")
#         return self._id
    
#     @property
#     def pw(self):
#         if self._pw == None:
#             self._pw = input("Enter aihub pw:")
#         return self._pw
    
#     def download(self, dataset_key, save_dir):
#         current_path = os.getcwd()
#         save_dir = Path(save_dir)
        
#         if save_dir.exists():
#             print(f"The save dir {save_dir} exist. The dataset might have been downloaded already. To continue, you can just delete it and do again")
#             # return 
        
#         save_dir.mkdir(parents=True, exist_ok=True)
#         os.chdir(save_dir)
        
#         # 다운로드
#         os.system(f"aihubshell -mode d -aihubid {self.id} -aihubpw '{self.pw}' -datasetkey {dataset_key}")
#         os.chdir(current_path) # 작업 디렉터리 복구


# AIHUB_ID = "hoonisone@gmail.com"
# AIHUB_PW = "is6E24nYiWPscH#"
# DATASET_KEY = 105
# SAVE_DIR = "/home/dataset/AIHUB/korean_real_outdoor_image"
# AiHubShell(AIHUB_ID, AIHUB_PW).download(DATASET_KEY, SAVE_DIR)