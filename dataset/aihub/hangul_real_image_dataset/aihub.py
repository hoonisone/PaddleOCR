from pathlib import Path
from .open_dataset import *
import json
from pathlib import Path
from PIL import Image
import os
import random
import yaml
from .checklist import DirChecklist
import numpy as np


with open(Path(os.path.realpath(__file__)).parent/"organize.json") as f:
    ALL_PATH_PAIR = json.load(f)
    
class HangulRealImageDatasetOganizer:
    """
        AIHUB 야외 실제 촬영 한글 이미지 데이터 셋에 대해
        폴더를 정리해줌
    """
    # def __init__(self, origin_root):
        # self.root = origin_root
        # assert self.check_all_dir_exist(), f"The root '{origin_root}' has to contain all ai-hub data."
        # assert self.check_all_pair_exist(), f"The renewal imfor is not correct"
    
    @staticmethod
    def unzip(root):
        ""
        # extensions = OpenDataset.check_zip_extensions(self.args.Unzip.extensions)
        origin_root = Path(root) # zip 파일들이 있는 데이터 셋의 root
        target_root = Path(root) # zip 파일들의 압축을 풀어 경로를 유지한 체 저장하기 시작할 대상 root
        
        for origin_path in sorted(origin_root.rglob(f"*.zip")):
            
            target_dir = target_root/origin_path.relative_to(origin_root).parent/origin_path.stem    # 파일을 저장할 디렉터리
            
            if target_dir.exists():
                print(f"Note: The target file '{target_dir}' already exist. It's not done to unzip this file. For doint this, delete the target file")
                continue
            
            target_dir.mkdir(parents=True, exist_ok=True)                                               # 유효성 체크
            
            print(f"unzip -o -O cp949 '{str(origin_path)}' -d '{str(target_dir)}'")
            os.system(f"unzip -o -O cp949 '{str(origin_path)}' -d '{str(target_dir)}'")         # 압축 해제 후 저장

    @staticmethod
    def clean_zip(root):
        for target_path in sorted(Path(root).rglob(f"*.zip")):
            target_path.unlink()
            
    @staticmethod
    def get_path_pair_where_origin_exist(root):
        # pair중 origin이 root 내에 존재하는 pair만 추출하여 반환
        existing_pair = []
        for origin, target in ALL_PATH_PAIR:
            origin = Path(root)/origin
            if (origin).exists():
                existing_pair.append((origin, target))
        return existing_pair
    
    @staticmethod
    def newly_organize(root):
        # 기존의 길고 복잡한 경로를 깔끔하게 정리해줌
        # 대응되는 경로를 미리 정해져 있음
        path_pairs = HangulRealImageDatasetOganizer.get_path_pair_where_origin_exist(root) # origin이 존재하는 pair에 대해서만
                
        for origin, target in path_pairs:
            origin = Path(root)/origin
            target = Path(root)/target
            target.parent.mkdir(parents=True, exist_ok=True)
            origin.rename(target)
            print(f"'{origin}' -> '{target}'")
    
    @staticmethod
    def get_existing_target_dir(root):
        # 전체 target_dir 중 실제로 존재하는 것 만 찾아서 반환
        existing_target_dir = []
        for origin, target in ALL_PATH_PAIR:
            target = Path(root)/origin
            if (target).exists():
                existing_target_dir.append(target)
        return existing_target_dir # 실제로 변경 가능한 수정 사항만 담음
    
    @staticmethod
    def check_valid(root):
        # 변경된 디렉터리들이 유효한지 체크
        existing_target_list = HangulRealImageDatasetOganizer.get_existing_target_dir(root)
        for target in existing_target_list:
            if ("source" in target) and (target.replace("source", "label") not in existing_target_list):
                label_dir = target.replace("source", "label") 
                print(f"The image dir '{target}' needs label dir '{label_dir}'")
                assert False, "Not Invalid Dataset"

    @staticmethod
    def clean_origin_dir(root):
        dir_list = ["030.야외 실제 촬영 한글 이미지", "야외 실제 촬영 한글 이미지"]
        
        for dir in dir_list:
            dir = Path(dir)
            if dir.exists():
                dir.unlink()



class HangulRealImageDataset(OpenDataset):
    """
        AI HUB에서 제공하는 '야외 실제 촬영 한글 이미지'의 데이터를 관리하는 클래스
        데이터 정리, 로드 등 기능 제공 
        데이터를 직접 다운해야 함
        
    """
    def __init__(self, args):
        self.args = args
        self.checked_dir_list = self.get_checked_dir_list()
        
        print("*******Checked Dir List*******")
        for dir in self.checked_dir_list:
            print(dir)
        print("******************************")
        super().__init__(args)    

    def get_checked_dir_list(self):
        checked_dir_list = DirChecklist.get_checked_dir_list(self.args.checklist_path)
        return [Path(self.args.root)/x for x in checked_dir_list]
    
    def load_image(self, path):
        return Image.open(path)
    
    def load_label(self, path):
        with open(path) as f:
            return json.load(f)
        
    def get_all_sample_list(self):
        img_ext = self.args.img_ext
        label_ext = self.args.label_ext
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
    def to_box_detextion_x(self, x):
        return np.array(x)
    
    # @override
    def to_box_detextion_y(self, y):
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
    

            
        

class HangulRealImageDataset_to_PaddleOCR:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def to_paddle_label(self, label):
        result = []
        for annotation in label["annotations"]:
            try:
                x, y, w, h = annotation["bbox"]
                upper_left = [x, y]
                upper_right = [x+w, y]
                bottom_right = [x+w, y+h]
                bottom_left = [x, y+h]
        
                result.append({"transcription":annotation["text"], "points":[upper_left, upper_right, bottom_right, bottom_left]})
            except:
                print(annotation)
                continue
        return json.dumps(result)

    def make_paddle_label(self, paddle_orc_dataset_root, seed = 100):  
        label_list = []
        for i in range(len(self.dataset)):
            label = self.dataset.get_label(i)
            relative_path = self.dataset.get_image_path(i).relative_to(paddle_orc_dataset_root)
            if "간판" not in str(relative_path):
                continue
            label = self.to_paddle_label(label)
            paddle_label = f"{relative_path}\t{label}"
            label_list.append(paddle_label)
            
        
        random.seed(seed)
        random.shuffle(label_list)
        train_num = int(len(label_list)*0.8)
        test_num = int(len(label_list)*0.9)
        
        train_label_list = label_list[:train_num]
        val_label_list = label_list[train_num:test_num]
        test_label_list = label_list[test_num:]
        
        return "\n".join(train_label_list), "\n".join(val_label_list), "\n".join(test_label_list)

class AiHubShell:
    def __init__(self, id=None, pw=None):
        self._id = id if id else os.environ["AIHUB_ID"]
        self._pw = pw if pw else os.environ["AIHUB_PW"]
        
        os.system("curl -o 'aihubshell' https://api.aihub.or.kr/api/aihubshell.do")
        os.system("chmod +x aihubshell")
        os.system("cp aihubshell /usr/bin")
        
    @property
    def id(self):
        if self._id == None:
            self._id = input("Enter aihub id:")
        return self._id
    
    @property
    def pw(self):
        if self._pw == None:
            self._pw = input("Enter aihub pw:")
        return self._pw
    
    def download(self, dataset_key, save_dir):
        current_path = os.getcwd()
        save_dir = Path(save_dir)
        
        if save_dir.exists():
            print(f"The save dir {save_dir} exist. The dataset might have been downloaded already. To continue, you can just delete it and do again")
            # return 
        
        save_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(save_dir)
        
        # 다운로드
        os.system(f"aihubshell -mode d -aihubid {self.id} -aihubpw '{self.pw}' -datasetkey {dataset_key}")
        os.chdir(current_path) # 작업 디렉터리 복구


def show_data_example_function():
    from dataset.aihub.hangul_real_image_dataset.config_loader import YamlConfigLoader
    from dataset.aihub.hangul_real_image_dataset.aihub import HangulRealImageDatasetOganizer, HangulRealImageDataset, HangulRealImageDataset_to_PaddleOCR

    DEFALT_CONFIG_PATH = "/home/code/dataset/aihub/hangul_real_image_dataset/config.yml"
    config = YamlConfigLoader.load_config(DEFALT_CONFIG_PATH)  
    dataset = HangulRealImageDataset(config)
    box_det_dataset = dataset.get_box_detection_dataset()

    sample_idx = 10
    x, y = box_det_dataset[sample_idx]
    box_det_dataset.show_xy(x, y) # 레이블링된 이미지 출력
    box_det_dataset.show_y(y) # 레이블 출력

# AIHUB_ID = "hoonisone@gmail.com"
# AIHUB_PW = "is6E24nYiWPscH#"
# DATASET_KEY = 105
# SAVE_DIR = "/home/dataset/AIHUB/korean_real_outdoor_image"
# AiHubShell(AIHUB_ID, AIHUB_PW).download(DATASET_KEY, SAVE_DIR)