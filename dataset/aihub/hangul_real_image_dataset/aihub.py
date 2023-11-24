from pathlib import Path
from open_dataset import *
import json
from pathlib import Path
from PIL import Image
import os

with open("./organize.json") as f:
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
        (Path(root)/"030.야외 실제 촬영 한글 이미지").unlink()
        (Path(root)/"야외 실제 촬영 한글 이미지").unlink()

class HangulRealImageDataset(OpenDataset):
    """
        AI HUB에서 제공하는 '야외 실제 촬영 한글 이미지'의 데이터를 관리하는 클래스
        데이터 정리, 로드 등 기능 제공 
        데이터를 직접 다운해야 함
        
    """
    def __init__(self, args):
        self.args = args

        super().__init__(args)
        
    def load_image(self, path):
        return Image.open(path)
    
    def load_label(self, path):
        with open(path) as f:
            return json.load(f)
        
    def get_all_sample_list(self):
        target_root = self.args.root
        img_ext = self.args.img_ext
        label_ext = self.args.label_ext
        
        sample_list = []
        
        for img_path in sorted(Path(target_root).rglob(f"*.{img_ext}")):
            label_path = img_path.parent/f"{img_path.stem}.{label_ext}"
            label_path = Path(str(label_path).replace("VS", "VL"))
            sample_list.append([img_path, label_path])
        
        return sample_list

class HangulRealImageDataset_to_PaddleOCR:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def to_paddle_label(self, label):
        result = []
        for annotation in label["annotations"]:
            x, y, w, h = annotation["bbox"]
            upper_left = [x, y]
            upper_right = [x+w, y]
            bottom_right = [x+w, y+h]
            bottom_left = [x, y+h]
        
            result.append({"transcription":annotation["text"], "points":[upper_left, upper_right, bottom_right, bottom_left]})
        return json.dumps(result)

    def make_paddle_label(self, paddle_orc_dataset_root):  
        label_list = []
        for i in range(len(self.dataset)):
            label = self.dataset.get_label(i)
            relative_path = self.dataset.get_image_path(i).relative_to(paddle_orc_dataset_root)
            label = self.to_paddle_label(label)
            paddle_label = f"{relative_path}\t{label}"
            label_list.append(paddle_label)
        
        return "\n".join(label_list)
