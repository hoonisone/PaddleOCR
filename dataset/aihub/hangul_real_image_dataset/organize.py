from config_loader import YamlConfigLoader
from aihub import HangulRealImageDatasetOganizer, HangulRealImageDataset, HangulRealImageDataset_to_PaddleOCR
from pathlib import Path        
import argparse
from pathlib import Path
import pandas as pd
    
DEFALT_CONFIG_PATH = "/home/code/dataset/aihub/hangul_real_image_dataset/config.yml"

def class_ratio_check(path):
    with open(path) as f:
        data = f.readlines()
    x = [Path(line.rstrip('\n').split('\t')[0]) for line in data]
    dir_list= [v.parent.stem for v in x]
    return pd.Series(dir_list).value_counts()

def main(args):
    root = args.root
    
    n = 8
    print(f"(1/{n}) root 내에 모든 zip 파일 압축 해제")#######################################################
    HangulRealImageDatasetOganizer.unzip(root) # root 내에 모든 zip 파일 압축 해제
    
    print(f"(2/{n}) 남아있는 모든 zip 파일 제거")#######################################################
    HangulRealImageDatasetOganizer.clean_zip(root) # 남아있는 모든 zip 파일 제거
    
    print(f"(3/{n}) 새롭게 디렉터리 구조화 (경로 정리)")#######################################################
    HangulRealImageDatasetOganizer.newly_organize(root) # 새롭게 디렉터리 구조화 (경로 정리)
    
    print(f"(4/{n}) 이미지와 레이블이 잘 대응되어 존재하는지 점검")#######################################################
    HangulRealImageDatasetOganizer.check_valid(root) # 이미지와 레이블이 잘 대응되어 존재하는지 점검
    
    print(f"(5/{n}) 불필요한 기존 폴더 제거 (빈 폴더 제거)")#######################################################
    HangulRealImageDatasetOganizer.clean_origin_dir(root) # 불필요한 기존 폴더 제거 (빈 폴더 제거)
    
    print(f"(6/{n}) Make label files")#######################################################
    dataset = HangulRealImageDataset(args)
    converter = HangulRealImageDataset_to_PaddleOCR(dataset)

    dataset_root = args.Task.PaddleOCR.dataset_root
    train_label_file, val_label_file, test_label_file = converter.make_paddle_label(dataset_root)
    
    print(f"(7/{n}) Save label files")#######################################################
    Path(dataset_root).mkdir(parents=True, exist_ok=True)
    with open(Path(dataset_root)/"train_label.txt", "w") as f:
        f.write(train_label_file)
    with open(Path(dataset_root)/"val_label.txt", "w") as f:
        f.write(val_label_file)
    with open(Path(dataset_root)/"test_label.txt", "w") as f:
        f.write(test_label_file)
    
    
    print(f"(8/{n}) Check |train|val|test| ratio per class")#######################################################
    train = class_ratio_check("/home/dataset/train_label.txt")
    val = class_ratio_check("/home/dataset/val_label.txt")
    test = class_ratio_check("/home/dataset/test_label.txt")

    for t, v, te in zip(train, val, test):
        total = (t+v+te)/100
        print(f"{int(t/total):>2}|{int(v/total):>2}|{int(te/total):>2}")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=DEFALT_CONFIG_PATH)
    return  parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    config = YamlConfigLoader.load_config(args.config_path)    
    main(config)