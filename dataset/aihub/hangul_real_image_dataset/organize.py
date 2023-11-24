from config_loader import YamlConfigLoader
from aihub import HangulRealImageDatasetOganizer, HangulRealImageDataset, HangulRealImageDataset_to_PaddleOCR
from pathlib import Path        
import argparse
    
DEFALT_CONFIG_PATH = "/home/code/dataset/aihub/hangul_real_image_dataset/config.yml"

def main(args):
    root = args.root
    
    # HangulRealImageDatasetOganizer.unzip(root) # root 내에 모든 zip 파일 압축 해제
    # HangulRealImageDatasetOganizer.clean_zip(root) # 남아있는 모든 zip 파일 제거
    # HangulRealImageDatasetOganizer.newly_organize(root) # 새롭게 디렉터리 구조화 (경로 정리)
    # HangulRealImageDatasetOganizer.check_valid(root) # 이미지와 레이블이 잘 대응되어 존재하는지 점검
    # HangulRealImageDatasetOganizer.clean_origin_dir(root) # 불필요한 기존 폴더 제거 (빈 폴더 제거)
     
    dataset = HangulRealImageDataset(args)
    converter = HangulRealImageDataset_to_PaddleOCR(dataset)

    paddle_root = args.Task.PaddleOCR.dataset_root
    label_file = converter.make_paddle_label(paddle_root)
    
    Path(paddle_root).mkdir(parents=True, exist_ok=True)
    with open(Path(paddle_root)/"label.txt", "w") as f:
        f.write(label_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=DEFALT_CONFIG_PATH)
    return  parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    config = YamlConfigLoader.load_config(args.config_path)    
    main(config)