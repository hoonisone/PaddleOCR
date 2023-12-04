import sys
sys.path.append("/home/code")

from dataset.aihub.hangul_real_image_dataset import HangulRealImageDataset
from pathlib import Path        
import pandas as pd
import random
import json
import argparse
from config_loader import YamlConfigLoader


def class_ratio_check(path):
    with open(path) as f:
        data = f.readlines()
    x = [Path(line.rstrip('\n').split('\t')[0]) for line in data]
    dir_list= [v.parent.stem for v in x]
    return pd.Series(dir_list).value_counts()

def to_paddle_y(y):
    result = []
    
    for annotation in y:
        try:
            result.append({"transcription":annotation["label"], "points":annotation["bbox"]})
        except:
            print(annotation)
            continue
    return json.dumps(result)
            
def main(args):
    dataset_dir = args.dataset_dir
    
    n=5
    print(f"(1/{n}) Check data dir")#######################################################
    dataset = HangulRealImageDataset().get_box_detection_dataset()
    
    print(f"(2/{n}) Make label file")#######################################################
    label_list = []
    for i in range(len(dataset)):
        label = dataset.get_y(i)
        relative_path = dataset.get_x_path(i).relative_to(dataset_dir)
        
        
        label = to_paddle_y(label)
        paddle_label = f"{relative_path}\t{label}"
        label_list.append(paddle_label)
    
    print(f"(3/{n}) split label file (train, val, test, infer)")#######################################################
    random.seed(args.random_seed)
    random.shuffle(label_list)
    train_ratio, val_ratio, test_ratio = args.train_val_test_ratio
    
    [train_num, test_num] = [int(len(label_list)*train_ratio), int(len(label_list)*(test_ratio))]
        
    train_label_file = "\n".join(label_list[:train_num])
    val_label_file = "\n".join(label_list[train_num:-test_num])
    test_label_file = "\n".join(label_list[-test_num:])
    infer_list_file = "\n".join([label.split("\t")[0] for label in label_list[-test_num:][:args.infer_num]])
    
    print(f"(4/{n}) Check |train|val|test| ratio per class")#######################################################
    train = class_ratio_check("/home/dataset/train_label.txt")
    val = class_ratio_check("/home/dataset/val_label.txt")
    test = class_ratio_check("/home/dataset/test_label.txt")
    for t, v, te in zip(train, val, test):
        total = (t+v+te)/100
        print(f"{int(t/total):>2}|{int(v/total):>2}|{int(te/total):>2}")
    
    print(f"(5/{n}) Save label files")#######################################################
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(dataset_dir)/"train_label.txt", "w") as f:
        f.write(train_label_file)
    with open(Path(dataset_dir)/"val_label.txt", "w") as f:
        f.write(val_label_file)
    with open(Path(dataset_dir)/"test_label.txt", "w") as f:
        f.write(test_label_file)
    with open(Path(dataset_dir)/"infer_list.txt", "w") as f:
        f.write(infer_list_file)




DEFALT_CONFIG_PATH = "/home/code/dataset/paddleOCR/config.yml"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=DEFALT_CONFIG_PATH)
    return  parser.parse_args()

def load_config(path):
    return YamlConfigLoader.load_config(path)  

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)    
    main(config)
 