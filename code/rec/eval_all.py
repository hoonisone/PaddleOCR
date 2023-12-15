
from config import *

from pathlib import Path
import pandas as pd
import math

def check_validity(save_path, model_path, label_path):
    if not Path(model_path+".pdparams").exists(): # 모델이 없다면
        return False
    for path in label_path:
        if not Path(path).exists():
            return False

    
    if Path(save_path).exists(): # 결과가 이미 있다면
        df = pd.read_csv(save_path, index_col=0)
        df = df[(df["model_path"]==model_path)&(df["label_path"]==str(label_path))]
        if 0 < len(df):
            return False
    return True 

def _make_eval_command(main_config, model_path, dataset_dir, label_path, save_path):
    if not check_validity(save_path, model_path, label_path):
        print("pass")
        return ""
    return f""" python /home/code/PaddleOCR/tools/eval.py\
    -c {main_config}\
    -o Global.pretrained_model={model_path}\
        Global.test_save_path={save_path}\
        Eval.dataset.data_dir={dataset_dir}\
        Eval.dataset.label_file_list={label_path}\n"""

def make_eval_command(order, save_path, work):
    model = order["pretrained_model"]
    config = order["config"]
    dataset_dir = order[f"{work}_data_dir"]
    label_path = order[f"{work}_label_file_list"]
    
    return _make_eval_command(config, model, dataset_dir, label_path, save_path)
    

if __name__=="__main__":
    split_num = 1
    task = "det"
    model_name = "MobileNetV3_large_x0_5"
    save_path=f"/home/output/det_k_fold_results.csv"
    shell_path = "/home/code/det/eval_det_all.sh"
    mode = "finetune"
    config = None
    
    datasets = [f"ai_hub_det_08_02_90_k_fold_5_{k}" for k in range(1,6)]
    epochs = range(5, 31, 5)
    commands = []
    
    for dataset in datasets:
        for i, epoch in enumerate(epochs):
        # 모든 k_fold에 대해 epoch 번째 가중치 파일이 없으면 종료
            order = make_PP_OCR_config(task, model_name, dataset, mode, f"iter_epoch_{epoch}", config)
            for work in ["train", "val"]:
                commands.append(make_eval_command(order, save_path, work))

    if split_num == 1:
        with open(shell_path, "w") as f:    
            for command in commands:
                f.write(command)

    else:
        seg_len = math.ceil(len(commands)/split_num)
        for i, idx in enumerate(range(0, len(commands), seg_len)):
            
            new_path = shell_path[:-3]+str(i+1)+shell_path[-3:]
            with open(new_path, "w") as f:    
                for command in commands[idx:idx+seg_len]:
                    f.write(command)