from pathlib import Path
from PIL import Image
import json
import os
import shutil
import tqdm


def main(data_dir, label_file, save_data_dir, save_label_file):
    
    task_num = 3
    print(f"(1/4)기존 데이터 삭제")
    if save_data_dir.exists():
        shutil.rmtree(save_data_dir) # 기존 내용 삭제 (안에 내용이 있어도 삭제)
    save_data_dir.mkdir(parents=True, exist_ok=True)
    if save_label_file.exists():
        save_label_file.unlink()

    print(f"(2/4) '{label_file}' 로드")
    with open(label_file, "r") as f:
        lines = [line.strip("\n").split("\t") for line in f.readlines()]

    print(f"(3/4) '{label_file}' Image cropping & Save")
    img_num = 1
    pass_num = 0
    xxx_num = 0
    for line in tqdm.tqdm(lines[:]):
        img = Image.open(data_dir/line[0])
        labels = json.loads(line[1])
        for label in labels:
            try:
                text = label["transcription"]
                points = label["points"]
                if text == "xxx":
                    xxx_num += 1
                    continue
                else:
                    bbox = points[0] + points[2]
                    cropped_img = img.crop(bbox)
                    cropped_img.save(save_data_dir/f"{img_num}.png")
                    label = f'word_{img_num}.png, "{text}"'
                    with open(save_label_file, "a") as f:
                        f.write(label+"\n")
            except:
                pass_num += 1
                print(f"error and pass ({pass_num}): {text}, {bbox}")
            
            img_num += 1
    print(f"total:{img_num}, xxx: {xxx_num}, pass:{pass_num}")
                
    print(f"(4/4) '{label_file}' 정리")
    # 마지막 개행 하나 지우기
    lines = open(save_label_file, "r").read()
    with open(save_label_file, "w") as f:
        f.write(lines[:-1])


def f(mode):
    return {
        f"data_dir":Path(f"/home/dataset/"),
        f"label_file":Path(f"/home/dataset/{mode}_label.txt"),
        f"save_data_dir":Path(f"/home/dataset/cropped_img/{mode}")  ,  
        f"save_label_file":Path(f"/home/dataset/cropped_img/{mode}_label.txt"),
    }
task_list = [f("train"), f("val"), f("test")]

     
for i, task in enumerate(task_list):
    main(task["data_dir"], task["label_file"], task["save_data_dir"], task["save_label_file"])