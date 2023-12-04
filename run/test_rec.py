import os

config="/home/code/PaddleOCR/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml"
data_dir="/home/rec/dataset"
label_file_list=["/home/rec/dataset/test_label.txt"]


pretrained_model="/home/rec/korean_PP-OCRv3_rec/pretrained/model/best_accuracy"
test_save_path="/home/rec/test"
character_dict_path="/home/code/PaddleOCR/ppocr/utils/dict/korean_dict.txt"

command = f""" python /home/code/PaddleOCR/tools/eval.py \
    -c {config} \
    -o Global.pretrained_model={pretrained_model} \
        Global.test_save_path={test_save_path} \
        Global.character_dict_path={character_dict_path} \
        Eval.dataset.data_dir={data_dir} \
        Eval.dataset.label_file_list={label_file_list}
"""

os.system(command)
