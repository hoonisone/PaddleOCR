model_name="ml_PP-OCRv3_det" # 테스트 할 모델 이름
config="det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml" # 모델에 맞는 config 파일 경로 (/home/PaddleOCR/configs 기준 상대경로)

# model_name="ml_PP-OCRv3_det"
# config="ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml"

python ../PaddleOCR/tools/eval.py \
    -c ../PaddleOCR/configs/${config} \
    -o Eval.dataset.label_file_list=/home/dataset/test_label.txt \
        Global.pretrained_model=/home/resource/${model_name}/pretrained/best_accuracy \
        Global.save_model_dir=/home/resource/${model_name}/pretrained/test