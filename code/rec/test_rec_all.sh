 python /home/code/PaddleOCR/tools/eval.py             -c /home/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml             -o Global.pretrained_model=/home/pretrained_models/korean_PP-OCRv3_rec/pretrained                 Global.test_save_path=/home/output/rec                 Global.character_dict_path=/home/code/PaddleOCR/ppocr/utils/dict/korean_dict.txt                 Eval.dataset.data_dir=/home/datasets                 Eval.dataset.label_file_list=['/home/dataset_labels/rec_08_02_90/test_label.txt']
        