 python /home/code/PaddleOCR/tools/eval.py    -c /home/configs/det/det_mv3_db.yml    -o Global.pretrained_model=/home/output/det___MobileNetV3_large_x0_5___ai_hub_det_08_02_90_random_k_fold_5_1___default_config___finetune/trained_model/iter_epoch_20        Global.test_save_path=/home/output/det_k_fold_results.csv        Eval.dataset.data_dir=/home/datasets        Eval.dataset.label_file_list=['/home/labelsets/ai_hub_det_08_02_90/test_label.txt']
