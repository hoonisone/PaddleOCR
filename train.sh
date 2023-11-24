# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml


python PaddleOCR/tools/train.py --log_dir=./debug/ --gpus '0' -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml




python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  PaddleOCR/tools/train.py -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml