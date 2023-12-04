def make_PP_OCR_config(model_name, mode):
    return {
        "model_name":f"/home/rec/{model_name}/{mode}/model/best_accuracy",
        "config":"/home/code/PaddleOCR/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml",
        
        "infer_save_dir":f"/home/rec/{model_name}/{mode}/infer"
    }

korean_PP_OCRv3_rec_pretrained = make_PP_OCR_config("korean_PP-OCRv3_rec", "pretrained")