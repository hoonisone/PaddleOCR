from outputs import OutputDB

if __name__ == "__main__":
    mdb = OutputDB()
    
    
    # mdb.make("/home/configs/det/det_mv3_db.yml", "output1", "ai_hub_det_08_02_90_random_k_fold_5_1", "MobileNetV3_large_x0_5")/
    mdb.make("/home/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml", "output2", "ai_hub_rec_08_02_90", "korean_PP-OCRv3_rec")
    
    
    
    models = mdb.get_name_list()
    print(models)
    print(models)
    print(models[0])
    print(mdb.get(models[0]))
    
    
    