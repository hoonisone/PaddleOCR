from .db import DB
import project
from pathlib import Path
import os
import subprocess

class ModelDB(DB):    
    DIR = "models"
    ROOT = f"{project.PROJECT_ROOT}/{DIR}"
    CONFIG_NAME = "model_config.yml"
    
    def __init__(self):
        super().__init__(self.ROOT, self.CONFIG_NAME)
    
    def make_inference_model(self, id):
        
        root = str(Path(self.ROOT)/id).replace("\\", "/")
        config = root+"/pretrained_model/config.yml"
        checkpoint = root+"/pretrained_model/pretrained"
        pretrained = root+"/pretrained_model/pretrained"
        save_inferencd = root+"/inference_model"
        command = f"""python code/PaddleOCR/tools/export_model.py \
        -c {config} \
        -o Global.pretrained_model={pretrained} \
        Global.save_inference_dir={save_inferencd} \
        Global.checkpoints={checkpoint}"""
        
        config = self.get_config(id)
        config["inference_model_dir"] = f"./models/{id}/inference_model"
        config["inference_model"] = f"./models/{id}/inference_model/inference"
        self.update_config(id, config)
        
        with open(Path(project.PROJECT_ROOT)/"code/database/make_inference_model.sh", "a") as f:
            f.write(command+"\n")
        
    
if __name__ == "__main__":
    mdb = ModelDB()
    print(mdb.get_all_id())
    print()
    id = mdb.get_all_id()[0]
    print(mdb.get_config(id))