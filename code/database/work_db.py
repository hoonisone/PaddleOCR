from pathlib import Path
import yaml
from .labelset_db import LabelsetDB
from .dataset_db import DatasetDB
from .model_db import ModelDB
from .db import DB
import copy
import project

class WorkDB(DB):
    DIR = "works"
    ROOT = f"{project.PROJECT_ROOT}/{DIR}"
    CONFIG_NAME = "work_config.yml"
    
    def __init__(self, root=None):
        super().__init__(self.ROOT, self.CONFIG_NAME)
        


    def make(self, name, labelsets, model, pretrained=True):
        modeldb = ModelDB()
        model_config = modeldb.get_config(model)
        train_config = model_config["train_config"]
        
        config = {
            "train_config":f"{self.ROOT}/{name}/train_config.yml",
            "id":name,
            "labelsets":labelsets,
            "model":model,
            "pretrained":pretrained
        }
        
        with open(str(Path(project.PROJECT_ROOT)/train_config)) as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        
        for labelset in labelsets:
            assert labelset in LabelsetDB().get_all_id(), f"the labelset '{labelset}' does not exist!"
        assert model in ModelDB().get_all_id(), f"model '{model}' does not exist!"
        
        model = str(ModelDB().get_config(model)["pretrained_model_weight"]) if pretrained else ""
        labelsets = [LabelsetDB().get_config(labelset) for labelset in labelsets]
        
        data_dir = str(DatasetDB().root)

        train_config["Global"]["pretrained_model"]=model
        train_config["Global"]["print_batch_step"]=1
        train_config["Global"]["save_epoch_step"]=1
        train_config["Global"]["eval_batch_step"]=[0, 2000]
        train_config["Global"]["save_model_dir"] = f"{self.root}/{name}/trained_model"
        train_config["Global"]["save_inference_dir"] = f"{self.root}/{name}/trained_model"
        train_config["Global"]["save_res_path"] = None
        train_config["Train"]["dataset"]["data_dir"] = data_dir
        train_config["Train"]["dataset"]["label_file_list"] = sum([labelset["label"]["train"] for labelset in labelsets], [])
        train_config["Eval"]["dataset"]["data_dir"] = data_dir
        train_config["Eval"]["dataset"]["label_file_list"] = sum([labelset["label"]["eval"] for labelset in labelsets], [])
        # config["Test"] = copy.deepcopy(config["Eval"])        
        # config["Test"]["dataset"]["data_dir"] = data_dir
        # config["Test"]["dataset"]["label_file_list"] = [str(x) for x in labelset["label"]["test"]]
        

        dir_path = Path(self.root)/name
        dir_path.mkdir(parents = True, exist_ok=True)
        with open(dir_path/self.CONFIG_NAME, "w") as f:
            
            yaml.dump(config, f)
        with open(dir_path/"train_config.yml", "w") as f:
            yaml.dump(train_config, f)
             
    def make_inference_model(self, id, epoch=None):
        #None epoch means best model
        root = str(Path(self.ROOT)/id).replace("\\", "/")
        
        model = f"iter_epoch_{epoch}" if epoch else "best_model/model"
        
        config = f"{root}/trained_model/config.yml"
        checkpoint = f"{root}/trained_model/{model}"
        # pretrained = root+"/pretrained_model/pretrained"
        save_folder = f'epoch_{epoch}' if epoch else "best"
        save_inferencd = f"{root}/inference_model/{save_folder}"
        
        command = f"""python code/PaddleOCR/tools/export_model.py \
        -c {config} \
        -o Global.save_inference_dir={save_inferencd} \
        Global.checkpoints={checkpoint}"""

        # -o Global.pretrained_model={pretrained} \        
        if epoch == None:
            config = self.get_config(id)
            config["inference_model_dir"] = f"./works/{id}/inference_model/best"
            config["inference_model"] = f"./works/{id}/inference_model/best/inference"
            self.update_config(id, config)
        
        path = Path(project.PROJECT_ROOT)/"code/database/make_inference_model.sh"
        with open(path, "a") as f:
            f.write(command+"\n")
        
        print(f"{str(path)}")
            
            
    def train(self, id, epoch):
        code = f"{project.PROJECT_ROOT}/code/PaddleOCR/tools/train.py"
        config = self.get_config(id)
        train_config = config["train_config"]        
        
        command = f"python {code} -c {train_config} -o Global.epoch_num={epoch}"
        
        with open(f"{project.PROJECT_ROOT}/works/train.sh", "a") as f:
            f.write(command+"\n")        
    
if __name__ == "__main__":
    mdb = WorkDB()
    
    
    # mdb.make("/home/configs/det/det_mv3_db.yml", "output1", "ai_hub_det_08_02_90_random_k_fold_5_1", "MobileNetV3_large_x0_5")/
    mdb.make("/home/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml", "output2", "ai_hub_rec_08_02_90", "korean_PP-OCRv3_rec")
    
    
    
    models = mdb.get_name_list()
    print(models)
    print(models)
    print(models[0])
    print(mdb.get(models[0]))
    
    
    