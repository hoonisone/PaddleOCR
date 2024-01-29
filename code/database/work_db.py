from pathlib import Path
import yaml
from .labelset_db import LabelsetDB
from .dataset_db import DatasetDB
from .model_db import ModelDB
from .db import DB
import copy
import project
import pandas as pd
class WorkDB(DB):
    DIR = "works"
    CONFIG_NAME = "work_config.yml"
    
    def __init__(self):
        super().__init__(self.DIR, self.CONFIG_NAME)
        


    def make(self, name, labelsets, model):
        assert name not in self.get_all_id(), f"the work '{name}' already exist!"
        for labelset in labelsets:
            assert labelset in LabelsetDB().get_all_id(), f"the labelset '{labelset}' does not exist!"
        assert model in ModelDB().get_all_id(), f"model '{model}' does not exist!"
        
        modeldb = ModelDB()
        model_config = modeldb.get_config(model, relative_to="absolute")
        train_config = model_config["train_config"]
        
        config = {
            "train_config":f"train_config.yml".replace("\\", "/"),
            "labelsets":labelsets,
            "model":model,
            "result_path": "result.csv",
            "trained_model_dir": "trained_model",
            "inference_result_dir": "inference_result"
        }
        
        # model = str(ModelDB().get_config(model)["pretrained_model_weight"]) if pretrained else ""
        # labelsets = [LabelsetDB().get_config(labelset) for labelset in labelsets]
        
        # data_dir = str(DatasetDB().root)

        # train_config["Global"]["pretrained_model"]=model
        # train_config["Global"]["print_batch_step"]=1
        # train_config["Global"]["save_epoch_step"]=1
        # train_config["Global"]["eval_batch_step"]=[0, 2000]
        # train_config["Global"]["save_model_dir"] = trained_model"
        # train_config["Global"]["save_inference_dir"] = f"{self.root}/{name}/trained_model"
        # train_config["Global"]["save_res_path"] = f"{self.root}/{name}/infer"
        # train_config["Train"]["dataset"]["data_dir"] = data_dir
        # train_config["Train"]["dataset"]["label_file_list"] = sum([labelset["label"]["train"] for labelset in labelsets], [])
        # train_config["Eval"]["dataset"]["data_dir"] = data_dir
        # train_config["Eval"]["dataset"]["label_file_list"] = sum([labelset["label"]["eval"] for labelset in labelsets], [])
        # config["Test"] = copy.deepcopy(config["Eval"])        
        # config["Test"]["dataset"]["data_dir"] = data_dir
        # config["Test"]["dataset"]["label_file_list"] = [str(x) for x in labelset["label"]["test"]]
        

        save_dir = Path(self.PROJECT_ROOT)/self.DIR/name
        save_dir.mkdir(parents = True, exist_ok=True)
        
        with open(save_dir/self.CONFIG_NAME, "w") as f:
            yaml.dump(config, f)
        
        # copy the model config to work train config
        with open(str(Path(project.PROJECT_ROOT)/train_config)) as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
            with open(save_dir/"train_config.yml", "w") as f:
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
        
        self.update(id)
        config = self.get_config(id)
        train_config = config["train_config"]        
        model = self.get_model_weight(id, "latest")
        command = f"python {code} -c {train_config} -o Global.epoch_num={epoch} Global.pretrained_model={model}"
        
        with open(f"{project.PROJECT_ROOT}/works/train.sh", "a") as f:
            f.write(command+"\n")        
    
    @staticmethod
    def report_eval(id, epoch, step, acc, loss, precision, recall):
        result_path = WorkDB().get_config(id)["result_path"]
        if Path(result_path).exists():
            df= pd.read_csv(result_path)
        else:
            df = pd.DataFrame({"epoch":[], "step":[], "acc":[], "loss":[], "precision":[], "recall":[]})    
        
        df._append({"epoch":epoch, "step":step, "acc":acc, "loss":loss, "precision":precision, "recall":recall})
        df.to_csv(result_path)
        pd.DataFrame()
    
    # def update(self, id):
    #     config = super().get_config(id)
    #     config["trained_epoch"] = self.get_trained_epoch(id)   
    #     self.update_config(id, config)
        
    def get_trained_epoch(self, id):
        return len(list(Path(self.get_config(id)["trained_model_dir"]).glob("iter_epoch_*.pdparams")))
    
    
    
    # def get_best_model_weight(self, id):
    #     config = self.get_config(id)
    #     if self.get_trained_epoch(id) == 0:
    #         model = config["model"]
    #         pretrained = config["pretrained"]
    #         return str(ModelDB().get_config(model)["pretrained_model_weight"]) if pretrained else ""
    #     else:
    #         return str(Path(self.ROOT)/id/"trained_model"/"best_model/model").replace("\\", "/")          
    
    def get_model_weight(self, id, epoch, relative_to=None):
        trained_epoch = self.get_trained_epoch(id)
        assert (epoch in ["best", "latest", "pretrained"]) or (isinstance(epoch, int) and epoch <= trained_epoch), f"epoch should be 'best', 'latest', 'pretrained', or positive integer less than trained_epoch {trained_epoch} but {epoch} is given"
        config = self.get_config(id)
        model_config = ModelDB().get_config(config["model"], relative_to=relative_to)
        
        if (epoch in [0, "pretrained"]) or (trained_epoch == 0):
            return str(model_config["pretrained_model_weight"])
        elif epoch == "best":
            return self.relative_to(id, Path(config["trained_model_dir"])/"best_model/model.pdparams", relative_to=relative_to)
        elif epoch == "latest":
            return self.relative_to(id, Path(config["trained_model_dir"])/"latest.pdparams", relative_to=relative_to)        
        else:
            return self.relative_to(id, Path(config["trained_model_dir"])/f"iter_epoch_{epoch}.pdparams", relative_to=relative_to)
    
    # def eval(self, id, epoch, dataset="eval"):
    #     code = f"{project.PROJECT_ROOT}/code/PaddleOCR/tools/train.py" 
    #     config = self.get_config(id)
    #     labelset_id = 
    #     labelfiles = [ for id in config["labelsets"]]
        
    #     train_config = config["train_config"]  
    #     model_weight = self.get_model_weight(id, epoch)
    #     command = f"python {code} -c {train_config} -o Global.pretrained_model={model_weight}" # train에 대해서도 할 수 있게 수정해야 함
    #     with open(f"{project.PROJECT_ROOT}/works/train.sh", "a") as f:
    #         f.write(command+"\n")
    
    def infer(self, id, epoch = "best", dataset="test", relative_to="absoulte", save_to="global"):
        
        assert dataset in ["train", "eval", "test"]
        assert save_to in ["global", "local"]
        code = f"{project.PROJECT_ROOT}/code/PaddleOCR/tools/infer_det.py"
        config = self.get_config(id, relative_to=relative_to)
        
        ppocr_config = config["train_config"]  
        model_weight = self.get_model_weight(id, epoch, relative_to=relative_to)
        model_weight = ".".join(model_weight.split(".")[:-1]) # 확장자 제거
        labelset_ids = config["labelsets"]
        data_dir = LabelsetDB().get_config(labelset_ids[0], relative_to=relative_to)["dataset_dir"]
        labelsets = sum([LabelsetDB().get_config(id, relative_to=relative_to)["infer"][dataset] for id in labelset_ids], [])
        save_dir = config["inference_result_dir"]
        command = f"python {code} -c {ppocr_config} -o Global.pretrained_model={model_weight} Global.save_res_path={save_dir} Infer.data_dir={data_dir} Infer.infer_file_list={labelsets}" # train에 대해서도 할 수 있게 수정해야 함
        print(command)
        with open(f"{project.PROJECT_ROOT}/works/infer.sh", "a") as f:
            f.write(command+"\n")
    
    def get_config(self, id, relative_to=None):
        config = super().get_config(id)
        
        if relative_to:
            for k in ["inference_result_dir", "trained_model_dir", "train_config", "result_path"]:
                config[k] = self.relative_to(id, config[k], relative_to=relative_to)    
        return config
        
if __name__ == "__main__":
    mdb = WorkDB()
    
        
    
    # mdb.make("/home/configs/det/det_mv3_db.yml", "output1", "ai_hub_det_08_02_90_random_k_fold_5_1", "MobileNetV3_large_x0_5")/
    # mdb.make("/home/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml", "output2", "ai_hub_rec_08_02_90", "korean_PP-OCRv3_rec")
    
    
    
    # models = mdb.get_name_list()
    # print(models)
    # print(models)
    # print(models[0])
    # print(mdb.get(models[0]))
    
    
    