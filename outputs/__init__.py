from pathlib import Path
import yaml
from pretrained_models import PretrainedModelDB
from labelsets import LabelsetDB
from datasets import DatasetDB
import copy

class OutputDB:
    ROOT = Path("/home/outputs")
    CONFIG_FILE = "config.yml"
    
    def __init__(self, root=None):
        self.root = root if root else OutputDB.ROOT
        self.name_path = OutputDB.get_name_path_list(self.root)        

    @staticmethod    
    def get_path_list(root):
        path_list = root.glob("*")

        dataset_list = []
        for path in path_list:
            if OutputDB.is_target(path):
                dataset_list.append(path)
            elif path.is_dir():
                dataset_list += OutputDB.get_path_list(path)
            else:
                pass
        return dataset_list
            
    @staticmethod
    def get_name_path_list(root):  
        name_path = {}
        for path in OutputDB.get_path_list(root):
            with open(str(path/OutputDB.CONFIG_FILE)) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            key = config["DB"]["name"]
            value = config
            name_path[key] = value
        
        return name_path
    
    @staticmethod
    def is_target(path):
        path_list = path.glob("*")
        return any([path.name == OutputDB.CONFIG_FILE for path in path_list])
    
    def get_name_list(self):
        return list(self.name_path.keys())
    
    
    def make(self, origin_config, name, labelset_name, model_name, pretrained=True):
        with open(origin_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        config["DB"] = {
            "origin_config":origin_config,
            "name":name,
            "labelset_name":labelset_name,
            "model_name":model_name,
            "pretrained":pretrained
        }
        
        # assert labelset_name in LabelsetDB().get_name_list(), f"the labelset '{labelset_name}' does not exist!"
        # assert model in PretrainedModelDB().get_name_list(), f"model '{model}' does not exist!"
        
        model = str(PretrainedModelDB().get(model_name)["pretrained"]) if pretrained else ""
        labelset = LabelsetDB().get(labelset_name)
        
        data_dir = str(DatasetDB().root)
        config["Global"]["pretrained_model"]=model
        config["Global"]["print_batch_step"]=1
        config["Global"]["save_epoch_step"]=1
        config["Global"]["eval_batch_step"]=[0, 2000]
        config["Global"]["save_model_dir"] = str(self.root/name/"trained_model")
        config["Train"]["dataset"]["data_dir"] = data_dir
        config["Train"]["dataset"]["label_file_list"] = [str(x) for x in labelset["label"]["train"]]
        config["Eval"]["dataset"]["data_dir"] = data_dir
        config["Eval"]["dataset"]["label_file_list"] = [str(x) for x in labelset["label"]["eval"]]
        # config["Test"] = copy.deepcopy(config["Eval"])        
        # config["Test"]["dataset"]["data_dir"] = data_dir
        # config["Test"]["dataset"]["label_file_list"] = [str(x) for x in labelset["label"]["test"]]

        (self.root/name).mkdir(parents = True, exist_ok=True)
        with open(self.root/name/OutputDB.CONFIG_FILE, "w") as f:
            yaml.dump(config, f)   
            
    def get(self, name):
        return self.name_path[name]