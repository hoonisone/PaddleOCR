from pathlib import Path
from matplotlib.dates import get_epoch
import yaml


from .labelset_db import LabelsetDB, LabelsetDB2
from .model_db import ModelDB
from .db import DB, DB2, Record2
import copy
import project
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import itertools
import fcntl
import pandas as pd
from tqdm import tqdm
import json
import pickle
from functools import lru_cache


def smooth(x, window):
    new = []
    x = list(x)
    for i in range(len(x)):
        r = max(i-window+1, 0)
        q = i
        new.append(sum(x[r:q+1])/(q-r+1))
        # print(new)
    return new
    
    
    
class WorkDB(DB):
    DIR = "./works"
    CONFIG_NAME = "work_config.yml"
    
    def __init__(self, dir = DIR, config_name = CONFIG_NAME):
        super().__init__(dir, config_name)
        
    def get_config(self, id, relative_to=None):
        config = super().get_config(id)
        
        if relative_to:
            for k in ["inference_result_dir", "trained_model_dir", "train_config", "report_file", "export_dir", "train_cache_file", "eval_cache_file"]:
                config[k] = self.relative_to(id, config[k], relative_to=relative_to)    
        return config      
        
        
    def get_df(self):
        ides = self.get_all_id()
        df = pd.DataFrame()
        for id in ides:
            config = self.get_config(id)
            new_df = pd.DataFrame([{"id": id, "labelsets": config["labelsets"], "model": config["model"], "trained_epoch": self.trained_epoch(id)}])
            df = pd.concat([df, new_df])
            
        return df
    def make(self, name, labelsets, model):
        assert name not in self.get_all_id(), f"the work '{name}' already exist!"
        for labelset in labelsets:
            assert labelset in LabelsetDB().get_all_id(), f"the labelset '{labelset}' does not exist!"
        assert model in ModelDB().get_all_id(), f"model '{model}' does not exist!"
        
        modeldb = ModelDB()
        labeldb = LabelsetDB()
        model_config = modeldb.get_config(model, relative_to="absolute")
        train_config = model_config["train_config"]
        
        model_task = set(model_config["task"])
        labelset_task = [set(labeldb.get_config(id)["task"]) for id in labelsets]
        task = list(reduce(lambda s1, s2: s1 & s2, labelset_task+[model_task]))
        
        
        
        assert 0 < len(task)

        config = {
            "task":task,
            "train_config":f"train_config.yml".replace("\\", "/"),
            "labelsets":labelsets,
            "model":model,
            "report_file": "report.csv",
            "trained_model_dir": "trained_model",
            "inference_result_dir": "inference_result",
            "export_dir": "inference_models"
        }

        save_dir = Path(self.PROJECT_ROOT)/self.DIR/name
        save_dir.mkdir(parents = True, exist_ok=True)
        
        with open(save_dir/self.CONFIG_NAME, "w") as f:
            yaml.dump(config, f)
        
        # copy the model config to work train config
        with open(str(Path(project.PROJECT_ROOT)/train_config)) as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
            with open(save_dir/"train_config.yml", "w") as f:
                yaml.dump(train_config, f)
                
        new_weight_path = save_dir/"trained_model"/"latest.pdparams"
        model_weight_path = Path(model_config["pretrained_model_weight"])
        if model_weight_path.exists():
            new_weight_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(model_weight_path, new_weight_path)
                
    def export(self, id, version=None, relative_to="absolute", command_to="global"):
        config = self.get_config(id, relative_to="absolute")
        
        train_config = f"{config['trained_model_dir']}/train_config.yml"
        checkpoint = self.get_model_weight(id, version=version, relative_to=relative_to)
        
        inference_model_path = self.make_inference_model_path(id, version, relative_to=relative_to)
        
        code = self.get_command_code(id, "export")
        options = {
            "Global.save_inference_dir":inference_model_path, # 저장될 infer model의 path (without extension)
            "Global.checkpoints":checkpoint,
            "Global.pretrained_model": checkpoint,
            }
        command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
        print(command)
        
        save_path = self.save_relative_to(id, "export.sh", relative_to=relative_to, save_to=command_to)
        with open(save_path, "a") as f:
            f.write(command+"\n")
       
    def update(self, work): # work에 수정이 일어난 경우 일관성을 위해 정리해주는 함수
        
        # 1. report 업데이트
        report_df = self.get_report_df(work)
        report_df["work_id"] = work
        self.save_report_df(work, report_df)   
        
    def get_report_df(self, id):
        config = self.get_config(id, relative_to="absolute")
        config["report_file"]
        report_path = self.save_relative_to(id, config["report_file"], "absolute", "local")
        if Path(report_path).exists():
            return pd.read_csv(report_path, index_col=0)
        else:
            return pd.DataFrame({"work_id":[], "version":[], "task":[]})
    
    def save_report_df(self, id, df):
        
        
        config = self.get_config(id, relative_to="absolute")
        report_path = self.save_relative_to(id, config["report_file"], "absolute", "local")

        df.to_csv(report_path)

    

    
    def modify_train_config(self, id, key, value):
        
        config = self.get_config(id, relative_to="project")
        train_config_name = Path(config["train_config"]).name
        
        train_config = super().get_config(id, config_name=train_config_name)
        config = train_config
        keys = key.split(".")
        for k in keys[:-1]:
            config = config[k]
        config[keys[-1]] = value
        super().update_config(id, train_config, config_name=train_config_name)
        
        # print(train_config_name)
        
        # config = self.get_config(id, relative_to="absolute", config_name=)
    
    def get_train_config(self, id):
        config = self.get_config(id, relative_to="project")
        train_config_name = Path(config["train_config"]).name
        return super().get_config(id, config_name=train_config_name)
    
    def modify_train_config2(self, id):
        config = self.get_config(id, relative_to="project")
        train_config_name = Path(config["train_config"]).name
        
        train_config = super().get_config(id, config_name=train_config_name)
        config = train_config
        del config["grapheme"]
        config["Global"]["grapheme"] = ["character", "first", "second", "third"]
        super().update_config(id, train_config, config_name=train_config_name)
        
        # print(train_config_name)
        
        # config = self.get_config(id, relative_to="absolute", config_name=)

        
    
    def report_eval(self, id, report):
        # 기존 데이터 로드
        
        with open("/home/works/log.txt", "a") as f:
            with open("/home/works/eval_lack", 'w') as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX) 
                try:
                    df = self.get_report_df(id)
                    f.write(f"{id}, {report['version']}, {len(df)}\n")
                    new_df = pd.DataFrame([report])
                    new_df = pd.concat([df, new_df])
                    self.save_report_df(id, new_df)
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    
        
        
        
        
        
        # pre_len = len(df)
        
        # # 데이터 추가

    
        # # 저장
        
        # new_df = self.get_report_df(id)
        # with open("/home/works/log.txt", "a") as f:
        #     f.write(f"{pre_len}, {len(new_df)}\n")
        #     if pre_len +1 != len(new_df):
            
        #         f.write(f"경고!! {report}\n")
        #         self.save_report_df(id, df)
        
    def get_work_dir_path(self, id):
        return Path(self.PROJECT_ROOT)/self.dir/id

    def get_report_value(self, id, version, task):
        df = self.get_report_df(id)
        df = df[(df["work_id"]==id) & (df["version"]==version) & (df["task"]==task)]
        if len(df) == 0:
            return None
        else:
            return df.iloc[0]
    
    def get_max_epoch(self, id):
        def get_max(extension:str):
            files = list(Path(self.get_config(id, relative_to="absolute")["trained_model_dir"]).glob(f"iter_epoch_*.{extension}"))
            if len(files) == 0:
                return 0    
            else:
                return max([int(path.stem[11:], base=0) for path in files])
        max_list = [get_max(extension) for extension in ["pdparams", "pdopt", "states"]]
        
        return max(max_list)
    
    def get_all_epoch(self, id):
        def get_epoch(extension:str):            
            files = list(Path(self.get_config(id, relative_to="absolute")["trained_model_dir"]).glob(f"iter_epoch_*.{extension}"))
            return [int(path.stem[11:]) for path in files]
        
        max_list = [get_epoch(extension) for extension in ["pdparams", "pdopt", "states"]]
        return sorted(list(set(sum(max_list, []))))        
    
    def check_weight_exist(self, id, version):
        return Path(self.make_model_weight_path(id, version)).exists()
        
        # epoch_list = self.get_all_epoch(id)
        # return version in epoch_list
                
    def make_inferenece_model_name(self, version):
        return f"inference_{version}"
    
    def make_inference_model_path(self, id, version, relative_to="absolute"):
        config = self.get_config(id, relative_to=relative_to)
        dir = config['export_dir']
        name = self.make_inferenece_model_name(version)
        return f"{dir}/{name}"
    
    
    def get_inference_model(self, id, version, relative_to="absolute"):
        path = self.make_inference_model_path(id, version, relative_to=relative_to)
        return path if Path(path).exists() else None

                    
        
    
    def make_model_weight_path(self, id, version, relative_to="absolute", extension = "pdparams"):
        assert (isinstance(version, int) and version > 0) or (isinstance(version, str) and (version in ["best", "latest", "origin", "last"])), version
        
        config = self.get_config(id)
        if version == "best":
            path = self.relative_to(id, Path(config["trained_model_dir"])/f"best_model/model.{extension}", relative_to=relative_to)
        elif version == "latest":
            path = self.relative_to(id, Path(config["trained_model_dir"])/f"latest.{extension}", relative_to=relative_to)      
        elif version == "origin": # 처음 모델 가중치 그대로 (초기화 또는 다른 테스크에서 pretrained)
            model_config = ModelDB().get_config(config["model"], relative_to=relative_to)
            path = str(model_config["pretrained_model_weight"])
            path = path[:-8]+extension
        elif version == "last":

            path = self.relative_to(id, Path(config["trained_model_dir"])/f"iter_epoch_{self.get_max_epoch(id)}.{extension}", relative_to=relative_to)
        else:
            path = self.relative_to(id, Path(config["trained_model_dir"])/f"iter_epoch_{version}.{extension}", relative_to=relative_to)
        return path
            
    def weight_file_exist(self, id, version):
        for extension in ["pdparams", "pdopt", "states"]:
            if Path(self.make_model_weight_path(id, version, relative_to="absolute", extension=extension)).exists():
                return True
        return False

    def get_model_weight(self, id, version, relative_to="absolute", no_exist_handling = False):
        if relative_to is not "absulute":
            Exception("현재 relative_to 변수가 absolute가 아니면 무조건 latest weight만 반환되는 오류가 있는데 방치해둠...")
        path = self.make_model_weight_path(id, version, relative_to=relative_to)    
        print(path)
        if Path(path).exists():    
            return path
        else:
            if no_exist_handling:
                return self.make_model_weight_path(id, "latest", relative_to=relative_to)
            else:
                return ""
    
    def get_unevaluated_epoches(self, id, task):
        epoches = self.get_all_epoch(id)
        evaluated_epoches = self.get_evaluated_epoches(id, task)
        return sorted(list(set(epoches) - set(evaluated_epoches)))
    
    def command_split(self, type, mode, n, shuffle = False, reverse_order = False):
        with open(f"/home/works/{type}.sh") as f:
            lines = f.readlines()
        
        if reverse_order:
            lines = lines[::-1]
        
        if shuffle:
            import random
            random.shuffle(lines)
            
        
        
        splits = [[] for i in range(n)]
        for i, line in enumerate(lines):
            splits[i%n].append(line)
                
                

        for i, split in enumerate(splits):
            with open(f"/home/works/{type}{i}.sh", mode) as f:
                for line in split:
                    f.write(line)  
                
    def make_empty_eval_file(self, mode):
        assert mode in ["train", "eval", "infer", "export"]
        with open(f"/home/works/{mode}.sh", "w") as f:
            f.write("")
    
    def eval_all(self, id, task, relative_to="absolute", command_to="global", check_result = True, labelsets=None, data_dir=None, save = True, filter = None):
        
        unevaluated_epoches =  self.get_unevaluated_epoches(id, task)
        
        code = self.get_command_code(id, "eval", relative_to=relative_to)
        config = self.get_config(id, relative_to="project")
        labelset_configs = [LabelsetDB().get_config(id, relative_to="project") for id in config["labelsets"]]
        
        if (data_dir == None) or (labelsets == None):
            data_dir = labelset_configs[0]["dataset_dir"]
            labelsets = sum([c["label"][task] for c in labelset_configs], [])
        
        for version in unevaluated_epoches:
            if filter:
                if not filter(version):
                    continue
            
            
            train_config = config["train_config"]
            model_weight = self.get_model_weight(id, version, relative_to=relative_to)
            model_weight = ".".join(model_weight.split(".")[:-1]) # 확장자 제거


            options = {
                    "Global.work_id":id,
                    "Global.version":version,
                    "Global.eval_task":task,
                    "Global.checkpoints":model_weight,
                    "Global.save_model_dir":config["trained_model_dir"],
                    "Global.use_amp":False,
                    "Eval.dataset.data_dir": data_dir,
                    "Eval.dataset.label_file_list":f"""['{"','".join(labelsets)}']""",
                    "Eval.save":save,
                    "Eval.check_exist":check_result,
                    # "Eval.dataset.cache_file": config["eval_cache_file"]
                    }
            
            command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
            
            save_path = self.save_relative_to(id, "eval.sh", relative_to=relative_to, save_to=command_to)
            with open(save_path, "a") as f:
                f.write(command+"\n")    
    
    def sort_report_df(self, id):
        df = self.get_report_df(id)
        df.sort_values(["task", "version"], inplace=True)
        self.save_report_df(id, df)
    
    
    def eval_one(self, id, version, task, relative_to="absolute", command_to="global", report_to="local", check_result=True, check_weight=True, labelsets=None, data_dir=None, save = True):


        item = self.get_report_value(id, version=version, task=task)
        
        if check_result:
            # 이미 결과가 있으면 취소
            if not(isinstance(item, type(None)) or item.empty):
                print(f"(id:{id}, version:{version}, task:{task}) already evaluated")
                return 
            
        if check_weight:
            # weight이 없으면 취소
            
            if not self.check_weight_exist(id, version):
                print(f"(id:{id}, version:{version}, task:{task}) has no weight")
                return
                
            
        code = self.get_command_code(id, "eval", relative_to=relative_to)
        
        config = self.get_config(id, relative_to="project")
        labelset_configs = [LabelsetDB().get_config(id, relative_to="project") for id in config["labelsets"]]
        
        train_config = config["train_config"]
        model_weight = self.get_model_weight(id, version, relative_to=relative_to)
        model_weight = ".".join(model_weight.split(".")[:-1]) # 확장자 제거

        if (data_dir == None) or (labelsets == None):
            data_dir = labelset_configs[0]["dataset_dir"]
            labelsets = sum([c["label"][task] for c in labelset_configs], [])

        options = {
                "Global.work_id":id,
                "Global.version":version,
                "Global.eval_task":task,
                "Global.checkpoints":model_weight,
                "Global.save_model_dir":config["trained_model_dir"],
                "Global.use_amp":False,
                "Eval.dataset.data_dir": data_dir,
                "Eval.dataset.label_file_list":f"""['{"','".join(labelsets)}']""",
                "Eval.save":save,
                "Eval.check_exist":check_result,
                # "Eval.dataset.cache_file": config["eval_cache_file"]
                }
        
        command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
        print(command)
        
        save_path = self.save_relative_to(id, "eval.sh", relative_to=relative_to, save_to=command_to)
        with open(save_path, "a") as f:
            f.write(command+"\n")       
    
    def get_evaluated_epoches(self, id, task):
        df = self.get_report_df(id)
        df = df[df["task"] == task]
        return df["version"].unique()
    
            
    def get_command_code(self, id, task, relative_to="project"):
        assert task in ["train", "eval", "infer", "export"], f"code should be 'train', 'eval', or 'infer' but {task} is given"
        assert relative_to in ["absolute", "project"], f"relative_to should be 'absolute' or 'project' but {relative_to} is given"
        if task == "train":
            task = "code/PaddleOCR/tools/train.py"
        elif task == "eval":
            task = "code/PaddleOCR/tools/eval.py"
        elif task == "infer":
            task = self.get_config(id)["task"]
            if "STD" in task:
                task = "code/PaddleOCR/tools/infer_det.py"
            elif "STR" in task:
                task = "code/PaddleOCR/tools/infer_rec.py"
            else:
                None
        elif task == "export":
            task = "code/PaddleOCR/tools/export_model.py"
        else:
            task = None
        
        if relative_to == "absolute":
            task = str(Path(self.PROJECT_ROOT)/task).replace('\\', '/')
        elif relative_to == "project":
            pass
        return task
    
    
    
        
    # 학습 코드 생성 (성공 실패 여부 반환) 
    def train(self, id, version, epoch, relative_to="project", command_to="global", epoch_check=True):
        assert relative_to in ["absolute", "project"], f"relative_to should be 'absolute' or 'project' but {relative_to} is given"
        
        
        pass_flag = False
        if epoch_check:
            trained_epoch = self.get_max_epoch(id)
            if epoch <= trained_epoch:
                pass_flag = True

        print(f"""Trained ({trained_epoch:3d}/{epoch:3d}) | {id} \t{"(Passed)"if pass_flag else ""}""")
        if pass_flag:
            return False
            
        code = self.get_command_code(id, "train", relative_to=relative_to)
        config = self.get_config(id, relative_to="absolute")
        labelset_configs = [LabelsetDB().get_config(id, relative_to="project") for id in config["labelsets"]]
        self.get_work_dir_path(id)
        
        
        
        model_weight = self.get_model_weight(id, version, relative_to=relative_to)
        model_weight = ".".join(model_weight.split(".")[:-1]) # 확장자 제거
        
        train_config = config["train_config"]
        train_labelsets = sum([c["label"]["train"] for c in labelset_configs], [])
        eval_labelsets = sum([c["label"]["eval"] for c in labelset_configs], [])
        
        options = {
                   "Global.checkpoints":model_weight,
                   "Global.epoch_num":epoch,
                   "Global.save_model_dir":config["trained_model_dir"],
                   
                   "Train.dataset.data_dir": labelset_configs[0]["dataset_dir"],
                   "Train.dataset.label_file_list":f"""['{"','".join(train_labelsets)}']""",
                #    "Train.dataset.cache_file": config["train_cache_file"],
                   
                   "Eval.dataset.data_dir": labelset_configs[0]["dataset_dir"],
                   "Eval.dataset.label_file_list":f"""['{"','".join(eval_labelsets)}']""",
                #    "Eval.dataset.cache_file": config["eval_cache_file"]
                   }
        
        command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
        # print(command)
        
        save_path = self.save_relative_to(id, "train.sh", relative_to=relative_to, save_to=command_to)
        with open(save_path, "a") as f:
            f.write(command+"\n")
        return True

    def make_datase_cache(self, id, version, epoch, relative_to="project", command_to="global", epoch_check=True):
        assert relative_to in ["absolute", "project"], f"relative_to should be 'absolute' or 'project' but {relative_to} is given"
                
        code = self.get_command_code(id, "make_dataset_cache", relative_to=relative_to)
        config = self.get_config(id, relative_to="absolute")
        labelset_configs = [LabelsetDB().get_config(id, relative_to="project") for id in config["labelsets"]]

        train_config = config["train_config"]
        
        train_labelsets = sum([c["label"]["train"] for c in labelset_configs], [])
        eval_labelsets = sum([c["label"]["eval"] for c in labelset_configs], [])
        test_labelsets = sum([c["label"]["eval"] for c in labelset_configs], [])
        
        eval_labelsets += test_labelsets
        
        options = {
                   "Train.dataset.data_dir": labelset_configs[0]["dataset_dir"],
                   "Train.dataset.label_file_list":f"""['{"','".join(train_labelsets)}']""",
                   "Train.dataset.cache_file": config["train_cache_file"],
                   
                   "Eval.dataset.data_dir": labelset_configs[0]["dataset_dir"],
                   "Eval.dataset.label_file_list":f"""['{"','".join(eval_labelsets)}']""",
                   "Eval.dataset.cache_file": config["eval_cache_file"]
                   }
        
        command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
        # print(command)
        
        save_path = self.save_relative_to(id, "make_dataset_cache.sh", relative_to=relative_to, save_to=command_to)
        with open(save_path, "a") as f:
            f.write(command+"\n")
        return True

    def save_relative_to(self, id, path, relative_to="project", save_to="global"):
        assert save_to in ["global", "local"], f"save_to should be 'global' or 'local' but {save_to} is given"
        
        if relative_to=="absolute":
            root = Path(project.PROJECT_ROOT)/self.DIR
        elif relative_to=="project":
            root = Path(self.DIR)
        else:
            root = None
            
        if save_to == "global":
            return str(root/path).replace("\\", "/")
        elif save_to == "local":
            return str(root/id/path).replace("\\", "/")
        else:
            return None
        
    def get_infer_result_path(self, id, version, task):
        print("!!!  제대로 구현 안함")
        config = self.get_config(id, relative_to="absolute")
        return config["inference_result_dir"]
    
    def get_infer_result(self, id, version, task):
        path = self.get_infer_result_path(id, version, task)
        print(path)
        with open(path) as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
            lines = [[path, json.loads(label)] for path, label in lines]
            return lines
        
    def get_infer_result_df(self, id, version, task):
        result_dict = dict()
        infer_result = self.get_infer_result(id, version, task)
        for img_path, pred in tqdm(infer_result):
            for model_name, pred in pred.items():
                for method, pred in pred.items():
                    name = f"{model_name}_{method}"
                    p_name = f"{name}_prob"
                    result_dict.setdefault(name, [])
                    result_dict.setdefault(p_name, [])
                    text, prob = pred[0]
                    
                    result_dict[name].append(text)
                    result_dict[p_name].append(prob)
                    # result.dict[p_name].append(json.dumps(prob))
            result_dict.setdefault("image", [])
            result_dict["image"].append(img_path)
        return pd.DataFrame(result_dict)

        

    def infer(self, id, version = "best", task="test", relative_to="absolute", command_to="global", data_dir=None, labelsets=None, save_path = None):
        assert task in ["train", "eval", "test"]
        assert command_to in ["global", "local"]
        config = self.get_config(id, relative_to=relative_to)
        if "STR" in config["task"]:
            code = f"{project.PROJECT_ROOT}/code/PaddleOCR/tools/infer_rec.py"
        elif "STD" in config["task"]:
            code = f"{project.PROJECT_ROOT}/code/PaddleOCR/tools/infer_det.py"
        
        ppocr_config = config["train_config"]  
        
        model_weight = self.get_model_weight(id, version, relative_to=relative_to)
        model_weight = ".".join(model_weight.split(".")[:-1]) # 확장자 제거
        
        labelset_ids = config["labelsets"]
        
        if data_dir==None and labelsets==None:
            data_dir = LabelsetDB().get_config(labelset_ids[0], relative_to=relative_to)["dataset_dir"]
            labelsets = sum([LabelsetDB().get_config(id, relative_to=relative_to)["infer"][task] for id in labelset_ids], [])
        
        
        save_path = save_path if save_path else config["inference_result_dir"]

        command = f"python {code} -c {ppocr_config} -o Global.pretrained_model={model_weight} Global.checkpoints={model_weight} Global.save_model_dir={model_weight} Global.save_res_path={save_path} Infer.data_dir={data_dir} Infer.infer_file_list={labelsets}" # train에 대해서도 할 수 있게 수정해야 함
        print(command)

        save_path = self.save_relative_to(id, "infer.sh", relative_to=relative_to, save_to=command_to)
        with open(save_path, "a") as f:
            f.write(command+"\n")


    def get_best_epoch(self, id, task = "test", metric=None):
        if metric is None:
            config = self.get_config(id)
            metric = config["metric"]
        
        df = self.get_report_df(id)
        df.reset_index(inplace=True)
        df = df[df["task"] == task]
        df = df[df["version"] != "best"]

        return df.loc[df[metric].idxmax()].version
    
    def get_best_report(self, id, task = "test", metric="acc"):
        epoch = self.get_best_epoch(id, task, metric=metric)
        df = self.get_report_df(id)
        return df[df["version"] == epoch]

    def draw_det_graph(self, id, window=1):
        plt.gcf().set_size_inches(8, 3)
    
        df = self.get_report_df(id).sort_values("version")
        plt.subplot(1, 2, 1)
        plt.title(f"Precision")
        plt.xlabel("Epochs")
        for task in ["train", "eval", "test"]:
            task_df = df[df["task"] == task]
            data = smooth(task_df["precision"], window=window)
            plt.plot(task_df["version"], data, label=f"{task}")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.title(f"Recall")
        plt.xlabel("Epochs")
        for task in ["train", "eval", "test"]:
            task_df = df[df["task"] == task]
            data = smooth(task_df["recall"], window=window)
            plt.plot(task_df["version"], data, label=f"{task}")   
        plt.legend()   
    
    def draw_rec_graph(self, id, window=1, tasks = ["train", "eval", "test"], labels = ["train", "eval", "test"]):
        plt.gcf().set_size_inches(8, 3)
        
        df = self.get_report_df(id).sort_values("version")
        
        plt.subplot(1, 2, 1)
        plt.title(f"Accuracy")
        plt.xlabel("Epochs")
        for task, label in zip(tasks, labels):
            task_df = df[df["task"] == task]
            data = smooth(task_df["acc"], window=window)
            plt.plot(task_df["version"], data, label=label)
        plt.legend()
            
        plt.subplot(1, 2, 2)
        plt.title(f"Norm-Edit-Distance")
        plt.xlabel("Epochs")
        for task, label in zip(tasks, labels):
            task_df = df[df["task"] == task]
            data = smooth(task_df["norm_edit_dis"], window=window)
            plt.plot(task_df["version"], data, label=label)    
        plt.legend()
        return plt    
    
    
    
    def draw_rec_graph_v2(self, id, window=1, tasks = ["train", "eval", "test"], labels = ["train", "eval", "test"], metrics = ["acc", "norm_edit"], titles = None, linestyle = "-", color = None):
        plt.gcf().set_size_inches(4*len(metrics), 3)
        
        df = self.get_report_df(id).sort_values("version")
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i+1)
            inner_title = titles[i] if titles else metric
            plt.title(inner_title) 
            plt.xlabel("Epochs")
            for task, label in zip(tasks, labels):
                task_df = df[df["task"] == task]
                data = smooth(task_df[metric], window=window)
                if task == "eval":
                    task = "test"
                plt.plot(task_df["version"], data, label=label, linestyle = linestyle, color = color)
            if i == 0:
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            
        # plt.subplot(1, 2, 2)
        # plt.title(f"Norm-Edit-Distance")
        # plt.xlabel("Epochs")
        # for task, label in zip(tasks, labels):
        #     task_df = df[df["task"] == task]
        #     data = smooth(task_df["norm_edit_dis"], window=window)
        #     if task == "eval":
        #         task = "test"
        #     plt.plot(task_df["version"], data, label=label)    
        # plt.legend()
        return plt    

# if __name__ == "__main__":
#     mdb = WorkDB()
#     id = "rec___ABI_A___aihub_rec_10k_horizontal___C_ALL+IMF"
#     task = "test"
#     print(mdb.get_evaluated_epoches(id, "test"))
    
#     mdb.eval_all(None, id, task, relative_to="project", command_to="global", report_to="local", check_result=True, check_weight=True, labelsets=None, data_dir=None, save = True)
    
#     mdb.make("/home/configs/det/det_mv3_db.yml", "output1", "ai_hub_det_08_02_90_random_k_fold_5_1", "MobileNetV3_large_x0_5")
#     mdb.make("/home/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml", "output2", "ai_hub_rec_08_02_90", "korean_PP-OCRv3_rec")
    
    
    
#     models = mdb.get_name_list()
#     print(models)
#     print(models)
#     print(models[0])
#     print(mdb.get(models[0]))


class WorkDB2(DB2):
    def __init__(self, name = "works"):
        super().__init__(name, record_class = Work2, record_config_name="work_config.yml")
        
        self.path_dir = {
            "train_bash": self.dir/"train.sh",
            "eval_bash": self.dir/"eval.sh",
            "eval_result": self.dir/"eval_result.csv",
            "infer_bash": self.dir/"infer.sh",
        }

    
    def make_empty_eval_file(self, mode):
        assert mode in ["train", "eval", "infer", "export"]
        with open(f"/home/works/{mode}.sh", "w") as f:
            f.write("")
    
    def command_split(self, task, mode, n, shuffle = False, reverse_order = False):
        """
        command bash 파일의 명령어를 n개로 쪼개어 저장
        mode: 파일 읽기 모드 (누적할 지 새로 쓸 지 결정)
        """
        with open(f"/home/works/{task}.sh") as f:
            lines = f.readlines()
        
        if reverse_order:
            lines = lines[::-1]
        
        if shuffle:
            import random
            random.shuffle(lines)
            
        
        
        splits = [[] for i in range(n)]
        for i, line in enumerate(lines):
            splits[i%n].append(line)
                
                

        for i, split in enumerate(splits):
            with open(f"/home/works/{task}{i}.sh", mode) as f:
                for line in split:
                    f.write(line)  
                    
                    
    def read_commands(self, task):
        path = self.get_path(f"{task}_bash")
        with open(path) as f:
            lines = f.readlines()
        return lines
    
    def print_commands_num(self, task):
        lines = self.read_commands(task)
        print(f"{task} commands: ", len(lines))

    @property
    def record_info_table(self):
        info_list = [record.info for id, record in self.records.items()]
        df = pd.DataFrame(info_list, index = None)
        return df.reset_index(drop=True)
    
    def init_cache(self):
        for record in self.records.values():
            record.cache_manager.init_cache()

class CommandFileWriter:
    """
        명령어를 파일에 쓸 때 사용하는 클래스
        만든 이유는 상황에 따라 명령어를 쓰는 순간에만 파일을 열거나
        아니면 원하는 시점에 파일을 열거나 닫는 기능을 지원해서
        반복적인 명렁어 쓰기를 효율적으로 수행하기 위함
        
        명령어 쓰기를 반복 수행을 효율적으로 하기 위해서는
        1. open_and_save_command_file을 수해하여 파일을 열어둠
        2. write_command를 원하는 만큼 호출
        3. 작업이 끝나면 close_and_remove_command_file을 호출하여 파일을 닫음
    """
    def __init__(self, workdb):
        self.workdb = workdb
    
    def get_file_attr_name(self, task):
        """ task에 맞는 파일 포인터를 저장하는 객체의 속성이름 반환"""
        return f"{task}_command_file"
    
    def get_file_path(self, task):
        """ 
            task에 맞는 command 명령어를 저장하는 파일의 경로 반환 
            실제 파일 경로는 workdb가 관리함
        """
        return self.workdb.get_path(f"{task}_bash", level = "db")
    

    def get_file_attr(self, task):
        """
            return the file pointer for the task
            but return None if the file is not opened
        """
        attr_name = self.get_file_attr_name(task)
        f = getattr(self, attr_name, None)
        return f

    def open_file(self, task, mode):
        """
            task에 대한 파일을 열어서 포인터 반환
            mode: 파일 열기 모드
        """
        file_path = self.get_file_path(task)
        f = open(file_path, mode)
        return f
    
    def open_and_save_file(self, task, mode = "a"):
        """
            task에 맞는 파일 포인터가 열려있는지 확인하고
            열려있지 않다면 열고 객체에 속성값으로 저장
        """
        f = self.get_file_attr(task) # 기존에 열려있는 파일 포인터 요청
        if f is None:   # 기존에 열린게 없다면
            f = self.open_file(task, mode) # 새롭게 열고
            attr_name = self.get_file_attr_name(task)
            setattr(self, attr_name, f) # 저장
            
    def close_and_remove_file(self, task):
        f = self.get_file_attr(task)
        if f is not None:
            f.close()
            attr_name = self.get_file_attr_name(task)
            setattr(self, attr_name, None)
            
    def write(self, task, command):
        """
            file이 열렸있다면 바로 명령어를 쓰고
            없다면 임시적으로 열어서 명령어를 쓴다.
            command는 개행 없이 문자열로만 입력
        """
        f = self.get_file_attr(task)
        if f is not None:
            f.write(command+"\n")
        else:
            f = self.open_file(task, "a")
            f.write(command+"\n")
            f.close()



class PickleCache:
    def __init__(self, cache_file_path):
        self.cache_file_path = cache_file_path
        if not Path(self.cache_file_path).exists():
            self.save({})
        
    def load(self):
        with open(self.cache_file_path, "rb") as f:
            return pickle.load(f)
            
        
    def save(self, data):
        with open(self.cache_file_path, "wb") as f:
            pickle.dump(data, f)    
            
            
    def init_cache(self):
        cache_file_path = Path(self.cache_file_path)
        if cache_file_path.exists():
            cache_file_path.unlink()


class Work2(Record2):
    
    WEIGHT_EXTENSION = "pdparams"

    def __init__(self, db, name):
        
        
        super().__init__(db, name, ["inference_result_dir", "trained_model_dir", "train_config", "report_file", "export_dir", "train_cache_file", "eval_cache_file"])
        self.command_file_writer = CommandFileWriter(self)
        self.path_dir = {
            "train_bash": self.record_dir/"train.sh",
            "eval_bash": self.record_dir/"eval.sh",
            "infer_bash": self.record_dir/"infer.sh",
            "eval_result": self.record_dir/"eval_result.csv",
            
            "infer_det_command": self.project_dir/"code/PaddleOCR/tools/infer_det.py",
            "infer_rec_command": self.project_dir/"code/PaddleOCR/tools/infer_rec.py"
        }    
        
        self.not_metric_columns = ["work_id", "version", "fps", "labelset"]
        
        self.cache_manager = PickleCache(self.cache_file_path)

    @property
    def cache(self):
        return self.cache_manager.load()
    
    @cache.setter
    def cache(self, data):
        self.cache_manager.save(data)

    @property
    def cache_file_path(self):
        return self.record_dir/"cache.pkl"

    def is_cached(self, key):
        return key in self.cache

    def get_cache_data(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            return None
    
    def set_cache_data(self, key, data):
        cache = self.cache
        cache[key] = data
        self.cache = cache
    
    @property
    def info(self):
        if not self.is_cached("info"):
            main_indicator, best_epoch, best_score = self.main_indicator
            info = {
                
                "id": self.id,
                "name": self.name,
                "target_epoch": self.target_epoch,
                "train_epoch": self.trained_epoch,
                "main_indicator": main_indicator,
                "best_epoch": best_epoch,
                "best_score": best_score            
            }
        
            eval_labelset = self.labelset_argment_handling("eval")[0]
            info["unevaluated_epochs"] = len(self.get_unevaluated_epochs(eval_labelset))

            best_test_result = self.best_test_result
            if len(best_test_result) > 0:
                for labelset, value in self.best_test_result[["labelset_name", "value"]].itertuples(index=False):
                    info[labelset] = value
            
            self.set_cache_data("info", info)
        return self.get_cache_data("info")
        
    def validate_task_value(self, task, make_error = True):
        TASK = ["train", "eval", "test", "infer"]
        self.validate_value_by_valid_value_list(task, TASK, make_error = make_error)    

    def validate_save_to_value(self, task, make_error = True):
        SAVE_TO = ["global", "local"]
        self.validate_value_by_valid_value_list(task, SAVE_TO, make_error = make_error)    
        

    def validate_version_value(self, version, make_error = True): 
        TEXT_VERSION = ["scratch", "best", "latest", "origin", "last"]
        
        # validation step
        if isinstance(version, str):
            validity = version in TEXT_VERSION
        elif isinstance(version, int):
            validity = 0 < version
        else:
            raise ValueError(f"version should be str or int, but {type(version)} is given")
        # validity handling step 
        if make_error:
            assert validity, f"version should be in {TEXT_VERSION} or positive integer, but {version} is given"
        else:
            return validity
        
    def get_command_code(self, task):
        self.validate_task_value(task)
        
        if task == "train":
            command = "code/PaddleOCR/tools/train.py"
        elif task == "eval":
            command = "code/PaddleOCR/tools/eval.py"
        elif task == "infer":
            task = self.config["task"]
            if "STD" in task:
                command = "code/PaddleOCR/tools/infer_det.py"
            elif "STR" in task:
                command = "code/PaddleOCR/tools/infer_rec.py"
            else:
                raise NotImplementedError

        elif task == "export":
            command = "code/PaddleOCR/tools/export_model.py"
        else:
            raise NotImplementedError
        
        command =  self.relative_to(command, current_relative = "project", target_relative=self.RELATIVE_TO)
        return command
    
    

    def get_eval_result(self, version, labelset):
        # eval 결과중 조건에 맞는 값 반환
        
        df = self.result_df
        df = df[(df["version"]==version) & (df["labelset"]==labelset)]
        
        
        if len(df) == 0:
            return None
        else:
            return df.iloc[0]
    
    @property
    def result_df(self):
        config = self.config
        
        eval_result_path = self.get_path("eval_result", level = "record") # record 끼리 result file 구분
        if Path(eval_result_path).exists():
            df = pd.read_csv(eval_result_path, index_col=0)
        else:
            df = pd.DataFrame({"work_id":[], "version":[], "labelset":[]})
    
        df["version"] = df["version"].astype(int)
        df["labelset"] = df["labelset"].astype(int)
        
        return df
    
    def transform_result_df(self, df):
        "효율적으로 저장되어있으나 보기 힘든 result_df를 보기 좋게 변환"
        if len(df) == 0:
            return df
        
        ### 형태 변환
        df_list = list()
        not_metric_columns = ["work_id", "version", "labelset"]
        metric_columns = [col for col in df.columns if col not in not_metric_columns]
        for metric_column in metric_columns:
            df_temp = df[["work_id", "version", "labelset", metric_column]].copy()
            df_temp.rename(columns={metric_column:"value"}, inplace=True)
            df_temp["metric"] = metric_column
            df_list.append(df_temp)
        df = pd.concat(df_list)[["work_id", "version", "labelset", "metric", "value"]]
        df.sort_values(["work_id", "version", "labelset"], inplace=True)
        
        labelsetdb = LabelsetDB2()
        labelset_rerecods = labelsetdb.records
        df["labelset_name"] = df["labelset"].apply(lambda x: labelset_rerecods[x].name)
        
        df["version"] = df["version"].astype(int)
        df["labelset"] = df["labelset"].astype(int)
        
        return df
    
    
    
    @property
    @lru_cache(maxsize=1)
    def eval_result_df(self):
        eval_labelset = self.get_label_set("eval")
        df = self.result_df
        df = df[df["labelset"] == eval_labelset]
        return df
    
    @property
    @lru_cache(maxsize=1)
    def test_result_df(self):
        test_labelsets = self.get_label_set("test")
        df = self.result_df
        df = df[df["labelset"].isin(test_labelsets)]
        return df

    @property
    def best_test_result(self):
        df = self.test_result_df
        df = self.transform_result_df(df)
        if len(df) == 0:
            return df
        
        metric, version, score = self.main_indicator
        df = df[(df["metric"] == metric) & (df["version"] == version)]
        return df

    @result_df.setter
    def result_df(self, df):
        eval_result_path = self.get_path("eval_result", level = "record")
        df.to_csv(eval_result_path)       
        

    
    def is_evaluated(self, version, labelset):
        return self.get_eval_result(version, labelset) is not None
    

    
    def get_model_weight_path(self, version, validate=True):
        # validate: True이면 weight path에 대해 유효성 검사를 수행하고, False이면 결과값을 반환
        
        self.validate_version_value(version)
        
        config = self.config
        
        model_dir = Path(config["trained_model_dir"])
        if version == "scratch":
            return None
        elif version == "best":
            path = model_dir/f"best_model/model.{self.WEIGHT_EXTENSION}"
        elif version == "latest":
            path = model_dir/f"latest.{self.WEIGHT_EXTENSION}"      
        elif version == "origin": # 처음 모델 가중치 그대로 (초기화 또는 다른 테스크에서 pretrained)
            model_config = ModelDB().get_config(config["model"], relative_to=self.RELATIVE_TO)
            path = str(model_config["pretrained_model_weight"])
            path = path[:-8]+extension
        elif version == "last":
            path = model_dir/f"iter_epoch_{self.trained_epoch}.{self.WEIGHT_EXTENSION}"
        else:
            path = model_dir/f"iter_epoch_{version}.{self.WEIGHT_EXTENSION}"
        
        if validate:
            absolute_path = self.relative_to(path, current_relative = "record", target_relative = "absolute")
            if not Path(absolute_path).exists():
                return None
        
        return self.relative_to(path, current_relative = "record", target_relative = self.RELATIVE_TO)
    
    
    # EVAL_COMMAND_PATH = ""
    
    def get_label_set(self, task):
        """
        task에 맞는 레이블 셋 반환
        return: [labelset1, ...]
        """
        config = self.config
        labelset_dict = config["labelsets"]
        return labelset_dict[task]
    
    @property    
    def all_trained_epochs(self):
        config = self.config
        model_dir = Path(config["trained_model_dir"])
        weight_paths = list(model_dir.glob(f"iter_epoch_*.{self.WEIGHT_EXTENSION}"))
        epochs = sorted([int(path.stem[11:], base=0) for path in weight_paths])
        return epochs
    
    @property
    def trained_epoch(self):
        epochs = self.all_trained_epochs
        return max(epochs) if len(epochs) > 0 else 0     
            
    
    def get_evaluated_epochs(self, labelset):
        df = self.eval_result_df
        df = df[df["labelset"] == labelset]
        return df["version"].unique()
    
    def get_unevaluated_epochs(self, labelset):
        all_epochs = self.all_trained_epochs
        evaluated_epochs = self.get_evaluated_epochs(labelset)        
        return sorted(list(set(all_epochs) - set(evaluated_epochs)))


    

    def eval_all(self, labelsets=None, 
                 command_to="global", report_to="local", 
                 filter = None, verbose=False):
        """
            labelset-task 에 대하여 학습된 모든 epoch에 대해 평가를 수행
            filter: 전체 epoch중 어떤 epoch에 대해 평가를 수행할 지 결정하는 함수
        """
    
        labelsets = self.labelset_argment_handling(labelsets)

        self.command_file_writer.open_and_save_file("eval") # 효율적인 파일 쓰기를 위해 파일 열어둠
        for labelset in labelsets:  
            unevaluated_epochs =  self.get_unevaluated_epochs(labelset)
            if filter is not None:
                unevaluated_epochs = [epoch for epoch in unevaluated_epochs if filter(epoch)]
            print("unevaluated_epochs: ", unevaluated_epochs)
            for version in unevaluated_epochs:    
                self.eval_one(version, labelset,  
                            command_to=command_to, report_to=report_to, 
                            check_result=False, check_weight=False, save = True, verbose=verbose)
        self.command_file_writer.close_and_remove_file("eval") # 작업이 끝났으니 command file을 닫음
    
    
    
    def labelset_argment_handling(self, labelsets):
        """
        labelsets 입력에 특수 값을 처리하여 labelset으로 변환
        """
        if labelsets in ["train", "eval", "test"]:
            labelsets = self.get_label_set(labelsets)
                    
        if not isinstance(labelsets, list):
            labelsets = [labelsets]
        
        return labelsets
    
    def test(self):
        indicator, epoch, score = self.main_indicator
        
        if epoch is None:
            print("no main indicator, check evaluation is completed")
        else:
            print(f"weight infor: indicator={indicator}, epoch={epoch}, score={score}")
            self.eval_one(version = epoch, labelsets = "test",  check_result=True, save=True, verbose=True)

    def eval_one(self, version, labelsets=None, 
                 command_to="global", report_to="local", 
                 check_result=True, check_weight=True, save = True, verbose=False, command_file = None):
        '''
            check_result: 결과가 이미 있으면 실행하지 않음
            check_weight: weight이 없으면 실행하지 않음
            command_to, report_to = {global, local} => work 별로 구분할 지 말지
        '''
        labelsets = self.labelset_argment_handling(labelsets)
        for labelset in labelsets:    
            # evaluation result handling step 
            if check_result and self.is_evaluated(version, labelset):
                print(f"(id:{self.id}, version:{version}, labelset:{labelset}) already evaluated")
                continue

            # model weight handling step
            model_weight_path = self.get_model_weight_path(version, validate=True)
            if check_weight and model_weight_path is None:
                    print(f"(id:{self.id}, version:{version} has no weight")
                    continue
            model_weight_path = str(Path(model_weight_path).with_suffix('')).replace("\\", "/") # 확장자 제거

            # command code handling step(코드 실행 파이썬 파일 경로)
            code = self.get_command_code("eval")
        
            # config handling step
            config = self.config
            
            # labelset handling step
            labelset = LabelsetDB2().get_record(labelset)
            labelset_config = labelset.config
            data_dir = labelset_config["dataset_dir"]
            labelset_file = labelset_config["label"]
            
            train_config = config["train_config"]

            options = {
                    "Global.work_id":self.id,
                    "Global.labelset":labelset.id,
                    "Global.version":version,
                    "Global.checkpoints":model_weight_path,
                    "Global.save_model_dir":config["trained_model_dir"],
                    "Global.use_amp":False,
                    "Eval.dataset.data_dir": data_dir,
                    "Eval.dataset.label_file_list":f"{labelset_file}",
                    "Eval.save":save,
                    "Eval.check_exist":check_result,
                    # "Eval.dataset.cache_file": config["eval_cache_file"]
                    }
            
            self.make_and_write_command_context("eval", code, train_config, options, verbose)

    def get_infer_result_path(self, version, labelset):
        config = self.config
        save_path = Path(config["inference_result_dir"])/f"epoch_{version}"/f"labelset_{labelset}/infer_result.txt"
        save_path = str(save_path).replace("\\", "/")
        return save_path
    

    def infer(self, version, labelsets=None, command_to="global",
              check_result=True, check_weight=True, verbose=False):
        # argument validation step
        self.validate_version_value(version)
        
        code = self.get_command_code("infer")
        train_config = self.config["train_config"]
        
                
        model_weight_path = self.get_model_weight_path(version, validate=True)
        if check_weight and model_weight_path is None:
            print(f"(id:{self.id}, version:{version} has no weight")
            return
        model_weight_path = str(Path(model_weight_path).with_suffix('')).replace("\\", "/") # 확장자 제거
        
        labelsets = self.labelset_argment_handling(labelsets)
        
        for labelset in labelsets:
            # labelset handling step
            labelset_config = LabelsetDB2().get_record(labelset).config
            
            save_path = self.get_infer_result_path(version, labelset)
            if check_result and Path(save_path).exists():
                print(f"(name:{self.name}, version:{version}, labelset:{labelset}) already infered")
                return

            options = {
                "Global.pretrained_model":model_weight_path,
                "Global.checkpoints":model_weight_path,
                # "Global.save_model_dir":model_weight, 
                "Global.save_res_path":save_path, 
                
                "Infer.data_dir":labelset_config["dataset_dir"],
                "Infer.infer_file_list":[labelset_config["infer"]],
            }
            self.make_and_write_command_context("infer", code, train_config, options, verbose)
        
    def make_and_write_command_context(self, task, code, config, options, verbose):
        command = f"python {code} -c {config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}" # train에 대해서도 할 수 있게 수정해야 함
        if verbose:
            print(command)
        
        self.command_file_writer.write(task, command)
        
        
    def get_best_epoch(self, metric=None, df=None):
        """
            검증 데이터에서 가장 좋은 성능을 보인 epoch 반환
            df: 원래는 직접 로드해야 하는데, 반복 수행시 새로 로드하는게 비효율적이라서 인자로 받는 기능 추가해줌
        """

        labelset = self.get_label_set("eval")
        
            
        if metric is None:
            config = self.config
            metric = config["metric"]
        
        df = df if df is not None else self.eval_result_df

        if len(df) == 0:
            return None


        df = df.reset_index()
        
            
        if metric not in df.columns:
            return None
        
        version = df.loc[df[metric].idxmax()].version
                
        return int(version)
    
    @property
    def all_metrics(self):
        """
        현재 평가된 데이터 즉 "eval_result_df"에서 모든 평가 지표를 탐색하여 반환
        """
        df = self.result_df
        columns = df.columns
        columns = [c for c in columns if c not in ["work_id", "version", "fps", "labelset"]]
        return columns
    

    
    @property
    def main_indicator_cadidates(self):
        """
            전체 성능 지표 중 실제 성능을 담당하는 지표만 반환
            예로 자소 단위 성능 지표는 제외 (CNED)
            정확히 단어 전체를 맞췄는지 평가하는 지표만 반환 (ACC)
        """
        return [metric for metric in self.all_metrics if "acc" in metric and "ideal" not in metric]    
    
    @property
    def main_indicator(self):
        df = self.eval_result_df
        df = self.transform_result_df(df)
        if len(df) == 0:
            return None, None, None
        main_indicator_cadidates = self.main_indicator_cadidates
        df = df[df["metric"].apply(lambda x: x in main_indicator_cadidates)]
        
        
        df.reset_index(inplace = True)
        
        df = df.loc[df["value"].idxmax()]
    
        return df["metric"], int(df["version"]), df["value"]
    
    
    def performance_per_indicator(self, indicator):
        try:
            df = self.eval_result_df
            df = self.transform_result_df(df)            

            sub_df = df[df["metric"] == indicator]

            sub_df.reset_index(inplace = True)
            sub_df = sub_df.loc[sub_df["value"].idxmax()]
            
            return {
                "version":int(sub_df["version"]),
                "performance":sub_df["value"]
            }
        except Exception as e:
            print(e)
            return {"version":None, "performance":None}
        
    def performance_per_indicators(self, indicators):
        df = self.eval_result_df
        df = self.transform_result_df(df)
        if len(df) == 0:
            return None
        data = {
            "metric":[],
            "version":[],
            "performance":[]
        }
        for indicator in indicators:
            data["metric"].append(indicator)
            sub_df = df[df["metric"] == indicator]
            sub_df.reset_index(inplace = True)
            sub_df = sub_df.loc[sub_df["value"].idxmax()]
            data["version"].append(int(sub_df["version"]))
            data["performance"].append(sub_df["value"])
        
        df = pd.DataFrame(data)
        df.sort_values("performance", ascending=False, inplace=True)
        return df
    
    @property
    def performance_per_all_indicators(self):
        return self.performance_per_indicators(self.all_metrics)

    
    @property
    def performance_per_main_indicator_cadidates(self):
        return self.performance_per_indicators(self.main_indicator_cadidates)
        
    def get_best_epoch_for_all_metric(self):
        """
            검증 데이터셋에 대해, 모든 metric에 대해 가장 좋은 성능을 보인 epoch 리스트 반환
            즉 서로 다른 평가 지표에 대해 최대 epoch이 서로 다르다면 둘 다 반환
        """
        df = self.eval_result_df
        
        
        all_metrics = self.all_metrics
        best_epoch_list = []
        
        for metric in all_metrics:
            best_epoch = self.get_best_epoch(metric = metric, df = df)
            if best_epoch is not None:
                best_epoch_list.append(int(best_epoch))
        best_epoch_list = list(set(best_epoch_list))
        best_epoch_list.sort() 
        return best_epoch_list
    
    
    
    def report_result(self, report):
        # 기존 데이터 로드

        with open("/home/works/eval_lack", 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX) 
            try:
                df = self.result_df
                new_df = pd.DataFrame([report])
                new_df = pd.concat([df, new_df])
                self.result_df = new_df
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                
    def remove_result_df(self):
        Path(self.get_path("eval_result", level = "record")).unlink()
        
    
    def get_train_labelset_info(self):
        config = self.config
        labelset_id = config["labelsets"]["train"]
        labelset = LabelsetDB2().get_record(labelset_id)
        labelset_config = labelset.config
        
        return labelset_config["dataset_dir"], labelset_config["label"]
    
    def get_eval_labelset_info(self):
        config = self.config
        labelset_id = config["labelsets"]["eval"]
        labelset = LabelsetDB2().get_record(labelset_id)
        labelset_config = labelset.config
        
        return labelset_config["dataset_dir"], labelset_config["label"]
    
    
    def train(self, start_version="last", command_to="global"):
        epoch = self.get_config_epoch()
        
        trained_epoch = self.trained_epoch
        pass_flag = epoch <= trained_epoch
            
        print(f"""Trained ({trained_epoch:3d}/{epoch:3d}) | {self.name} \t{"(Passed)"if pass_flag else ""}""")
        
        if pass_flag:
            return False

        model_weight_path = self.get_model_weight_path(version = start_version, validate=True)
        if model_weight_path is not None:
            model_weight_path = str(Path(model_weight_path).with_suffix('')).replace("\\", "/") # 확장자 제거

        
        config = self.config
        train_config = config["train_config"]
        train_data_dir, train_labelset_file_path = self.get_train_labelset_info()
        eval_data_dir, eval_labelset_file_path = self.get_eval_labelset_info()

        code = self.get_command_code("train")
        
        
        options = {
                   "Global.checkpoints":model_weight_path,
                   "Global.epoch_num":epoch,
                   "Global.save_model_dir":config["trained_model_dir"],
                   
                   "Train.dataset.data_dir": train_data_dir,
                   "Train.dataset.label_file_list":[train_labelset_file_path],
                #    "Train.dataset.cache_file": config["train_cache_file"],
                   
                   "Eval.dataset.data_dir": eval_data_dir,
                   "Eval.dataset.label_file_list":[eval_labelset_file_path],
                #    "Eval.dataset.cache_file": config["eval_cache_file"]
                   }
        
        command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
        # print(command)
        
        save_path = self.get_path("train_bash", level = "db") # record끼리 명령어 파일 공유
        # save_path = self.save_relative_to(id, "train.sh", relative_to=relative_to, save_to=command_to)
        with open(save_path, "a") as f:
            f.write(command+"\n")
        return True
    
    @property
    def train_config_path(self):
        return self.config["train_config"]
    
    @property
    def train_config(self):        
        return self.load_yaml(self.train_config_path)
    
    @train_config.setter
    def train_config(self, config):
        self.save_yaml(self.train_config_path, config)
    
    
    def get_config_epoch(self):
        config = self.train_config
        return config["Global"]["epoch_num"]
    

    @property
    def target_epoch(self):
        if not hasattr(self, "__target_epoch"):
            config = self.train_config
            self.__target_epoch = config["Global"]["epoch_num"]
        return self.__target_epoch

    def modify_train_config(self, key, value):
        config = self.train_config
        c = config
        keys = key.split(".")
        for k in keys[:-1]:
            c = c[k]
        c[keys[-1]] = value
        self.train_config = config

    # @property
    # def ensemble_main_indicators(self):
    #     indicators = self.main_indicator_cadidates
    #     return [indicator for indicator in indicators if "acc" in indicator]
        