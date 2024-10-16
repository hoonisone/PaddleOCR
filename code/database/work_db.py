from pathlib import Path
from matplotlib.dates import get_epoch
import yaml


from .labelset_db import LabelsetDB
from .model_db import ModelDB
from .db import DB
import copy
import project
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import itertools
import fcntl


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
            new_df = pd.DataFrame([{"id": id, "labelsets": config["labelsets"], "model": config["model"], "trained_epoch": self.get_max_epoch(id)}])
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
        
        train_config = f"{config['trained_model_dir']}/config.yml"
        checkpoint = self.get_model_weight(id, version=version, relative_to=relative_to)
        
        inference_model_path = self.make_inference_model_path(id, version, relative_to=relative_to)
        
        code = self.get_command_code(id, "export")
        options = {
            "Global.save_inference_dir":inference_model_path, # 저장될 infer model의 path (without extension)
            "Global.checkpoints":checkpoint
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
        epoch_list = self.get_all_epoch(id)
        return version in epoch_list
                
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
        assert (isinstance(version, int) and version > 0) or (isinstance(version, str) and (version in ["best", "latest", "origin"])), version
        
        config = self.get_config(id)
        if version == "best":
            path = self.relative_to(id, Path(config["trained_model_dir"])/f"best_model/model.{extension}", relative_to=relative_to)
        elif version == "latest":
            path = self.relative_to(id, Path(config["trained_model_dir"])/f"latest.{extension}", relative_to=relative_to)      
        elif version == "origin": # 처음 모델 가중치 그대로 (초기화 또는 다른 테스크에서 pretrained)
            model_config = ModelDB().get_config(config["model"], relative_to=relative_to)
            path = str(model_config["pretrained_model_weight"])
            path = path[:-8]+extension
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
    
    def command_split(self, type, mode, n):
        with open(f"/home/works/{type}.sh") as f:
            lines = f.readlines()
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
    
    def eval_all(self, id, task, relative_to="absolute", command_to="global", check_result = True, labelsets=None, data_dir=None, save = True):
        
        unevaluated_epoches =  self.get_unevaluated_epoches(id, task)
        
        code = self.get_command_code(id, "eval", relative_to=relative_to)
        config = self.get_config(id, relative_to="project")
        labelset_configs = [LabelsetDB().get_config(id, relative_to="project") for id in config["labelsets"]]
        
        if (data_dir == None) or (labelsets == None):
            data_dir = labelset_configs[0]["dataset_dir"]
            labelsets = sum([c["label"][task] for c in labelset_configs], [])
        
        for version in unevaluated_epoches:
            
            train_config = config["train_config"]
            model_weight = self.get_model_weight(id, version, relative_to=relative_to)
            model_weight = ".".join(model_weight.split(".")[:-1]) # 확장자 제거


            options = {
                    "Global.work_id":id,
                    "Global.version":version,
                    "Global.eval_task":task,
                    "Global.checkpoints":model_weight,
                    "Global.save_model_dir":config["trained_model_dir"],

                    "Eval.dataset.data_dir": data_dir,
                    "Eval.dataset.label_file_list":f"""['{"','".join(labelsets)}']""",
                    "Eval.save":save,
                    "Eval.check_exist":check_result,
                    "Eval.dataset.cache_file": config["eval_cache_file"]
                    }
            
            command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
            
            save_path = self.save_relative_to(id, "eval.sh", relative_to=relative_to, save_to=command_to)
            with open(save_path, "a") as f:
                f.write(command+"\n")    
            
    
    def eval_one(self, id, version, task, relative_to="project", command_to="global", report_to="local", check_result=True, check_weight=True, labelsets=None, data_dir=None, save = True):
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

                   "Eval.dataset.data_dir": data_dir,
                   "Eval.dataset.label_file_list":f"""['{"','".join(labelsets)}']""",
                   "Eval.save":save,
                   "Eval.check_exist":check_result,
                   "Eval.dataset.cache_file": config["eval_cache_file"]
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
                   "Train.dataset.cache_file": config["train_cache_file"],
                   
                   "Eval.dataset.data_dir": labelset_configs[0]["dataset_dir"],
                   "Eval.dataset.label_file_list":f"""['{"','".join(eval_labelsets)}']""",
                   "Eval.dataset.cache_file": config["eval_cache_file"]
                   }
        
        command = f"python {code} -c {train_config} -o {' '.join([f'{k}={v}' for k, v in options.items()])}"
        # print(command)
        
        save_path = self.save_relative_to(id, "train.sh", relative_to=relative_to, save_to=command_to)
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


    def get_best_epoch(self, id, criteria = "test", metric=None):
        if metric is None:
            config = self.get_config(id)
            metric = config["metric"]
        
        df = self.get_report_df(id)
        df.reset_index(inplace=True)
        df = df[df["task"] == criteria]
        df = df[df["version"] != "best"]
        return df.loc[df[metric].idxmax()].version
    
    def get_best_report(self, id, criteria = "test", metric="acc"):
        epoch = self.get_best_epoch(id, criteria, metric=metric)
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
    # mdb = WorkDB()
    # id = "rec___ABI_A___aihub_rec_10k_horizontal___C_ALL+IMF"
    # task = "test"
    # print(mdb.get_evaluated_epoches(id, "test"))
    
    # mdb.eval_all(None, id, task, relative_to="project", command_to="global", report_to="local", check_result=True, check_weight=True, labelsets=None, data_dir=None, save = True)
    
    # mdb.make("/home/configs/det/det_mv3_db.yml", "output1", "ai_hub_det_08_02_90_random_k_fold_5_1", "MobileNetV3_large_x0_5")/
    # mdb.make("/home/configs/rec/PP-OCRv3/multi_language/korean_PP-OCRv3_rec.yml", "output2", "ai_hub_rec_08_02_90", "korean_PP-OCRv3_rec")
    
    
    
    # models = mdb.get_name_list()
    # print(models)
    # print(models)
    # print(models[0])
    # print(mdb.get(models[0]))
    
    
    