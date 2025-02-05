import yaml
import random
from pathlib import Path
from .dataset_db import DatasetDB
from .db import DB, DB2, Record2
import project
import pandas as pd

def split_list(data, ratio, size=None):
    total_num = len(data)
    if isinstance(size, int):
        total_num = min(total_num, size)
    total_ratio = sum(ratio)
    num_list = [int(total_num*(r/total_ratio)) for r in ratio]
    result = []
    acc = 0
    for num in num_list[:-1]:
        result.append(data[acc:acc+num])
        acc += num
    result.append(data[acc:total_num])
    return result
    
class LabelsetDB(DB):
    DIR = "./labelsets"
    CONFIG_NAME = "config.yml"
    
    TRAIN_LABEL_FILE = "train_label.txt"
    EVAL_LABEL_FILE = "eval_label.txt"
    TEST_LABEL_FILE = "test_label.txt"
    TRAIN_INFER_FILE = "train_infer.txt"
    EVAL_INFER_FILE = "eval_infer.txt"
    TEST_INFER_FILE = "test_infer.txt"
    
    def __init__(self):
        super().__init__(LabelsetDB.DIR, LabelsetDB.CONFIG_NAME)
    
    def make(self, name, datasets, split_ratio=[8, 1, 1], individual_split_ratio={}, shuffle=True, random_seed=100, size=None):
        """_summary_
            make label set from the datasets with split_ratio option
        Args:
            name (_type_): _description_
            datasets (_type_): split_ratio와 함께 쓰이며, 사용할 전체 데이터 셋을 명시 ["dataset1", "dataset2", ...]
            split_ratio (_type_): 전체 데이터 셋을 [train, val, test]로 나눌 비율 [8, 1, 1]
            individual_split_ratio (_dict_): split_ratio에 세부적으로 특정 데이터 셋에 대해 split_ratio 지정 가능 {"dataset":[8, 1, 1], ...}
            shuffle (bool): whether suffle the dataset before spliting
            random_seed (int, optional): split 수행 시 사용할 random seed. Defaults to 100.
            
        """
        # datasets에 원하는 데이터 셋 id를 모두 나열 하고 split_ratio를 지정하면 전체를 해당 비율로 나누어 레이블 셋 구성
        # 
        
        assert name not in self.get_all_id(), f"The name '{name}' already exists!"
        
        assert split_ratio or individual_split_ratio, f"input datasets or individual_split_ratio"
        
        
        
        # 데이터 셋에 대한 train, val, test 레이블 셋 구성
        datasetDB = DatasetDB()
        assert len(set(sum([datasetDB.get_config(id)["task"] for id in datasets], []))) <= 1  ##### 미완성
        
        random.seed(random_seed)        
        
        # 각 데이터 셋 마다 split ratio 계산
        split_ratio = {dataset:split_ratio for dataset in datasets}
        split_ratio.update(individual_split_ratio)
        
        labelsets = [[], [], []] # 전체 레이블 셋 [train, val, test]
        
        labels_list = [datasetDB.get_all_labels(dataset, relative_to="dir") for dataset in split_ratio.keys()]
        size_list = [len(labels) for labels in labels_list]
        if size:
            total_num = sum(size_list)
            size_list = [int(num*size/total_num) for num in size_list]
        
        
        for i, dataset in enumerate(split_ratio.keys()):
            labels = labels_list[i]
            size = size_list[i]
            random.shuffle(labels) if shuffle else ""
            print(dataset, split_ratio[dataset])
            for whole, patial in zip(labelsets, split_list(labels, split_ratio[dataset], size = size)): # dataset을 지정된 비율에 따라 [train, val, test]로 split 한 뒤 더함 
                whole+=patial
                 
        print("2. save label files")
        train_labels, val_labels, test_labels = labelsets
        save_dir = Path(self.PROJECT_ROOT)/self.DIR/name
        (save_dir).mkdir(parents=True, exist_ok=True)
        open(save_dir/LabelsetDB.TRAIN_LABEL_FILE, "w").write("\n".join(train_labels))
        open(save_dir/LabelsetDB.EVAL_LABEL_FILE, "w").write("\n".join(val_labels))
        open(save_dir/LabelsetDB.TEST_LABEL_FILE, "w").write("\n".join(test_labels))
        
        # 전체 레이블을 나누어 레이블 파일로 저장
        print("3. make infer label files")
        train_infer = [label.split("\t")[0] for label in train_labels]
        val_infer = [label.split("\t")[0] for label in val_labels]
        test_infer = [label.split("\t")[0] for label in test_labels]
        open(save_dir/LabelsetDB.TRAIN_INFER_FILE, "w").write("\n".join(train_infer))
        open(save_dir/LabelsetDB.EVAL_INFER_FILE, "w").write("\n".join(val_infer))
        open(save_dir/LabelsetDB.TEST_INFER_FILE, "w").write("\n".join(test_infer))
            
        # config 생성 및 저장
        config = {}
        config["datasets"]=split_ratio
        config["task"] = set(sum([datasetDB.get_config(id)["task"] for id in datasets], [])) ##### 미완성
        config["label"]={
            "train":[LabelsetDB.TRAIN_LABEL_FILE],
            "eval":[LabelsetDB.EVAL_LABEL_FILE],
            "test":[LabelsetDB.TEST_LABEL_FILE],
        }
        config["infer"]={
            "train":[LabelsetDB.TRAIN_INFER_FILE],
            "eval":[LabelsetDB.EVAL_INFER_FILE],
            "test":[LabelsetDB.TEST_INFER_FILE],
        }
        config["seed"] = random_seed
        config["dataset_dir"] = DatasetDB.DIR
        with open(save_dir/LabelsetDB.CONFIG_NAME, "w") as f:
                yaml.dump(config, f)   
        
    
    def get_config(self, id, relative_to=None):
        config = super().get_config(id)
        for category in ["label", "infer"]:
            for work in ["train", "eval", "test"]:
                if config[category][work] and relative_to: # 지정된 상대 경로로 변환
                    config[category][work] = [self.relative_to(id, label, relative_to=relative_to) for label in config[category][work]]
                elif "origin_labelset" in config:     # 없는 경우 origin_labelset에서 가져옴   
                    config[category][work] = sum([self.get_config(origin_labelset, relative_to=relative_to)["label"][work] for origin_labelset in config["origin_labelset"]], [])
        return config
    
    def get_label_file_path(self, id, task, relative_to=None, file_name=None):
        if file_name:
            return Path(self.PROJECT_ROOT)/self.DIR/id/file_name
        if task == "train":
            return Path(self.PROJECT_ROOT)/self.DIR/id/LabelsetDB.TRAIN_LABEL_FILE
        elif task == "eval":
            return Path(self.PROJECT_ROOT)/self.DIR/id/LabelsetDB.EVAL_LABEL_FILE
        elif task == "test":
            return Path(self.PROJECT_ROOT)/self.DIR/id/LabelsetDB.TEST_LABEL_FILE
        else:
            raise ValueError(f"task should be one of ['train', 'eval', 'test']")
            
    
    def get_label(self, id, task, relative_to=None, label_file_path = None):
        if label_file_path == None:
            label_file_path = self.get_label_file_path(id, task, relative_to)
            
        with open(label_file_path) as f:
            return [line.strip().split("\t") for line in f.readlines()]

    def get_label_df(self, id, task, relative_to=None, label_file_path=None):
        return pd.DataFrame(self.get_label(id, task, relative_to, label_file_path=label_file_path), columns=["image", "label"])

    
    def make_k_fold(self, labelsets, name, k, random_seed=100):
        
        for labelset in labelsets:
            assert labelset in self.get_all_id(), f"The labelset {labelset} does not exists!"

        new_names = [f"{name}_{k}_{i+1}" for i in range(k)]
        for new_name in new_names:
            assert new_name not in self.get_all_id(), f"The labelset {labelset} already exists!"     
        
        print("1. Load all label files")

        whole_labels = []
        for labelset in labelsets:
            config = self.get_config(labelset, relative_to="absolute")
            label_files = config["label"]["train"] + config["label"]["eval"]
            # label_files = [str(Path(self.ROOT/config["name"]for label_file in label_files]
            labels = sum([[line.strip("\n") for line in open(label_file).readlines()] for label_file in label_files], [])
            whole_labels += labels

        # 전체 레이블을 나누어 레이블 파일로 저장
        print("2. split labels in k segments")
        segments = []
        s_size = int(len(labels)/k) # segment_size
        for i in range(k-1):
            segments.append(labels[s_size*i:s_size*(i+1)])
        segments.append(labels[s_size*(k-1):])
        
        print("3. save")
        for i in range(k):
            train = sum([[] if i==j else segment for j, segment in enumerate(segments)], [])
            val = segments[i]
            new_name = new_names[i]
            new_label_dir = Path(self.DIR)/new_name
            new_label_dir.mkdir(parents=True, exist_ok=True)
            open(new_label_dir/LabelsetDB.TRAIN_LABEL_FILE, "w").write("\n".join(train))
            open(new_label_dir/LabelsetDB.EVAL_LABEL_FILE, "w").write("\n".join(val))
            print(new_label_dir/LabelsetDB.TRAIN_LABEL_FILE)
            
            train_infer = [label.split("\t")[0] for label in train]
            val_infer = [label.split("\t")[0] for label in val]
            open(new_label_dir/LabelsetDB.TRAIN_INFER_FILE, "w").write("\n".join(train_infer))
            open(new_label_dir/LabelsetDB.EVAL_INFER_FILE, "w").write("\n".join(val_infer))
            
            config = {}
            config["name"] = new_name
            config["dataset_dir"] = self.DIR
            config["origin_labelset"] = labelsets
            config["k_fold"] = {"k":k, "i":i+1}
            config["label"]={
                "train":[LabelsetDB.TRAIN_LABEL_FILE],
                "eval":[LabelsetDB.EVAL_LABEL_FILE],
                "test":[],
            }
            config["infer"]={
                "train":[LabelsetDB.TRAIN_INFER_FILE],
                "eval":[LabelsetDB.EVAL_INFER_FILE],
                "test":[],
            }
            config["seed"] = random_seed
            with open(new_label_dir/LabelsetDB.CONFIG_NAME, "w") as f:
                yaml.dump(config, f)



# if __name__=="__main__":
#     labeldb = LabelsetDB()
#     labeldb.make("ai_hub_rec__40_10_50", ["korean_image_rec"], 40, 10, 50, random_seed=100)
#     # name = LabelsetDB3().get_name_list()[0]
#     LabelsetDB().make_k_fold("ai_hub_rec__40_10_50", "ai_hub_rec__40_10_50_random_k_fold", 5)

#     # print(LabelsetDB().get_name_list())
#     print(LabelsetDB().get("ai_hub_det_08_02_90_random_k_fold_5_1"))
    

class LabelsetDB2(DB2):

    def __init__(self, name = "labelsets"):
        super().__init__(name = name, record_class = Labelset2, record_config_name = "config.yml")
        

    TRAIN_LABEL_FILE = "train_label.txt"
    EVAL_LABEL_FILE = "eval_label.txt"
    TEST_LABEL_FILE = "test_label.txt"
    TRAIN_INFER_FILE = "train_infer.txt"
    EVAL_INFER_FILE = "eval_infer.txt"
    TEST_INFER_FILE = "test_infer.txt"
    
    def make(self, name, datasets, split_ratio=[8, 1, 1], individual_split_ratio={}, shuffle=True, random_seed=100, size=None):
        """_summary_
            make label set from the datasets with split_ratio option
        Args:
            name (_type_): _description_
            datasets (_type_): split_ratio와 함께 쓰이며, 사용할 전체 데이터 셋을 명시 ["dataset1", "dataset2", ...]
            split_ratio (_type_): 전체 데이터 셋을 [train, val, test]로 나눌 비율 [8, 1, 1]
            individual_split_ratio (_dict_): split_ratio에 세부적으로 특정 데이터 셋에 대해 split_ratio 지정 가능 {"dataset":[8, 1, 1], ...}
            shuffle (bool): whether suffle the dataset before spliting
            random_seed (int, optional): split 수행 시 사용할 random seed. Defaults to 100.
            
        """
        # datasets에 원하는 데이터 셋 id를 모두 나열 하고 split_ratio를 지정하면 전체를 해당 비율로 나누어 레이블 셋 구성
        # 
        
        assert name not in self.get_all_id(), f"The name '{name}' already exists!"
        
        assert split_ratio or individual_split_ratio, f"input datasets or individual_split_ratio"
        
        
        
        # 데이터 셋에 대한 train, val, test 레이블 셋 구성
        datasetDB = DatasetDB()
        assert len(set(sum([datasetDB.get_config(id)["task"] for id in datasets], []))) <= 1  ##### 미완성
        
        random.seed(random_seed)        
        
        # 각 데이터 셋 마다 split ratio 계산
        split_ratio = {dataset:split_ratio for dataset in datasets}
        split_ratio.update(individual_split_ratio)
        
        labelsets = [[], [], []] # 전체 레이블 셋 [train, val, test]
        
        labels_list = [datasetDB.get_all_labels(dataset, relative_to="dir") for dataset in split_ratio.keys()]
        size_list = [len(labels) for labels in labels_list]
        if size:
            total_num = sum(size_list)
            size_list = [int(num*size/total_num) for num in size_list]
        
        
        for i, dataset in enumerate(split_ratio.keys()):
            labels = labels_list[i]
            size = size_list[i]
            random.shuffle(labels) if shuffle else ""
            print(dataset, split_ratio[dataset])
            for whole, patial in zip(labelsets, split_list(labels, split_ratio[dataset], size = size)): # dataset을 지정된 비율에 따라 [train, val, test]로 split 한 뒤 더함 
                whole+=patial
                 
        print("2. save label files")
        train_labels, val_labels, test_labels = labelsets
        save_dir = Path(self.PROJECT_ROOT)/self.DIR/name
        (save_dir).mkdir(parents=True, exist_ok=True)
        open(save_dir/LabelsetDB.TRAIN_LABEL_FILE, "w").write("\n".join(train_labels))
        open(save_dir/LabelsetDB.EVAL_LABEL_FILE, "w").write("\n".join(val_labels))
        open(save_dir/LabelsetDB.TEST_LABEL_FILE, "w").write("\n".join(test_labels))
        
        # 전체 레이블을 나누어 레이블 파일로 저장
        print("3. make infer label files")
        train_infer = [label.split("\t")[0] for label in train_labels]
        val_infer = [label.split("\t")[0] for label in val_labels]
        test_infer = [label.split("\t")[0] for label in test_labels]
        open(save_dir/LabelsetDB.TRAIN_INFER_FILE, "w").write("\n".join(train_infer))
        open(save_dir/LabelsetDB.EVAL_INFER_FILE, "w").write("\n".join(val_infer))
        open(save_dir/LabelsetDB.TEST_INFER_FILE, "w").write("\n".join(test_infer))
            
        # config 생성 및 저장
        config = {}
        config["datasets"]=split_ratio
        config["task"] = set(sum([datasetDB.get_config(id)["task"] for id in datasets], [])) ##### 미완성
        config["label"]={
            "train":[LabelsetDB.TRAIN_LABEL_FILE],
            "eval":[LabelsetDB.EVAL_LABEL_FILE],
            "test":[LabelsetDB.TEST_LABEL_FILE],
        }
        config["infer"]={
            "train":[LabelsetDB.TRAIN_INFER_FILE],
            "eval":[LabelsetDB.EVAL_INFER_FILE],
            "test":[LabelsetDB.TEST_INFER_FILE],
        }
        config["seed"] = random_seed
        config["dataset_dir"] = DatasetDB.DIR
        with open(save_dir/LabelsetDB.CONFIG_NAME, "w") as f:
                yaml.dump(config, f)   
    def make_k_fold(self, labelsets, name, k, random_seed=100):
        
        for labelset in labelsets:
            assert labelset in self.get_all_id(), f"The labelset {labelset} does not exists!"

        new_names = [f"{name}_{k}_{i+1}" for i in range(k)]
        for new_name in new_names:
            assert new_name not in self.get_all_id(), f"The labelset {labelset} already exists!"     
        
        print("1. Load all label files")

        whole_labels = []
        for labelset in labelsets:
            config = self.get_config(labelset, relative_to="absolute")
            label_files = config["label"]["train"] + config["label"]["eval"]
            # label_files = [str(Path(self.ROOT/config["name"]for label_file in label_files]
            labels = sum([[line.strip("\n") for line in open(label_file).readlines()] for label_file in label_files], [])
            whole_labels += labels

        # 전체 레이블을 나누어 레이블 파일로 저장
        print("2. split labels in k segments")
        segments = []
        s_size = int(len(labels)/k) # segment_size
        for i in range(k-1):
            segments.append(labels[s_size*i:s_size*(i+1)])
        segments.append(labels[s_size*(k-1):])
        
        print("3. save")
        for i in range(k):
            train = sum([[] if i==j else segment for j, segment in enumerate(segments)], [])
            val = segments[i]
            new_name = new_names[i]
            new_label_dir = Path(self.DIR)/new_name
            new_label_dir.mkdir(parents=True, exist_ok=True)
            open(new_label_dir/LabelsetDB.TRAIN_LABEL_FILE, "w").write("\n".join(train))
            open(new_label_dir/LabelsetDB.EVAL_LABEL_FILE, "w").write("\n".join(val))
            print(new_label_dir/LabelsetDB.TRAIN_LABEL_FILE)
            
            train_infer = [label.split("\t")[0] for label in train]
            val_infer = [label.split("\t")[0] for label in val]
            open(new_label_dir/LabelsetDB.TRAIN_INFER_FILE, "w").write("\n".join(train_infer))
            open(new_label_dir/LabelsetDB.EVAL_INFER_FILE, "w").write("\n".join(val_infer))
            
            config = {}
            config["name"] = new_name
            config["dataset_dir"] = self.DIR
            config["origin_labelset"] = labelsets
            config["k_fold"] = {"k":k, "i":i+1}
            config["label"]={
                "train":[LabelsetDB.TRAIN_LABEL_FILE],
                "eval":[LabelsetDB.EVAL_LABEL_FILE],
                "test":[],
            }
            config["infer"]={
                "train":[LabelsetDB.TRAIN_INFER_FILE],
                "eval":[LabelsetDB.EVAL_INFER_FILE],
                "test":[],
            }
            config["seed"] = random_seed
            with open(new_label_dir/LabelsetDB.CONFIG_NAME, "w") as f:
                yaml.dump(config, f)
                
                
    # def get_record(self, record_name):
        
    #     if isinstance(record_name, int): # is id
    #         print(self.name_to_id)
    #         record_name = self.name_to_id[record_name]
        
            
    #     return LabelsetRecord2(self, record_name)
    
    def get_label(self, label_file_path):
        with open(label_file_path) as f:
            return [line.strip().split("\t") for line in f.readlines()]                  

    def get_record_table(self):
        # db의 항목들을 DataFrame으로 변환하여 보여줌        
        record_dict = {column: [] for column in ["id", "name", "size"]}
        for id, labelset in self.records.items():
            record_dict["id"].append(labelset.id)
            record_dict["name"].append(labelset.name)
            record_dict["size"].append(labelset.size)
            
        df =  pd.DataFrame(record_dict, index=None)
        return df.reset_index(drop=True)

    def get_all_records(self):
        names = self.get_all_id()
        records = [self.get_record(name) for name in names]
        return records



    def validate_and_update(self):
        self.validate_id_uniqueness()
        for id, record in self.records.items():
            record.validate_and_update()

class Labelset2(Record2):


    def __init__(self, db, record_id):
        super().__init__(db, record_id, ["dataset_dir", "infer", "label"])
        
        # self.path_dir = {
        #     "label_file_path": self.record_dir/"eval.sh",
        #     "infer_file_path": self.record_dir/"infer.sh",
        # }    
    
    
    @property
    def size(self):
        return self.config["size"]

    
    def get_label_file_path(self, file_name=None):
        path = self.config["label"]
        return self.relative_to(path, current_relative="record", target_relative=self.RELATIVE_TO)

    def get_infer_file_path(self, file_name=None):
        path = self.config["infer"]            
        return self.relative_to(path, current_relative="record", target_relative=self.RELATIVE_TO)
        
    @property
    def labels(self):
        if not hasattr(self, "__labels"):
            label_file_path = self.get_label_file_path()
            self.__labels = self.db.get_label(label_file_path)
        return self.__labels
        
    # def get_label(self):
    #     label_file_path = self.get_label_file_path(relative_to="absolute")
    #     return self.db.get_label(label_file_path)



    def validate_size_in_config(self, config):
        def validate_existing(config):
            """ config에 size 값이 존재하는지 확인후 없으면 경고와 함께 추가"""
            if "size" not in config:
                print("warnning: size is not in config, it was changed to 0")
                config["size"] = 0
            return config

        def validate_value(config):
            """ 기록된 사이즈가 실제 사이즈와 일치하는지 체크하고, 다르다면 경고와 함께 실제 값으로 수정"""
            size_in_config = config["size"]
            real_size = len(self.labels)
            
            if size_in_config != real_size:
                print(f"warnning: size in config is {size_in_config}, but real size is {real_size}, it was changed to {real_size}")
                config["size"] = real_size
            return config
        
        config = validate_existing(config)
        config = validate_value(config)
            
    def validate_and_update(self):
        config = self.config
        self.validate_size_in_config(config)    
        self.update_config(config)
        print(f"{self.id}:{self.name} is validated")
        
    def check_relation(self):
        pass
    
    