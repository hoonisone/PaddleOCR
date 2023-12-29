import yaml
import random
from pathlib import Path
from datasets import DatasetDB

class LabelsetDB:
    ROOT = Path("/home/labelsets")
    CONFIG_FILE = "config.yml"
    TRAIN_LABEL_FILE = "train_label.txt"
    EVAL_LABEL_FILE = "eval_label.txt"
    TEST_LABEL_FILE = "test_label.txt"
    INFER_LABEL_FILE = "infer_label.txt"
    
    def __init__(self, root=None):
        self.root = root if root else LabelsetDB.ROOT
        self.name_path = LabelsetDB.get_name_path_list(self.root)        

    @staticmethod    
    def get_path_list(root):
        path_list = root.glob("*")

        element_list = []
        for path in path_list:
            if LabelsetDB.is_target(path):
                element_list.append(path)
            if path.is_dir():
                element_list += DatasetDB.get_path_list(path)
            else:
                pass
        return element_list
            
    @staticmethod
    def get_name_path_list(root):  
        name_path = {}
        for path in DatasetDB.get_path_list(root):
            with open(str(path/LabelsetDB.CONFIG_FILE)) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            for work in ["train", "eval", "test", "infer"]:                
                if config["label"][work]:
                    config["label"][work] = [path/x for x in config["label"][work]]
            
            key = config["name"]
            value = config
            name_path[key] = value
        
        return name_path
            
    @staticmethod
    def is_target(path):
        path_list = path.glob("*")
        return any([path.name == LabelsetDB.CONFIG_FILE for path in path_list])
    
    def get_name_list(self):
        return list(self.name_path.keys())
    
    def get(self, name):
        config = self.name_path[name]
        for work in ["train", "eval", "test", "infer"]:
            if (not config["label"][work]) and config["origin_labelset"]:
                config["label"][work] = self.get(config["origin_labelset"])["label"][work]                    
        return config
    
    def make(self, name, datasets, train_ratio, eval_ratio, test_ratio, random_seed=100):
        assert name not in self.get_name_list(), f"The name '{name}' already exists!"
        
        random.seed(random_seed)
        datasetDB = DatasetDB()
        label_file_path_list = sum([datasetDB.get_label_file_path(dataset) for dataset in datasets], [])

        label_list = []
        for label_file_path in label_file_path_list:
            labels = open(label_file_path).readlines()
            labels = [label.strip("\n").split("\t") for label in labels]
            labels = [f"{str((label_file_path.absolute().parent/path).relative_to(datasetDB.root))}\t{label}" for path, label in labels]
            label_list += labels
            random.shuffle(label_list)

        # 전체 레이블을 나누어 레이블 파일로 저장
        print("2. split and save label files")

        total = sum([train_ratio, eval_ratio, test_ratio])
        train_n = int(train_ratio/total*len(label_list))
        eval_n = int(eval_ratio/total*len(label_list))
        test_n = int(test_ratio/total*len(label_list))

        train_labels = label_list[:train_n]
        val_labels = label_list[train_n:-test_n]
        test_labels = label_list[-test_n:]
        infer_list = [label.split("\t")[0] for label in test_labels]

        (self.root/name).mkdir(parents=True, exist_ok=True)
        print(self.root/name)
        open(self.root/name/LabelsetDB.TRAIN_LABEL_FILE, "w").write("\n".join(train_labels))
        open(self.root/name/LabelsetDB.EVAL_LABEL_FILE, "w").write("\n".join(val_labels))
        open(self.root/name/LabelsetDB.TEST_LABEL_FILE, "w").write("\n".join(test_labels))
        open(self.root/name/LabelsetDB.INFER_LABEL_FILE, "w").write("\n".join(infer_list))
            
        # config 생성 및 저장
        config = {}
        config["name"]=name
        config["datasets"]=datasets
        config["label_files"]=[str(path) for path in label_file_path_list]
        config["split"] = {
            "train_ratio":train_ratio,
            "eval_ratio":eval_ratio,
            "test_ratio":test_ratio  
        }
        config["label"]={
            "train":[LabelsetDB.TRAIN_LABEL_FILE],
            "eval":[LabelsetDB.EVAL_LABEL_FILE],
            "test":[LabelsetDB.TEST_LABEL_FILE],
            "infer":[LabelsetDB.INFER_LABEL_FILE]
        }
        config["seed"] = random_seed
        with open(self.root/name/LabelsetDB.CONFIG_FILE, "w") as f:
                yaml.dump(config, f)   
                
                
    def make_k_fold(self, dataset, name, k, random_seed=100):
        assert dataset in self.get_name_list(), f"The dataset {dataset} does not exists!"
        label_root = Path("/home/dataset_labels")

        print("1. Load all label files")
        labelset = self.get(dataset)
        label_files = labelset["label"]["train"] + labelset["label"]["eval"]
        
        labels = sum([[line.strip("\n") for line in open(label_file).readlines()] for label_file in label_files], [])

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
            new_name = f"{name}_{k}_{i+1}"
            new_label_dir = self.root/new_name
            new_label_dir.mkdir(parents=True, exist_ok=True)
            open(new_label_dir/LabelsetDB.TRAIN_LABEL_FILE, "w").write("\n".join(train))
            open(new_label_dir/LabelsetDB.EVAL_LABEL_FILE, "w").write("\n".join(val))
            config = {}
            config["name"] = new_name
            config["origin_labelset"] = labelset['name']
            config["k_fold"] = {"k":k, "i":i+1}
            config["label"]={
                "train":[LabelsetDB.TRAIN_LABEL_FILE],
                "eval":[LabelsetDB.EVAL_LABEL_FILE],
                "test":[],
                "infer":[]
            }
            config["seed"] = random_seed
            with open(new_label_dir/LabelsetDB.CONFIG_FILE, "w") as f:
                yaml.dump(config, f)

if __name__=="__main__":
    labeldb = LabelsetDB()
    labeldb.make("ai_hub_rec__40_10_50", ["korean_image_rec"], 40, 10, 50, random_seed=100)
    # name = LabelsetDB3().get_name_list()[0]
    LabelsetDB().make_k_fold("ai_hub_rec__40_10_50", "ai_hub_rec__40_10_50_random_k_fold", 5)

    # print(LabelsetDB().get_name_list())
    print(LabelsetDB().get("ai_hub_det_08_02_90_random_k_fold_5_1"))
    