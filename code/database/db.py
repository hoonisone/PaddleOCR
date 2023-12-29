import project

import yaml
from pathlib import Path
class DB:
    def __init__(self, type):
        assert type in ["dataset_db", "labelset_db", "model_db", "work_db"]
        
        self.root = project.config["db"][type]["root_dir"]
        self.config_name = project.config["db"][type]["config_name"]
    
        
    def get_all_id(self):
        path_list = Path(self.root).glob("*")
        element_list = []
        for path in path_list:
            if path.is_dir():
                element_list.append(path)
        return element_list
        

    def get_path(self, id):
        return self.root/id/self.config_name
    
    def get_value(self, id):
        path = self.get_path(id)
        with open(path) as f:
            value = yaml.load(f, Loader=yaml.FullLoader)
        return self.insert_root(value)

    def insert_root(self, config):
        # config 안에 문자열 중 "{root}" 가 있으면 이를 실제 root로 대체해줌
        for k, v in config.items():
            if isinstance(v, str):
                config[k] = v.replace("{root}", self.root)
            if isinstance(v, dict):
                config[k] = self.insert_root(v)
        return config