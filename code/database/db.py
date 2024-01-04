import project

import yaml
from pathlib import Path
class DB:
    def __init__(self, root, config_name):
        self.root = root
        self.config_name = config_name

        
    def get_all_id(self):
        path_list = Path(self.root).glob("*")
        element_list = []
        for path in path_list:
            if path.is_dir() and self.is_target(path):
                element_list.append(path.name)
        return element_list
        
    def is_target(self, path):
        return len(list(path.glob(self.config_name))) == 1
        
    def get_path(self, id):
        return str(Path(self.root)/id/self.config_name)
    
    def get_config(self, id):
        path = self.get_path(id)
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if config == None:
            config = {}
        config["name"] = id
        return config
    
    def update_config(self, id, config):
        path = self.get_path(id)
        with open(path, "w") as f:
            yaml.dump(config, f)
    
    def get_root(self):
        return self.root

    def insert_root(self, config):
        # config 안에 문자열 중 "{root}" 가 있으면 이를 실제 root로 대체해줌
        for k, v in config.items():
            if isinstance(v, str):
                config[k] = v.replace("{root}", self.root)
            if isinstance(v, dict):
                config[k] = self.insert_root(v)
        return config