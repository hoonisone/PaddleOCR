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
    
    def get_config(self, id, abs_path=True):
        path = self.get_path(id)
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if config == None:
            config = {}
        config["id"] = id
        if abs_path:
            config = self.replace(config, "{ROOT}", self.root)
            config = self.replace(config, "{ID}", id)
        return config
    
    def update_config(self, id, config):
        path = self.get_path(id)
        with open(path, "w") as f:
            yaml.dump(config, f)
    
    def get_root(self):
        return self.root

    def replace(self, config, target , replace):
        # config 안에 모든 값들에 대해 target을 replace로 대체해줌
        for k, v in config.items():
            if isinstance(v, str):
                config[k] = v.replace(target, replace)
            if isinstance(v, dict):
                config[k] = self.replace(v, target, replace)
        return config