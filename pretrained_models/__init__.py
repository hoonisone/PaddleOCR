from pathlib import Path
import yaml

class PretrainedModelDB:
    root = Path("/home/pretrained_models")
    CONFIG_FILE = "config.yml"
    def __init__(self, root=None):
        self.root = root if root else PretrainedModelDB.root
        self.name_path = PretrainedModelDB.get_name_path_list(self.root)
    
    @staticmethod    
    def get_path_list(root):
        path_list = root.glob("*")

        dataset_list = []
        for path in path_list:
            if PretrainedModelDB.is_target(path):
                dataset_list.append(path)
            elif path.is_dir():
                dataset_list += PretrainedModelDB.get_path_list(path)
            else:
                pass
        return dataset_list
    
    @staticmethod
    def get_name_path_list(root):  
        name_path = {}
        for path in PretrainedModelDB.get_path_list(root):
            with open(str(path/PretrainedModelDB.CONFIG_FILE)) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config["pretrained"] = path/config["pretrained"]

            key = config["name"]
            value = config
            name_path[key] = value
        
        return name_path
    
    @staticmethod
    def is_target(path):
        path_list = path.glob("*")
        return any([path.name == PretrainedModelDB.CONFIG_FILE for path in path_list])
    
    def get_name_list(self):
        return list(self.name_path.keys())
    
    def get(self, name):
        return self.name_path[name]

if __name__ == "__main__":
    mdb = PretrainedModelDB()
    models = mdb.get_name_list()
    print(models)
    print(models[0])
    print(mdb.get(models[0]))