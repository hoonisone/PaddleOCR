from pathlib import Path
import project
from .db import DB

class DatasetDB(DB):    
    DIR = "datasets"
    ROOT = f"{project.PROJECT_ROOT}/{DIR}"
    CONFIG_NAME = "dataset_config.yml"
    def __init__(self):

        super().__init__(DatasetDB.ROOT, DatasetDB.CONFIG_NAME)    
    
    def get_all_labels(self, id):
        # 모든 레이블 파일을 로드하여 리스트로 합쳐 반환
        config = self.get_config(id)       
        
        label_paths = [Path(self.ROOT)/config["id"]/labelfile for labelfile in config["labelfiles"]]
        labels = sum([self.load_text_file(labelfile_path) for labelfile_path in label_paths], [])
        print(labels[0])
        return [str(Path(config["id"])/label) for label in labels] # dataset 기준 상대 경로로 바꾸기
        
    @staticmethod
    def load_text_file(path):
        # 단일 레이블 파일을 로드하여 리스트로 반환
        # datasets root로 부터 상대 경로로 변환
        with open(path) as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines] #
        return lines

if __name__ == "__main__":
    mdb = DatasetDB()
    print(mdb.get_all_id())
    print()
    id = mdb.get_all_id()[0]
    print(id)
    print(mdb.get_config(id))
    
# class DatasetDB2: # DatasetDB
#     ROOT = project.ROOT/"datasets"
#     CONFIG_FILE = "config.yml"
    
#     def __init__(self, root=None):
#         self.root = root if root else DatasetDB2.ROOT
        
#         self.name_path = DatasetDB2.get_name_path_list(self.root)        

#     @staticmethod    
#     def get_path_list(root):
#         path_list = root.glob("*")

#         dataset_list = []
#         for path in path_list:
#             if DatasetDB2.is_target(path):
#                 dataset_list.append(path)
#             elif path.is_dir():
#                 dataset_list += DatasetDB2.get_path_list(path)
#             else:
#                 pass
#         return dataset_list
            
#     @staticmethod
#     def get_name_path_list(root):  
#         name_path = {}
#         for path in DatasetDB2.get_path_list(root):
#             with open(str(path/DatasetDB2.CONFIG_FILE)) as f:
#                 config = yaml.load(f, Loader=yaml.FullLoader)
#             for i, label_file in enumerate(config["labelfiles"]):
#                 config["labelfiles"][i] = path/label_file
#             key = config["name"]
#             value = config
#             name_path[key] = value
        
#         return name_path
    
#     def get_name_list(self):
#         return list(self.name_path.keys())
    
#     def get_path(self, name):
#         return self.name_path[name]
    
#     def get_label_file_path(self, name):
#         return self.name_path[name]["labelfiles"]

# if __name__ == "__main__":


#     # exit(-1)
#     datasetDB = DatasetDB2()
#     names = datasetDB.get_name_list()
#     print(names)
#     print(datasetDB.get_path(names[0]))
#     print(datasetDB.get_label_file_path(names[0]))
    