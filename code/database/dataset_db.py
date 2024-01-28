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
        return [str(Path(config["id"])/label).replace('\\', '/') for label in labels] # dataset 기준 상대 경로로 바꾸기
        
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