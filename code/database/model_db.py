from .db import DB
import project

class ModelDB(DB):    
    DIR = "models"
    ROOT = f"{project.PROJECT_ROOT}/{DIR}"
    CONFIG_NAME = "config.yml"
    
    def __init__(self):
        super().__init__(self.ROOT, self.CONFIG_NAME)
        
    
if __name__ == "__main__":
    mdb = ModelDB()
    print(mdb.get_all_id())
    print()
    id = mdb.get_all_id()[0]
    print(mdb.get_config(id))