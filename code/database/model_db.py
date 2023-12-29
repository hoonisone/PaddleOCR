from db import DB


class ModelDB(DB):    
    def __init__(self):
        super().__init__("model_db")
        
    
if __name__ == "__main__":
    mdb = ModelDB()
    print(mdb.get_all_id())
    print()
    id = mdb.get_all_id()[0]
    print(mdb.get_value(id))