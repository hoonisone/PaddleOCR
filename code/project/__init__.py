import yaml
from pathlib import Path


config_file_path = Path(__file__).parent/"config.yml"
with open(config_file_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PROJECT_ROOT = "E:/workspace/paddleocr2"
config["project_root"] = PROJECT_ROOT
for x in ["model_db", "dataset_db", "labelset_db", "work_db"]:
    if not config["db"][x]["root_dir"]:
        config["db"][x]["root_dir"] = f"{PROJECT_ROOT}/{x.split('_')[0]+'s'}"
        
# print(config)