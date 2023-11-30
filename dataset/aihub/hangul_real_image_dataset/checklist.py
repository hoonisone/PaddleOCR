import yaml
from pathlib import Path
class DirChecklist:
    @staticmethod
    def get_checklist(path):
        with open(path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)    
    
    @staticmethod
    def get_checked_dir_list(path):
        checklist = DirChecklist.get_checklist(path)
        
        return DirChecklist._get_checked_dir_list(checklist, "")  
    
    @staticmethod
    def _get_checked_dir_list(checklist, path):
        # checklist = dict(checklist)
        path = Path(path)
        if isinstance(checklist, dict):
            if checklist["all"] == True:
                return [path]
            elif checklist["all"] == False:
                return []
            else:
                del checklist["all"]
                return sum([DirChecklist._get_checked_dir_list(v, path/k) for k, v in checklist.items()], [])

        if checklist == True:
            return [path]
        else:
            return []