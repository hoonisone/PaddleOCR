import project

import yaml
from pathlib import Path
from abc import ABC, abstractmethod


class DB:
    # 지정된 디렉터리 안에 모든 데이터를 저장하며
    # 마치 DB 처럼 쿼리를 제공하는 클래스
    # 모든 메타 데이터는 config.yaml 파일로 관리 
    PROJECT_ROOT = project.PROJECT_ROOT
    
    def __init__(self, dir, config_name):
        self.dir = dir
        self.config_name = config_name

        
    def get_all_id(self):
        path_list = (Path(project.PROJECT_ROOT)/self.dir).glob("*")
        element_list = []
        for path in path_list:
            if path.is_dir() and self.is_target(path):
                element_list.append(path.name)
        return element_list
        
    def is_target(self, path):
        return len(list(path.glob(self.config_name))) == 1
        
    def get_path(self, id, config_name=None):
        
        config_name = self.config_name if config_name == None else config_name
        return str(Path(self.PROJECT_ROOT)/self.dir/id/config_name).replace("\\", "/")
    
    def get_config(self, id, config_name=None):
        
        path = self.get_path(id, config_name=config_name)
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if config == None:
            config = {}
        # config["id"] = id
        # if abs_path:
        #     config = self.replace(config, "{DIR}", self.dir)
        #     config = self.replace(config, "{ID}", id)
        return config
    
    def update_config(self, id, config, config_name=None):
        path = self.get_path(id, config_name=config_name)
        with open(path, "w") as f:
            yaml.dump(config, f)

    def replace(self, value, target , replace):
        # config 안에 모든 값들에 대해 target을 replace로 대체해줌
        if isinstance(value, str):
            return value.replace(target, replace)
        if isinstance(value, dict):
            return {k:self.replace(v, target, replace) for k, v in value.items()}
        if isinstance(value, list):
            return [self.replace(x, target, replace) for x in value]
        return value

    def relative_to(self, id, path, relative_to):    
        assert relative_to in ["dir", "project", "absolute"], f"relative_to should be 'dir', 'project', or 'absolute' but {relative_to} is given"
        if relative_to == "dir":
            return str(Path(id)/path).replace("\\", "/")
        elif relative_to == "project":
            return str(Path(self.dir)/id/path).replace("\\", "/")
        elif relative_to == "absolute":
            return str(Path(self.PROJECT_ROOT)/self.dir/id/path).replace("\\", "/")
        else:
            return path
    


class DB2(ABC):
    # 지정된 디렉터리 안에 모든 데이터를 저장하며
    # 마치 DB 처럼 쿼리를 제공하는 클래스
    # 모든 메타 데이터는 config.yaml 파일로 관리 
    PROJECT_ROOT = project.PROJECT_ROOT
    
    def __init__(self, db_name, config_name):    
        self.db_name = db_name
        
        self.project_dir = Path(self.PROJECT_ROOT)
        self.db_dir = self.project_dir/db_name
        
        self.config_name = config_name

    def get_all_id(self):
        path_list = self.db_dir.glob("*")
        element_list = []
        for path in path_list:
            if path.is_dir() and self.is_target(path):
                element_list.append(path.name)
        return element_list

    def is_target(self, path):
        return len(list(path.glob(self.config_name))) == 1
    
    @abstractmethod    
    def get_record(self, record_id):
        pass
        # return Record(self, record_id)
        
    def get_path(self, target, relative_to="absolute"):
        path = self.path_dir[target]
        path = self.relative_to(path, current_relative="absolute", target_relative=relative_to)
        return str(path).replace("\\", "/")
    
    def relative_to(self, path, current_relative, target_relative):
        if current_relative == target_relative:
            return path
        
        # argument validation
        self.validate_current_relative_value(current_relative)
        self.validate_target_relative_value(target_relative)
        

        # convert to absolute path
        if current_relative == "record":
            absolute_path = self.record_dir/path
        elif current_relative == "db":
            absolute_path = self.db_dir/path
        elif current_relative == "project":
            absolute_path = self.project_dir/path
        elif current_relative == "absolute":
            absolute_path = path
        else:
            raise NotImplementedError()
        
        
        
        # convert to relative path        
        if target_relative == "record": # 현재 db 폴더 기준 상대 경로로 변경
            final_path = absolute_path.relative_to(self.record_dir)
        elif target_relative == "db": # 현재 db 폴더 기준 상대 경로로 변경
            final_path = absolute_path.relative_to(self.db_dir)    
        elif target_relative == "project": # 프로젝트 루트 기준 상대 경로로 변경
            final_path = absolute_path.relative_to(self.project_dir)
        elif target_relative == "absolute": # 절대 경로로 변경
            final_path = absolute_path
            pass
        else:
            raise NotImplementedError()
        
        return str(final_path).replace("\\", "/")
    
    def validate_value_by_valid_value_list(self, value, valid_values, make_error = True, value_name="value"):
        # 인자와 인자의 유효한 값 리스트를 받아, 주어진 값이 유효한 값에 속하는지 검사
        # make_error가 True이면 assert error를 발생시키고, False이면 결과값을 반환
        validity = value in valid_values
        if make_error:
            assert validity, f"{value_name} should be in {valid_values}, but {value} is given"
        else:
            return validity


    def validate_current_relative_value(self, value, make_error=True):
        CURRENT_RELATIVE = ["record", "db", "project", "absolute"]
        self.validate_value_by_valid_value_list(value, CURRENT_RELATIVE, make_error=make_error, value_name="current_relative")
        
    def validate_target_relative_value(self, value, make_error=True):
        TARGET_RELATIVE = ["record", "db", "project", "absolute"]
        self.validate_value_by_valid_value_list(value, TARGET_RELATIVE, make_error=make_error, value_name="target_relative")


    
class Record2(DB2):
    def __init__(self, db, record_id):
        super().__init__(db.db_name, db.config_name)
        self.db = db
        
        self.record_id = record_id
        self.record_dir = self.db_dir/record_id

    
    def get_record(self, record_id):
        raise NotImplementedError()
    
    def get_config_path(self):
        return str(self.record_dir/self.config_name).replace("\\", "/")
    
    def get_config(self):
        
        path = self.get_config_path()
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if config == None:
            config = {}
        # config["id"] = id
        # if abs_path:
        #     config = self.replace(config, "{DIR}", self.dir)
        #     config = self.replace(config, "{ID}", id)
        return config

    def replace(self, value, target, replace):
        # config 안에 모든 값들에 대해 target을 replace로 대체해줌
        if isinstance(value, str):
            return value.replace(target, replace)
        if isinstance(value, dict):
            return {k:self.replace(v, target, replace) for k, v in value.items()}
        if isinstance(value, list):
            return [self.replace(x, target, replace) for x in value]
        return value
    
    def update_config(self, config):
        path = self.get_config_path()
        with open(path, "w") as f:
            yaml.dump(config, f)
    
    def get_path(self, target, level, relative_to="absolute"):
        """
            level = record 수준을 사용할 지, db 수준을 사용할 지
            db 수준은 record끼리 공유하는 파일을 말함
        """
        if level == "record":
            return super().get_path(target, relative_to=relative_to)
        elif level == "db":
            return self.db.get_path(target, relative_to=relative_to)
