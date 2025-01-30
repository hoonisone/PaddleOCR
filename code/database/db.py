import project

import yaml
from pathlib import Path
from abc import ABC, abstractmethod
import os
from collections import OrderedDict
import shutil
    
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
    
class DBElement(ABC):
    RELATIVE_TO = "absolute"
    
    def __init__(self, dir_name, config_name):
        """
            dir_path: 파일 시스템에서 데이터를 저장하는 디렉터리 경로
            config_name: 디렉터리 내에 config를 저장하는 파일 이름
        """
        self.project_dir = Path(project.PROJECT_ROOT)
        self.path_dir = {}
        self.dir_path = self.project_dir/dir_name
        self.config_name = config_name
    
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
            absolute_path = Path(path)
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

    def get_path(self, target):
        path = self.path_dir[target]
        path = self.relative_to(path, current_relative="absolute", target_relative=self.RELATIVE_TO)
        return str(path).replace("\\", "/")


    def validate_current_relative_value(self, value, make_error=True):
        CURRENT_RELATIVE = ["record", "db", "project", "absolute"]
        self.validate_value_by_valid_value_list(value, CURRENT_RELATIVE, make_error=make_error, value_name="current_relative")
        
    def validate_target_relative_value(self, value, make_error=True):
        TARGET_RELATIVE = ["record", "db", "project", "absolute"]
        self.validate_value_by_valid_value_list(value, TARGET_RELATIVE, make_error=make_error, value_name="target_relative")
    

    @property
    def config_path(self):
        return self.dir_path/self.config_name
    
    @property
    def config(self):
        return self.load_yaml(self.config_path)
    
    @config.setter
    def config(self, config):
        self.save_yaml(self.config_path, config)
            
    def load_yaml(self, path):
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if config == None:
            config = {}
        return config     
    
    def save_yaml(self, path, config):
        with open(path, "w") as f:
            yaml.dump(config, f)
        
class DB2(DBElement):
    # 지정된 디렉터리 안에 모든 데이터를 저장하며
    # 마치 DB 처럼 쿼리를 제공하는 클래스
    # 모든 메타 데이터는 config.yaml 파일로 관리 
    
    def __init__(self, name, record_class, 
                 config_name = "db_config.yml", record_config_name="config.yml"):
        """
        name: 데이터 베이스 테이블 이름이자 디렉터리 이름
        record_class: 테이블이 관리하는 record에 대한 class
        """
        super().__init__(dir_name = name, config_name = config_name) 
        self.name = name
        self.dir = self.project_dir/name
        self.record_config_name = record_config_name
        self.record_class = record_class
        
        
    @property
    def records(self):
        if hasattr(self, "__records") == False:
            
            for id, name in self.record_id_to_name.items():
                dir_name = self.compose_dir_name(id, name)
            records = {id:self.record_class(self, self.compose_dir_name(id, name) ) for id, name in self.record_id_to_name.items()}
            
            sorted_records = {key: records[key] for key in sorted(records)}
            self.__records = sorted_records
 
        return self.__records

    @property
    def record_names(self):
        if hasattr(self, "__names") == False:
            self.__names = list(self.record_id_to_name.values())
            self.__names.sort()
        return self.__names
    
    @property
    def record_ids(self):
        if hasattr(self, "__ids") == False:
            self.__ids = list(self.record_name_to_id.values())
            self.__ids.sort()
        return self.__ids
    
    @property
    def record_id_to_name(self): # main setting : 이 속성 값을 기반으로 다른 속성도 처리 됌
        if hasattr(self, "__id_to_name") == False:
            # db_dir 내에서 지정된 config file들의 경로 수집
            # recort dir는 반드시 config file을 가지고 있다고 가정
            path_list = self.dir.rglob(f"{self.record_config_name}")
            
            # 폴더 경로만 추출
            names = [str(path.parent.relative_to(self.dir)).replace("\\", "/") for path in path_list]
            # 폴더 경로에서 id와 name을 추출
            id_and_names = [self.decompose_dir_name(name) for name in names]
            id_to_name = {id:name for id, name in id_and_names}
            self.__id_to_name = id_to_name
        assert len(names) == len(id_to_name), "id 중복이 있을 수 있음"
        return self.__id_to_name
    
    @property
    def record_name_to_id(self):
        if hasattr(self, "__name_to_id") == False:
            self.__name_to_id = {name: id for id, name in self.record_id_to_name.items()}
        return self.__name_to_id
    
    @property
    def next_id(self):
        return max(self.record_ids) + 1
    
    # def validate_id_uniqueness(self):
    #     ids = self.records.keys()
    #     ids = list(ids)
    #     unique_ids = set(ids)
    #     print(ids)
    #     print(unique_ids)
    #     if len(ids) != len(unique_ids):
    #         raise ValueError("id should be unique")
    #     print("id uniqueness is validated")
        
    
    def get_record(self, id):
        if isinstance(id, str):
            id = self.record_name_to_id[id]
        return self.records[id]

    def compose_dir_name(self, id, name):
        """
        dir name은 id와 name으로 구성된다.
        이 두 요소를 가지고 dir_name을 만들어 반환한다.
        """
        return f"{name}___id_{id}"
    
    def decompose_dir_name(self, dir_name):
        
        """
        extract id and name from dir_name by decomposing it
        """
        split = dir_name.split("___id_")
        name = "___".join(split[:-1])
        id = int(split[-1]) 
        return id, name
    


    # def get_next_id(self):
    #     config = self.config
    #     next_id = config["last_id"]+1
    #     config["last_id"] = next_id
    #     self.config = config
    #     return next_id
    
    def copy_record(self, id):
        if isinstance(id, str):
            id = self.record_name_to_id[id]
            
        record = self.get_record(id)
        source_path = record.record_dir
        
        name = f"{record.name}_copy"
        destination_name = self.compose_dir_name(self.get_next_id(), name)
        destination_path = source_path.parent/destination_name
        
        shutil.copytree(source_path, destination_path)
        print("record is copied successfully")
        print(f"source: {source_path}")
        print(f"destination: {destination_path}")
    
class Record2(DBElement):
    """
        relative_to_list: config를 로드 및 저장할 때 경로를 자동으로 변환해주는 항목 리스트
    """
    def __init__(self, db, dir_name, relative_to_list):
        super().__init__(dir_name = Path(db.name)/dir_name, config_name = db.record_config_name)
        self.relative_to_list = relative_to_list
                
        self.db = db
        self.record_dir = self.db_dir/dir_name
        self.id, self.name = db.decompose_dir_name(dir_name)
    
    @property
    def db_dir(self):
        return self.db.dir

    @property
    def db_name(self):
        return self.db.name

    
    def get_config_path(self):
        return str(self.record_dir/self.config_name).replace("\\", "/")
    
 
    
    @property
    def config(self):
        if hasattr(self, "__config") == False:
            path = self.get_config_path()
            self.__config = self.load_yaml(path)
        
        return self.config_relative_to(self.__config, current_relative_to="record", target_relative_to=self.RELATIVE_TO)
    
    
    def config_relative_to(self, config, current_relative_to, target_relative_to):
        for k in self.relative_to_list:
            config[k] = self.relative_to(config[k], current_relative=current_relative_to, target_relative=target_relative_to)
        config["relative_to"] = target_relative_to
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
        current_relative_to = config["relative_to"]
        config = self.config_relative_to(config, current_relative_to=current_relative_to, target_relative_to="record")
        path = self.get_config_path()
        with open(path, "w") as f:
            yaml.dump(config, f)
    
    def get_path(self, target, level):
        """
            level = record 수준을 사용할 지, db 수준을 사용할 지
            db 수준은 record끼리 공유하는 파일을 말함
        """
        if level == "record":
            return super().get_path(target)
        elif level == "db":
            return self.db.get_path(target)

