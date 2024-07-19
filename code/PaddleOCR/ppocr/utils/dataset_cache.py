import h5py
import numpy as np
from pathlib import Path
from filelock import FileLock
import fcntl

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Path("/home/dataset_cache.h5").unlink()

class DatasetCache:
    def __init__(self, cache_file):
        self.filename = cache_file
        self.target_filename = self.filename
        # i = 1
        # self.target_filename = f"/home/dataset_cache{i}.h5"
        # while Path(self.target_filename).exists():
        #     i+=1
        #     self.target_filename = f"/home/dataset_cache{i}.h5"
            

        self.lock = FileLock(f"{self.filename}.lock", timeout=10)
        
        self.cache_list = []

    # HDF5 파일에 샘플 데이터 저장
    def save_dict_to_hdf5(self, group, data):
        
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self.save_dict_to_hdf5(subgroup, value)
            else:
                if isinstance(value, np.ndarray) and value.dtype.type is np.str_:
                    dt = h5py.special_dtype(vlen=str)
                    group.create_dataset(key, data=value.astype(dt))
                else:
                    group.create_dataset(key, data=value)




    def save_samples_to_hdf5(self, data_sample, sample_key, lock = True):
        while True:
            try:
                if lock:
                    with self.lock:
                        with h5py.File(self.target_filename, 'a') as f:
                            if sample_key in f:
                                return
                            group = f.create_group(sample_key)
                            self.save_dict_to_hdf5(group, data_sample)
                        break
                else:
                    with h5py.File(self.target_filename, 'a') as f:
                        if sample_key in f:
                            return
                        group = f.create_group(sample_key)
                        self.save_dict_to_hdf5(group, data_sample)
                    break
            except BlockingIOError as e:
                continue        
            
    def load_dict_from_hdf5(self, group):
        data = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                data[key] = self.load_dict_from_hdf5(group[key])
            else:
                dataset = group[key]
                if dataset.shape == ():  # 스칼라 데이터셋
                    value = dataset[()]
                    if isinstance(value, (bytes, np.bytes_)):
                        value = value.decode('utf-8')  # 바이트를 문자열로 디코딩
                else:
                    value = dataset[:]
                    if isinstance(value, np.ndarray) and value.dtype.type is np.bytes_:
                        value = value.astype(str)
                    elif isinstance(value, np.ndarray) and value.dtype.type is np.object_:
                        value = np.array([item.decode('utf-8') if isinstance(item, (bytes, np.bytes_)) else item for item in value])
                data[key] = value
        return data

    def load_samples_from_hdf5(self, sample_key):
        
        if not Path(self.filename).exists():
            return None
        with open(self.filename, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            with h5py.File(self.filename, 'r', swmr=True) as h5f:
                if sample_key in h5f:
                    out = self.load_dict_from_hdf5(h5f[sample_key])
                else:
                    out = None
            fcntl.flock(f, fcntl.LOCK_UN)
            return out
