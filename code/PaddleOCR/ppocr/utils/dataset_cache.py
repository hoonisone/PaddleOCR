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
        self.read_fp = None

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
         with h5py.File(self.target_filename, 'a') as f:
            if sample_key in f:
                return
            group = f.create_group(sample_key)
            self.save_dict_to_hdf5(group, data_sample)

    # def save_samples_to_hdf5_old(self, data_sample, sample_key, lock = True):
    #     while True:
    #         try:
    #             if lock:
    #                 with self.lock:
    #                     with h5py.File(self.target_filename, 'a') as f:
    #                         if sample_key in f:
    #                             return
    #                         group = f.create_group(sample_key)
    #                         self.save_dict_to_hdf5(group, data_sample)
    #                     break
    #             else:
    #                 with h5py.File(self.target_filename, 'a') as f:
    #                     if sample_key in f:
    #                         return
    #                     group = f.create_group(sample_key)
    #                     self.save_dict_to_hdf5(group, data_sample)
    #                 break
    #         except BlockingIOError as e:
    #             continue        
            
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

    # def load_samples_from_hdf5(self, sample_key):
    #     with open(self.filename, "r") as f:
    #         with h5py.File(self.filename, 'r', swmr=True) as h5f:
    #             if sample_key in h5f:
    #                 out = self.load_dict_from_hdf5(h5f[sample_key])
    #             else:
    #                 out = None
    #         return out

    def open_hdf5(self):
        """HDF5 파일을 미리 열어둠"""
        if self.read_fp is None:
            self.read_fp = h5py.File(self.filename, 'r', swmr=True)

    def close_hdf5(self):
        """HDF5 파일을 닫음"""
        if self.read_fp is not None:
            self.read_fp.close()
            self.read_fp = None
    
    def load_samples_from_hdf5(self, sample_key):
        if not Path(self.filename).exists():
            return None
        self.open_hdf5()
        try:
            return self.load_dict_from_hdf5(self.read_fp[sample_key])
        except:
            return None
        
        
    # def load_samples_from_hdf5_old(self, sample_key):
        
    #     if not Path(self.filename).exists():
    #         return None
    #     with open(self.filename, "r") as f:
    #         fcntl.flock(f, fcntl.LOCK_SH)
    #         with h5py.File(self.filename, 'r', swmr=True) as h5f:
    #             if sample_key in h5f:
    #                 out = self.load_dict_from_hdf5(h5f[sample_key])
    #             else:
    #                 out = None
    #         fcntl.flock(f, fcntl.LOCK_UN)
    #         return out
        
    def __del__(self):
        self.close_hdf5()

import pickle
class HDF5PickleStorage:
    def __init__(self, file_path):
        self.filename = file_path
        self.read_fp = None
        if not Path(self.filename).exists():
            with h5py.File(self.filename, 'w'):
                pass


    def save_samples_to_hdf5(self, obj, key):
        """Save a Python object in HDF5 using pickle serialization."""
        with h5py.File(self.filename, 'a') as hdf5_file:
            pickled_obj = pickle.dumps(obj)
            
            # Check if the key already exists, and if so, delete it before saving
            if key in hdf5_file:
                return 
                # del hdf5_file[key]
            
            # Save the binary pickle data as a byte array using numpy
            hdf5_file.create_dataset(key, data=np.frombuffer(pickled_obj, dtype='uint8'))

    def load_samples_from_hdf5(self, key):
        """Load a Python object from HDF5 by unpickling."""
        
        with h5py.File(self.filename, 'r') as hdf5_file:
            if key not in hdf5_file:
                return None
            pickled_obj = hdf5_file[key][()]
            obj = pickle.loads(pickled_obj.tobytes())  # Convert back to bytes and unpickle
        return obj
    
    
    def open_hdf5(self):
        """HDF5 파일을 미리 열어둠"""
        if self.read_fp is None:
            self.read_fp = h5py.File(self.filename, 'r', swmr=True)

    def close_hdf5(self):
        """HDF5 파일을 닫음"""
        if self.read_fp is not None:
            self.read_fp.close()
            self.read_fp = None
            
    def __del__(self):
        self.close_hdf5()
        
    def get_all_keys(self):
        """Recursively return a list of all keys (paths) in the HDF5 file."""
        with h5py.File(self.filename, 'r') as hdf5_file:
            all_keys = []

            def recursive_keys(name, obj):
                # If the object is a dataset, add its full path to the key list
                if isinstance(obj, h5py.Dataset):
                    all_keys.append(name)

            # Visit all nodes in the HDF5 file and collect datasets
            hdf5_file.visititems(recursive_keys)

            return all_keys