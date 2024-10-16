import h5py
import numpy as np
import cv2
from paddle.io import Dataset
from .imaug import transform, create_operators


class H5Dataset(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        """
        HDF5 파일에서 데이터를 읽는 Dataset 클래스.
        
        Args:
            config (dict): 데이터셋 설정 정보를 담은 구성 파일
            mode (str): 'train', 'test' 등 모드를 지정
            logger (object): 로깅 객체
            seed (int, optional): 랜덤 시드를 설정 (기본값: None)
        """
        super(H5Dataset, self).__init__()
        
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        batch_size = loader_config['batch_size_per_card']
        data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        # HDF5 파일 경로 설정
        h5_file_path = f"{data_dir}/dataset.h5"
        h5_file_path = dataset_config["data_dir"]
        logger.info(f"Loading HDF5 dataset from {h5_file_path}")
        
        # HDF5 파일 열기 (읽기 전용)
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']
        self.num_samples = self.images.shape[0]

        # 데이터 인덱스 순서 리스트 생성
        self.data_idx_order_list = self.dataset_traversal()
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)

        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 1)
        
        ratio_list = dataset_config.get("ratio_list", [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]

    def dataset_traversal(self):
        """HDF5 dataset을 순회하며 각 데이터의 인덱스를 저장"""
        total_sample_num = self.num_samples
        data_idx_order_list = np.arange(total_sample_num)
        return data_idx_order_list

    def get_img_data(self, idx):
        """인덱스 기반으로 HDF5에서 이미지 데이터를 읽어옴"""
        img_bytes = self.images[idx]
        if img_bytes is None:
            return None
        img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img


    def get_label(self, idx):
        """인덱스 기반으로 HDF5에서 레이블 데이터를 읽어옴"""
        label = self.labels[idx].decode('utf-8')
        return label

    def get_ext_data(self):
        """확장 데이터를 불러오는 함수 (필요에 따라 구현)"""
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            idx = np.random.randint(len(self))
            sample_info = self.__getitem__(idx)
            if sample_info is None:
                continue
            img, label = sample_info
            data = {'image': img, 'label': label}
            data = transform(data, load_data_ops)
            if data is None:
                continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        """인덱스를 기반으로 HDF5에서 이미지와 레이블 데이터를 불러옴"""
        idx = self.data_idx_order_list[idx]
        img = self.get_img_data(idx)
        label = self.get_label(idx)

        if img is None or label is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        data = {'image': img, 'label': label}
        data['ext_data'] = self.get_ext_data()
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        """데이터셋의 총 샘플 수 반환"""
        return self.data_idx_order_list.shape[0]

    def close(self):
        """HDF5 파일 사용이 끝나면 파일 닫기"""
        self.h5_file.close()
        
# class H5Dataset(Dataset):
#     def __init__(self, h5_file_path, transform=None):
#         """
#         HDF5 파일에서 데이터를 읽는 Dataset 클래스.
        
#         Args:
#             h5_file_path (str): HDF5 파일 경로
#             transform (callable, optional): 이미지에 적용할 변환 함수
#         """
#         super(H5Dataset, self).__init__()
#         self.h5_file_path = h5_file_path
#         self.transform = transform

#         # HDF5 파일 열기 (읽기 전용)
#         self.h5_file = h5py.File(self.h5_file_path, 'r')
#         self.images = self.h5_file['images']
#         self.labels = self.h5_file['labels']
#         self.num_samples = self.images.shape[0]

#     def __getitem__(self, idx):
#         # 인덱스를 기반으로 이미지와 레이블 불러오기
#         img_bytes = self.images[idx]
#         label = self.labels[idx].decode('utf-8')

#         # 이미지를 바이너리에서 디코딩
#         img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

#         # 변환 적용 (있을 경우)
#         if self.transform:
#             img = self.transform(img)

#         return img, label

#     def __len__(self):
#         # 데이터셋의 총 샘플 수 반환
#         return self.num_samples

#     def close(self):
#         # 데이터셋 사용이 끝나면 HDF5 파일 닫기
#         self.h5_file.close()