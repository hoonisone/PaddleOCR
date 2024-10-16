# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from charset_normalizer import CharsetMatches
import numpy as np
import cv2
import math
import os
import json
import random
import traceback
from paddle.io import Dataset
from .imaug import transform, create_operators
from ppocr.utils.dataset_cache import DatasetCache, HDF5PickleStorage

class SimpleDataSet(Dataset): 
    def __init__(self, config, mode, logger, seed=None):
        
        super(SimpleDataSet, self).__init__()
        self.logger = logger # 메인 코드에서 logger를 세팅한 뒤 각 객체에 넘겨 공유 -> 일관된 로그 출력 및 관리 가능
        self.mode = mode.lower() # 소문자로 통일하면 대소 문자 구분 실수를 줄일 수 있을것이다.


        # 필요한 sub config 파일 추출
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        

        # config로 부터 필요한 데이터(속성, 옵션) 추출
        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list') # 왜 get이 아니라 pop으로 했을까?
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        self.cache = dataset_config.get('use_cache', False)
        if self.cache:
            self.cache_file = dataset_config.get('cache_file', "/home/dataset_cache.h5")
            

            self.dataset_cache = HDF5PickleStorage(self.cache_file)


        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(ratio_list) == data_source_num, "The length of ratio_list should be the same as the file_list."
        # config 파일에 2개의 값을 입력해야 하고, 두 값의 특정 제약 조건이 있는 경우임, 이럴때는 입력 값 체크가 필요함

        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed # 이 값도 config로 받아도 되지 않나?

        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines))) # 언제 쓰는걸까?
        if self.mode == "train" and self.do_shuffle: # train이 아니면 suffle이 의미가 없긴 하지
            self.shuffle_data_random()

        self.set_epoch_as_seed(self.seed, dataset_config)

        self.ops = create_operators(dataset_config['transforms'], global_config)
        # print(dataset_config.get('transforms_uncachiable', []))
        self.ops_uncachiable = create_operators(dataset_config.get('transforms_uncachiable', []), global_config)
        
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",2)

        self.need_reset = True in [x < 1 for x in ratio_list]
        
        
    """
        self.data_lines: 각 샘플별 정보를 담는 리스트
    """
        

    def set_epoch_as_seed(self, seed, dataset_config):
        """
            seed로 설정 가능한 옵션을 갖는 transform 함수에 대해 주어진 seed로 세팅하는 함수
            Args:
                - seed(int): 세팅하고 싶은 seed 값
                - dataset_config(dict): 설정을 바꾸고 싶은 대상 config 데이터

        """
        if self.mode == 'train':
            try:
                border_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeBorderMap' in dictionary][0]
                shrink_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeShrinkMap' in dictionary][0]

                dataset_config['transforms'][border_map_id]['MakeBorderMap'][
                    'epoch'] = seed if seed is not None else 0

                dataset_config['transforms'][shrink_map_id]['MakeShrinkMap'][
                    'epoch'] = seed if seed is not None else 0
            except Exception as E:
                print(E)
                return

    def get_image_info_list(self, file_list, ratio_list):
        """
            데이터 셋의 샘플 리스트를 읽어들여 통합하여 반환
            Args:
                file_list (list[str], str): 레이블 파일 리스트 []
                ratio_list (list): 각 레이블 파일별로 얼마나 사용할 지

            Return:
                (list): 전체 데이터 샘플 정보 리스트 ex [sample1, sample2, ...]
        """
        # ratio_list는 각 대응되는 file_list를 얼마나 쓸 것인지 결정하는 듯
        
        if isinstance(file_list, str): # 이런식으로 하면 복수개인 경우 list, 단일 값인 경우 1개만 적어도 되겠네
            file_list = [file_list] # 이 부분은 함수 내부 말고 객체 init 단에서 수행했어도 좋았을 듯


        data_lines = []
        for idx, file in enumerate(file_list): # ratio_list와 file_list를 zip으로 연결하면 더 깔끔할 텐데
            with open(file, "rb") as f: # 왜 byte 모드로 읽어들였을까?
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0: # 왜 train으로 한정했을까? eval, test도 그대로 적용해도 될 텐데?
                    random.seed(self.seed) # Dataset init 때 한 번만 하면 안되나?, 아 항상 특정 순서로 하려면 함수를 수행할 때 마다 하는게 맞을 듯
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines) # shuffle 부분을 전체를 다 가져온 후 하면 안되는건가?
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        """_summary_
            extract only one file name
        Args:
            file_name (json, str): file_name 정보

        Returns:
            str: file_name 중 하나 반황
        """
        if len(file_name) > 0 and file_name[0] == "[":
            # 개발자가 곧 인자 값을 작성하는 사람이니 "["만으로 구분하는게 큰 문제가 되지 않을 수 있지만
            # json으로 파싱 가능한가? 라고 했다면 더 좋지 않을까?
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def get_ext_data(self): # 무슨 말인지 모르겠음 (ext -> extension? 더 많은 데이터를 포함한다는것일까?)
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        ext_op_keys = list(self.ops.keys())[:self.ext_op_transform_idx]

        load_data_ops = {key: self.ops[key] for key in ext_op_keys}
        # load_data_ops = self.ops[:self.ext_op_transform_idx]
        # for key in 
        
        ext_data = []
 
        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__(
            ))]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if 'polys' in data.keys():
                if data['polys'].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
    
        file_idx = self.data_idx_order_list[idx] # self.data_idx_order_list[idx]는 항상 idx와 동일한 것 아닌가?..
        data_line = self.data_lines[file_idx] # data_line은 idx에 대한 데이터 정보 한 줄을 의미함 (label file에서의 한 줄)
        try:
            data_line = data_line.decode('utf-8') # data_line을 byte 형태로 읽어서 관리하고 있기 때문에 decode 과정이 필요
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name) # file_name 이 json인 경우 처리해주는 건데, 그런 경우가 언제 있는지?...
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
        
            if self.cache:
                cache_key = img_path
                cache_key = str(cache_key)[2:] # key 값에는 ./ 이 없음
                # print(cache_key)
                data = self.dataset_cache.load_samples_from_hdf5(cache_key)
                # print(data)
                if data is not None:
                    
                    ############################################################################################################################################# 임시 방편 코드
                    # h5 에 저장했다가 로드하면 list나 float나 죄다 numpy로 변환되서 문제 생감
                    # ABINet에 ext_data와 valid_ratio가 numpy가 되면 안되는 거 같음
                    # if "valid_ratio" in data:
                    #     data["valid_ratio"] = float(data["valid_ratio"])
                    # if "ext_data" in data:
                    #     data["ext_data"] = data["ext_data"].tolist()
                    ############################################################################################################################################
                    # print("hit")
                    # print(1, data)
                    data = transform(data, self.ops_uncachiable)
                    # print(2, data)
                    # print(data["image"].shape)
                    # print("hit")
                    return data
            # print("not hit")
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read() # 바이트로 넘기네? PIL.Image나 numpy.array가 아니고? 이유는 뭘까? 나중에 추가 변환을 해야 해서?
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
            data = transform(data, self.ops_uncachiable)
                      
            
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            with open("/home/data_error.txt", "a") as f:
                f.write(data_line)
                
            outs = None
        
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
            
        if self.cache:
            pass
            # self.dataset_cache.save_samples_to_hdf5(outs, cache_key)
        
        # print("B", type(outs["image"]))
        return outs            

    def __len__(self):
        return len(self.data_idx_order_list)


class SimpleDataSet_Test(Dataset):
    
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet_Test, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", 1.0)
        
        self.cache = dataset_config.get('use_cache', True)
        if self.cache:
            self.cache_file = dataset_config.get('cache_file', "/home/dataset_cache.h5")
        
            self.dataset_cache = DatasetCache(self.cache_file)

        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(
            ratio_list
        ) == data_source_num, "The length of ratio_list should be the same as the file_list."
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed

        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()

        self.set_epoch_as_seed(self.seed, dataset_config)

        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",2)

        self.need_reset = True in [x < 1 for x in ratio_list]
        
        

    def set_epoch_as_seed(self, seed, dataset_config):
        if self.mode == 'train':
            try:
                border_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeBorderMap' in dictionary][0]
                shrink_map_id = [index
                    for index, dictionary in enumerate(dataset_config['transforms'])
                    if 'MakeShrinkMap' in dictionary][0]

                dataset_config['transforms'][border_map_id]['MakeBorderMap'][
                    'epoch'] = seed if seed is not None else 0

                dataset_config['transforms'][shrink_map_id]['MakeShrinkMap'][
                    'epoch'] = seed if seed is not None else 0
            except Exception as E:
                print(E)
                return

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def _try_parse_filename_list(self, file_name):
        # multiple images -> one gt label
        if len(file_name) > 0 and file_name[0] == "[":
            try:
                info = json.loads(file_name)
                file_name = random.choice(info)
            except:
                pass
        return file_name

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            file_idx = self.data_idx_order_list[np.random.randint(self.__len__(
            ))]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)

            if data is None:
                continue
            if 'polys' in data.keys():
                if data['polys'].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            cache_key = img_path
            
            if self.cache:
                data = self.dataset_cache.load_samples_from_hdf5(cache_key)
                if data:
                    return data
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
        except:
            # self.logger.error(
            #     "When parsing line {}, error happened with msg: {}".format(
            #         data_line, traceback.format_exc()))
            outs = None
    
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
            
        if self.cache:
            pass
            # self.dataset_cache.save_samples_to_hdf5(outs, cache_key)
            

        return outs            

    def __len__(self):
        return len(self.data_idx_order_list)
    
class SimpleDataSet_Cache(SimpleDataSet):

    def __getitem__(self, idx):
        
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            cache_key = img_path

            if self.cache:
                data = self.dataset_cache.load_samples_from_hdf5(cache_key)
                if data:
                    # print(cache_key, "hit")
                    return None
                else:
                    pass
                    # print(cache_key)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
        except:
            # self.logger.error(
            #     "When parsing line {}, error happened with msg: {}".format(
            #         data_line, traceback.format_exc()))
            outs = None
    
        if outs is None:
             
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__(
            )) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
            
        if self.cache:
            pass
            # self.dataset_cache.save_samples_to_hdf5(outs, cache_key)
        return outs, cache_key
        return outs            


class MultiScaleDataSet(SimpleDataSet):
    def __init__(self, config, mode, logger, seed=None):
        super(MultiScaleDataSet, self).__init__(config, mode, logger, seed)
        self.ds_width = config[mode]['dataset'].get('ds_width', False)
        if self.ds_width:
            self.wh_aware()

    def wh_aware(self):
        data_line_new = []
        wh_ratio = []
        for lins in self.data_lines:
            data_line_new.append(lins)
            lins = lins.decode('utf-8')
            name, label, w, h = lins.strip("\n").split(self.delimiter)
            wh_ratio.append(float(w) / float(h))

        self.data_lines = data_line_new
        self.wh_ratio = np.array(wh_ratio)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.data_idx_order_list = list(range(len(self.data_lines)))

    def resize_norm_img(self, data, imgW, imgH, padding=True):
        img = data['image']
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')

        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = padding_im
        data['valid_ratio'] = valid_ratio
        return data

    def __getitem__(self, properties):
        # properites is a tuple, contains (width, height, index)
        img_height = properties[1]
        idx = properties[2]
        if self.ds_width and properties[3] is not None:
            wh_ratio = properties[3]
            img_width = img_height * (1 if int(round(wh_ratio)) == 0 else
                                      int(round(wh_ratio)))
            file_idx = self.wh_ratio_sort[idx]
        else:
            file_idx = self.data_idx_order_list[idx]
            img_width = properties[0]
            wh_ratio = None

        data_line = self.data_lines[file_idx]
    
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            file_name = self._try_parse_filename_list(file_name)
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops[:-1])
            if outs is not None:
                outs = self.resize_norm_img(outs, img_width, img_height)
                outs = transform(outs, self.ops[-1:])
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__([img_width, img_height, rnd_idx, wh_ratio])
        return outs


if __name__ == "__main__":
    print("hello")
