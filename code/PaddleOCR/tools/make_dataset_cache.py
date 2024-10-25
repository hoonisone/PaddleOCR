# # Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from PaddleOCR.tools import train

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import yaml
import paddle
import paddle.distributed as dist
import numpy as np

import multiprocessing
import time

from ppocr.utils.dataset_cache import DatasetCache, HDF5PickleStorage

from ppocr.data import build_dataloader, set_signal_handlers
import tools.program as program
import tqdm

dist.get_world_size()

def chunk_list(data, num_chunks):
    avg = len(data) / float(num_chunks)
    chunks = []
    last = 0.0

    while last < len(data):
        chunks.append(data[int(last):int(last + avg)])
        last += avg

    return chunks

def load_data(queue, data, dataset):
    for i in tqdm.tqdm(data):
        try:
            sample = dataset[i]
            if sample is not None:
                queue.append(sample)
        except Exception as e:
            print(e)
            continue
    print("종료")

def sub_task(dataset, cache, idx_list, worker_num):
    
    
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    
    data_parts = chunk_list(idx_list, worker_num)
    
    processes = []

    for part in data_parts:
        p = multiprocessing.Process(target=load_data, args=(shared_list, part, dataset))
        processes.append(p)
        p.start()


    for p in processes:
        p.join()
    
    cache.close_hdf5()
    for sample in tqdm.tqdm(shared_list):
        try:
            key = sample["img_path"]
            cache.save_samples_to_hdf5(sample, key)
        except Exception as e:
            print(e)
            continue
        

def main(config, device, logger, vdl_writer):

    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

    

    # build dataloader
    set_signal_handlers()

    
    # config["Eval"]["dataset"]["transforms_uncachiable"] = []
    # train_dataset = build_dataloader(config, 'Eval', device, logger).dataset
    # train_cache = HDF5PickleStorage(config["Eval"]["dataset"]["cache_file"])
    
    config["Train"]["dataset"]["transforms_uncachiable"] = []
    train_dataset = build_dataloader(config, 'Train', device, logger).dataset
    train_cache = HDF5PickleStorage(config["Train"]["dataset"]["cache_file"])
    
    
    size = len(train_dataset)
    for i in range(0, size//100000+1):    
        sub_task(train_dataset, train_cache, range(100000*i, min(100000*(i+1), size)), worker_num=30)

        

if __name__ == '__main__':
    print("시작")
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    main(config, device, logger, vdl_writer)

