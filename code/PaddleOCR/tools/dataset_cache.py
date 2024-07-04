# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from ppocr.utils.dataset_cache import DatasetCache

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
        sample = dataset[i]
        if sample is not None:
            queue.append(sample)
    print("종료")
    

def main(config, device, logger, vdl_writer):

    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()

    
    global_config = config['Global']

    # build dataloader
    set_signal_handlers()
    
    



    
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    
    train_dataloader = build_dataloader(config, 'Eval', device, logger)
    size = len(train_dataloader.dataset)
    
    print(size)
    size = 10000
    worker_num = 5
    
    data_parts = chunk_list(list(range(size)), worker_num)
    
    dataset_cache = DatasetCache()
    
    processes = []

    for part in data_parts:
        p = multiprocessing.Process(target=load_data, args=(shared_list, part, train_dataloader.dataset))
        processes.append(p)
        p.start()

    # idx = 0        
    # while True:
    #     if len(shared_list) > idx:
    #         print(idx)
    #         sample, key = shared_list[idx]
    #         dataset_cache.save_samples_to_hdf5(sample, key)
    #         idx += 1
            
    # 모든 프로세스가 종료될 때까지 기다립니다.
    for p in processes:
        p.join()
    
    print(len(shared_list))
    print(shared_list[:10])
    for sample, key in tqdm.tqdm(shared_list):
        dataset_cache.save_samples_to_hdf5(sample, key)
        

if __name__ == '__main__':
    print("시작")
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    main(config, device, logger, vdl_writer)

