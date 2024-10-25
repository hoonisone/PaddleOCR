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

from click import pass_obj
import numpy as np

import os
import sys
import json
from pathlib import Path
import numpy as np


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program


from ppocr.utils.korean_grapheme_label import compose_korean_char, _compose_korean_char
def main():
    global_config = config['Global']

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)


    config["Global"]["char_num"] = post_process_class.char_num # 기본적으로 int 값이나 Grapheme 알고리즘에서는 dict이다. {grapheme: num}
    config["Architecture"]["Head"]["out_channels"] = post_process_class.char_num
    
    
    # if config['PostProcess']["name"] in ["ABINetLabelDecode_GraphemeLabel", "ABINetLabelDecode_GraphemeLabel_B", "ABINetLabelDecode_GraphemeLabel_All"]:
    #     class_num_dict = post_process_class.class_num_dict
    #     config["Global"]["class_num_dict"] = class_num_dict

    # # build model
    # if hasattr(post_process_class, 'character'):
    #     # char_num = len(getattr(post_process_class, 'character'))
    #     character = getattr(post_process_class, 'character')
    #     if "use_grapheme" in global_config and global_config["use_grapheme"]:    
    #         char_num= np.array([len(character[grapheme]) for grapheme in global_config["handling_grapheme"]])
    #         # char_num = np.array([len(x) for x in character])

    #     else:
    #         char_num = len(character)
    #     if config["Architecture"]["algorithm"] in ["Distillation",
    #                                                ]:  # distillation model
    #         for key in config["Architecture"]["Models"]:
    #             if config["Architecture"]["Models"][key]["Head"][
    #                     "name"] == 'MultiHead':  # multi head
    #                 out_channels_list = {}
    #                 if config['PostProcess'][
    #                         'name'] == 'DistillationSARLabelDecode':
    #                     char_num = char_num - 2
    #                 if config['PostProcess'][
    #                         'name'] == 'DistillationNRTRLabelDecode':
    #                     char_num = char_num - 3
    #                 out_channels_list['CTCLabelDecode'] = char_num
    #                 out_channels_list['SARLabelDecode'] = char_num + 2
    #                 out_channels_list['NRTRLabelDecode'] = char_num + 3
    #                 config['Architecture']['Models'][key]['Head'][
    #                     'out_channels_list'] = out_channels_list
    #             else:
    #                 config["Architecture"]["Models"][key]["Head"][
    #                     "out_channels"] = char_num
    #     elif config['Architecture']['Head']['name'] in ['MultiHead', 'MultiHead_Grapheme']:
    #         out_channels_list = {}
    #         # char_num = len(getattr(post_process_class, 'character'))
    #         if config['PostProcess']['name'] == 'SARLabelDecode':
    #             char_num = char_num - 2
    #         if config['PostProcess']['name'] == 'NRTRLabelDecode':
    #             char_num = char_num - 3
    #         out_channels_list['CTCLabelDecode'] = char_num
    #         out_channels_list['SARLabelDecode'] = char_num + 2
    #         out_channels_list['NRTRLabelDecode'] = char_num + 3
    #         config['Architecture']['Head'][
    #             'out_channels_list'] = out_channels_list
    #         # print(out_channels_list)
    #         # exit()
    #     else:  # base rec model
    #         config["Architecture"]["Head"]["out_channels"] = char_num
    model = build_model(config['Architecture'], **global_config)

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif config['Architecture']['algorithm'] == "RobustScanner":
                op[op_name][
                    'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global'].get('save_res_path',
                                         "./output/rec/predicts_rec.txt")
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()




    with open(save_res_path, "w") as fout:
        #################################################### MH Modification Start
        infer_paths = []
        dataset_dir = Path(config['Infer']["data_dir"])
        for infor_file in config['Infer']['infer_file_list']:    
            with open(infor_file, "r") as f:
                paths = [str(dataset_dir/line.strip("\n")) for line in f.readlines()]                
            infer_paths += paths
        for file in infer_paths:
        # for file in get_image_file_list(config['Global']['infer_img']):
        ####################################################
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            if config['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    paddle.to_tensor(encoder_word_pos_list),
                    paddle.to_tensor(gsrm_word_pos_list),
                    paddle.to_tensor(gsrm_slf_attn_bias1_list),
                    paddle.to_tensor(gsrm_slf_attn_bias2_list)
                ]
            if config['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio)]
            if config['Architecture']['algorithm'] == "RobustScanner":
                valid_ratio = np.expand_dims(batch[1], axis=0)
                word_positons = np.expand_dims(batch[2], axis=0)
                img_metas = [
                    paddle.to_tensor(valid_ratio),
                    paddle.to_tensor(word_positons),
                ]
            if config['Architecture']['algorithm'] == "CAN":
                image_mask = paddle.ones(
                    (np.expand_dims(
                        batch[0], axis=0).shape), dtype='float32')
                label = paddle.ones((1, 36), dtype='int64')
            images = np.expand_dims(batch["image"], axis=0)
            images = paddle.to_tensor(images)
            if config['Architecture']['algorithm'] == "SRN":
                preds = model(images, others)
            elif config['Architecture']['algorithm'] == "SAR":
                preds = model(images, img_metas)
            elif config['Architecture']['algorithm'] == "RobustScanner":
                preds = model(images, img_metas)
            elif config['Architecture']['algorithm'] == "CAN":
                preds = model([images, image_mask, label])
            else:
                preds = model(images)
            if "grapheme" in config["Global"]:
            
                preds_args = {name: preds.get(name, None) for name in config["Global"]["grapheme"]}
                # labels_args = {f"{name}_label": batch[f"{name}_label"]["label_ctc"] for name in config["Global"]["grapheme"]}
                

                
                post_result = post_process_class(preds_args, label = None)

            else:
                label = batch["label"] if "label" in batch else None
                post_result = post_process_class(preds, label)


            info = None
               
            
            if isinstance(post_result, dict):
                info = json.dumps(post_result, ensure_ascii=False)
            elif isinstance(post_result, list) and isinstance(post_result[0], int):
                # for RFLearning CNT branch 
                info = str(post_result[0])
            else:
                info={k:v for k, v in post_result[0].items()}
            

            if info is not None:
                
                logger.info("\t result: {}".format(str(info)))
                fout.write(file + "\t" + str(info) + "\n")
    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
        
    main()
