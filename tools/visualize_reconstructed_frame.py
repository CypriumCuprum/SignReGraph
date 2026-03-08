import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import mmcv
from mmcv import Config
from mmcv import load
from mmcv.runner import load_checkpoint
from .helper.draw_keypoint import compare_skeletons_to_file

try:
    from signregraph.datasets import build_dataset
    from signregraph.models import build_model
    from signregraph.utils import cache_checkpoint
    from signregraph.datasets.pipelines import Compose
except ImportError:
    print("Errors: can not import 'signregraph'")
    exit(1)
# ----------------------------------------------------

import time
import torch.nn as nn
import sys
import numpy as np


CONNECTIVITY = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (7, 8), 
                           (8, 9), (9, 10), (6, 11), (11, 12), (12, 13), (13, 14), 
                           (6, 15), (15, 16), (16, 17), (17, 18), (6, 19), (19, 20), 
                           (20, 21), (21, 22), (6, 23), (23, 24), (24, 25), (25, 26), 
                           (27, 28), (28, 29), (29, 30), (30, 31), (27, 32), (32, 33), 
                           (33, 34), (34, 35), (27, 36), (36, 37), (37, 38), (38, 39), 
                           (27, 40), (40, 41), (41, 42), (42, 43), (27, 44), (44, 45), 
                           (45, 46), (46, 47)] + \
                            [(i, i+1) for i in range(48, 48 + 17 - 1)] + \
                            [(i, i+1) for i in range(48+17, 48+17+12 - 1)] + [(48+17+12 -1, 48+17)]

def extract_features(cfg: Config, args):
    """
    Extract features from backbone for one instance.
    model.eval() for avoiding errror BatchNorm.
    """
    data_id = args.input_file
    device = args.device
    checkpoint_path = args.checkpoint
    model = build_model(cfg.model) 
    checkpoint_path = cache_checkpoint(checkpoint_path)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    model = model.to(device)
    model.eval() 
    
    ann_data = load(cfg.data.train.ann_file)

    target_data_list = ann_data['split']['train']
    if not target_data_list:
        print("Error: can view any data in train split")
        return

    target_sample = None
    for item in ann_data['annotations']:
        if item['frame_dir'] == data_id:
            target_sample = item
    target_sample['start_index'] = 0
            
    if target_sample is None:
        raise ValueError(f"Not fount ID: {data_id} in {cfg.data.train.ann_file} (split 'train')")
    
    pipeline_cfg = cfg.data.train.pipeline
    target_sample['test_mode'] = False
    pipeline = Compose(pipeline_cfg)

    data = pipeline(target_sample)
    
    if data is None:
        raise Exception("Pipeline return None")

    input_data_dict = {}
    for key in data.keys():
        val = data[key]
        if isinstance(val, torch.Tensor):
            # Shape (NC, M, T, V, C) -> (B, NC, M, T, V, C)
            input_data_dict[key] = val.to(device).unsqueeze(0) 
        elif isinstance(val, (int, float, str, list)):
             input_data_dict[key] = [val] 
        else:
             input_data_dict[key] = val
    
    print("Đang chạy forward propagation (model.eval(), model.extract_feat)...")
    
    keypoint_tensor = input_data_dict['keypoint'] # Shape (B, NC, M, T, V, C)
    
    with torch.no_grad():
        bs, nc = keypoint_tensor.shape[:2] 
        keypoint_input_for_backbone = keypoint_tensor.reshape((bs * nc, ) + keypoint_tensor.shape[2:])
        features_flat, hand_recon_flat = model.extract_feat(keypoint_input_for_backbone)
    


    features = features_flat.reshape((bs, nc) + features_flat.shape[1:])
    hand_recon = hand_recon_flat.reshape((bs, nc) + hand_recon_flat.shape[1:])
    
    hand_squeeze = hand_recon.squeeze().squeeze()
    drop_area = data['drop_area']
    
    recontructed_kp_for_draw = hand_squeeze.T.tolist()
    drop_area_kp_for_draw = drop_area.T.tolist()
    
    output_filename = os.path.join('./draft', 'figs', args.model_type, f"{data_id}.png")
    os.makedirs(os.path.join("./draft", "figs", args.model_type), exist_ok=True)
    compare_skeletons_to_file(recontructed_kp_for_draw, drop_area_kp_for_draw, CONNECTIVITY, output_filename=output_filename) 
    
    return {
        'features': features.cpu().numpy(),
        'hand_reconstruction': hand_recon.cpu().numpy()
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description='signregraph feature extraction script for a single sample')
    parser.add_argument('config', help='config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--input-file', 
        type=str, 
        default="65097",
        help='ID of one instance.'
    )
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='cuda:0 or cpu'
    )
    parser.add_argument(
        '--model-type', 
        default='test', 
        help='Model type'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    checkpoint = args.checkpoint
    if checkpoint is None:
        work_dir = cfg.get('work_dir', './') 
        checkpoint = osp.join(work_dir, 'latest.pth')
        print(f"No checkpoint, automatically using: {checkpoint}")
        
    assert osp.exists(checkpoint), f"Not Found checkpoint. Let add through -C or file 'latest.pth' must be exist"

    print(f"Config:     {args.config}")
    print(f"heckpoint: {checkpoint}")
    print(f"Device:    {args.device}")
    print(f"Instance ID:  {args.input_file}\n")
    print(f"Save to: {args.model_type}\n")
    
    try:
        result = extract_features(cfg, args)
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()