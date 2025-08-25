"""
run on 187

cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python data_process/prepare_pc_cond_sample/prepare_pc_cond_feature.py \
    --sampled_pc_root /data/AIGP/pc_cond_sample_multitype/ \
    --pointcloud_encoder POINT_E \
    --device cuda:0
"""

import os
from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from src.network import PointcloudEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampled_pc_root', type=str, default='/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/data_process/prepare_pc_cond_sample/output_multitype')
    parser.add_argument('--pointcloud_encoder', type=str, default='POINT_E')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device
    sampled_pc_root = "/data/AIGP/pc_cond_sample_multitype/" # args.sampled_pc_root
    pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, "cuda:0")

    sampled_pc_dirs = glob(os.path.join(sampled_pc_root,"**","sampled_pc"),recursive=True)
    for sampled_dir in tqdm(sampled_pc_dirs):
        fp_list = glob(os.path.join(sampled_dir,"*.npy"))
        for fp in fp_list:
            output_fp = fp.replace("sampled_pc", f"feature_{args.pointcloud_encoder}")

            if os.path.exists(output_fp):
                continue

            os.makedirs(os.path.dirname(output_fp), exist_ok=True)

            pc_sampled = np.load(fp)
            feature = pointcloud_encoder(pc_sampled)
            np.save(output_fp, feature)