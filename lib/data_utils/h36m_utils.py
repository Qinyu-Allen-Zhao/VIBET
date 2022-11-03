# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys

import h5py

sys.path.append('.')

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm

from lib.models import spin
from lib.data_utils.kp_utils import *
from lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.data_utils.feature_extractor import extract_features
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6

def read_data(folder, set, debug=False):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'valid': [],
    }

    # Set up models to extract features
    model = spin.get_pretrained_hmr()
    J_regressor = None
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    if set == 'test' or set == 'validation':
        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    # Get image names and annotations
    image_set = 'valid' if set == 'test' else 'train'
    file_name = os.path.join(
        folder, 'annot', '%s_images.txt' % image_set
    )
    with open(file_name, 'r') as file:
        img_names = [f.strip() for f in file.readlines()]

    anno_name = os.path.join(
        folder, 'annot', '%s.h5' % image_set
    )
    ann = h5py.File(anno_name, 'r')

    for i, img in tqdm(enumerate(img_names)):
        img_path = os.path.join(folder, 'images', img)

        # process bbox
        c = np.array(ann['center'][i], dtype=np.float)
        s = np.array([ann['scale'][i], ann['scale'][i]], dtype=np.float)
        bbox = np.vstack([c[0], c[1], s[0], s[1]]).T

        # process keypoints
        joints_2d = np.array(ann['part'][i])
        joints_2d = np.hstack([joints_2d, np.ones([17, 1])])
        joints_2d = convert_kps(joints_2d, 'h36m', 'spin').reshape((-1, 3))

        joints_3d_raw = np.reshape(ann['S'][0, :, :], (1, 17, 3)) / 1000
        joints_3d = convert_kps(joints_3d_raw, "h36m", "spin").reshape((-1, 3))
        # joints_3d = joints_3d - joints_3d[39]

        dataset['vid_name'].append(img[:-11])
        dataset['frame_id'].append(img)
        dataset['img_name'].append(img_path)
        dataset['joints2D'].append(joints_2d)
        dataset['joints3D'].append(joints_3d)
        dataset['bbox'].append(bbox)
        dataset['valid'].append(1)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
        print(k, dataset[k].shape)

    dataset['features'] = []
    j = 0
    for i in tqdm(range(len(dataset['frame_id']))):
        if dataset['vid_name'][i] != dataset['vid_name'][j]:
            features = extract_features(model, dataset['img_name'][j:i],
                                        dataset['bbox'][j:i],
                                        kp_2d=dataset['joints2D'][j:i],
                                        dataset='spin', debug=False)
            j = i
            dataset['features'].append(features)
    dataset['features'] = np.concatenate(dataset['features'])
    print('features', dataset['features'].shape)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/3dpw')
    args = parser.parse_args()

    debug = False

    dataset = read_data(args.dir, 'test', debug=debug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'h36m_test_db.pt'))

