# -*- coding: utf-8 -*-
import os
import sys
import joblib
import argparse
import numpy as np
import cv2
import os.path as osp
import scipy.io as scio
from tqdm import tqdm

from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.penn_action_utils import calc_kpt_bound
from lib.models import spin
from lib.core.config import VIBE_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.feature_extractor import extract_features

sys.path.append('.')


def read_data(folder):
    dataset = {'img_name': [], 'joints2D': [], 'bbox': [], 'vid_name': [], 'features': [], }

    model = spin.get_pretrained_hmr()

    tot_frames = 0

    for vid in tqdm(range(575)):
        fname = osp.join(folder, '%d_info.mat' % vid)
        anns = scio.loadmat(fname)

        # Get frames from the videos
        video_cap = cv2.VideoCapture(osp.join(folder, 'syn_videos/%d.mp4' % vid))
        dir = '/content/syn_videos/frame%d' % vid
        if not os.path.exists(dir):
            os.makedirs(dir)
        nframes = 0
        img_paths = []
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            path = osp.join(dir, "%04d.jpg" % nframes)
            cv2.imwrite(path, frame)
            img_paths.append(path)
            nframes += 1

        kp_2d = anns['joints2D'].transpose((2, 1, 0))
        kp_2d = np.append(kp_2d, np.ones((kp_2d.shape[0], 24, 1)), axis=2)
        kp_2d = convert_kps(kp_2d, src="syn_videos", dst="spin").reshape((-1, 3))

        # Compute bounding box
        bbox = np.zeros((nframes, 4))
        for fr_id, fr in enumerate(kp_2d):
            print(fr.shape)
            print(fr)
            u, d, l, r = calc_kpt_bound(fr)
            center = np.array([(l + r) * 0.5, (u + d) * 0.5], dtype=np.float32)
            c_x, c_y = center[0], center[1]
            w, h = r - l, d - u
            w = h = np.where(w / h > 1, w, h)

            bbox[fr_id, :] = np.array([c_x, c_y, w, h])

        img_paths = np.array(img_paths)

        # Set up the dataset
        dataset['vid_name'].append(np.array([f'{vid}'] * img_paths.shape[0]))
        dataset['img_name'].append(np.array(img_paths))
        dataset['joints2D'].append(kp_2d)
        dataset['bbox'].append(np.array(bbox))

        # Compute features
        features = extract_features(model, np.array(img_paths), bbox, kp_2d=kp_2d, dataset='spin', debug=False, )

        assert kp_2d.shape[0] == img_paths.shape[0] == bbox.shape[0]
        tot_frames += nframes

        dataset['features'].append(features)

    print(tot_frames)
    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/syn_videos')
    args = parser.parse_args()

    dataset_train = read_data(args.dir)
    joblib.dump(dataset_train, osp.join(VIBE_DB_DIR, 'syn_videos_train_db.pt'))
