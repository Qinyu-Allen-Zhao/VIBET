import sys

from lib.data_aug.augment_utils import cut_augmentation
from lib.data_utils.penn_action_utils import load_mat, calc_kpt_bound

sys.path.append('.')

import glob
import joblib
import argparse
from tqdm import tqdm
import os.path as osp

from lib.models import spin
from lib.data_utils.kp_utils import *
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.feature_extractor import extract_features


def read_data(folder, aug_methods='cut'):
    dataset = {
        'img_name': [],
        'joints2D': [],
        'bbox': [],
        'vid_name': [],
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    file_names = sorted(glob.glob(folder + '/labels/' + '*.mat'))

    for fname in tqdm(file_names):
        vid_dict = load_mat(fname)
        imgs = sorted(glob.glob(folder + '/frames/' + fname.strip().split('/')[-1].split('.')[0] + '/*.jpg'))
        kp_2d = np.zeros((vid_dict['nframes'], 13, 3))
        perm_idxs = get_perm_idxs('pennaction', 'common')

        kp_2d[:, :, 0] = vid_dict['x']
        kp_2d[:, :, 1] = vid_dict['y']
        kp_2d[:, :, 2] = vid_dict['visibility']
        kp_2d = kp_2d[:, perm_idxs, :]

        # fix inconsistency
        n_kp_2d = np.zeros((kp_2d.shape[0], 14, 3))
        n_kp_2d[:, :12, :] = kp_2d[:, :-1, :]
        n_kp_2d[:, 13, :] = kp_2d[:, 12, :]
        kp_2d = n_kp_2d

        bbox = np.zeros((vid_dict['nframes'], 4))

        for fr_id, fr in enumerate(kp_2d):
            u, d, l, r = calc_kpt_bound(fr)
            center = np.array([(l + r) * 0.5, (u + d) * 0.5], dtype=np.float32)
            c_x, c_y = center[0], center[1]
            w, h = r - l, d - u
            w = h = np.where(w / h > 1, w, h)

            bbox[fr_id, :] = np.array([c_x, c_y, w, h])

        if aug_methods == 'cut':
            bbox = cut_augmentation(bbox)

        dataset['vid_name'].append(np.array([f'{fname}'] * vid_dict['nframes']))
        dataset['img_name'].append(np.array(imgs))
        dataset['joints2D'].append(kp_2d)
        dataset['bbox'].append(bbox)

        features = extract_features(model, np.array(imgs), bbox,
                                    dataset='pennaction', debug=False, random_mask=aug_methods == 'erase')
        dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/pennaction')
    parser.add_argument('--aug', type=str, help='augmentation method', default='cut')
    args = parser.parse_args()

    dataset = read_data(args.dir, args.aug)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'pennaction_train_db_aug.pt'))
