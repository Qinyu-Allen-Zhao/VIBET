# -*- coding: utf-8 -*-

import os
import joblib
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm

from lib.core.config import VIBE_DB_DIR
from lib.data_utils.amass_utils import joints_to_use, all_sequences


class AmassReader:
    def __init__(self, sequences, output_folder):
        self.idx = 0
        self.sequences = sequences
        self.output_folder = output_folder

    def read_data(self, folder):
        for seq_name in self.sequences:
            print(f'Reading {seq_name} sequence...')
            seq_folder = osp.join(folder, seq_name)
            self.read_single_sequence(seq_folder, seq_name)

    def read_single_sequence(self, folder, seq_name, fps=25):
        subjects = os.listdir(folder)

        for subject in tqdm(subjects):
            actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

            for action in actions:
                fname = osp.join(folder, subject, action)

                if fname.endswith('shape.npz'):
                    continue

                data = np.load(fname)

                mocap_framerate = int(data['mocap_framerate'])
                sampling_freq = mocap_framerate // fps
                pose = data['poses'][0::sampling_freq, joints_to_use]

                if pose.shape[0] < 60:
                    continue

                shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
                theta = np.concatenate([pose, shape], axis=1)
                vid_name = f'{seq_name}_{subject}_{action[:-4]}'

                joblib.dump(theta, osp.join(self.output_folder, vid_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/amass')
    parser.add_argument('--output', type=str, help='output directory', default='output/')
    args = parser.parse_args()

    reader = AmassReader(all_sequences, args.output)
    reader.read_data(args.dir)
