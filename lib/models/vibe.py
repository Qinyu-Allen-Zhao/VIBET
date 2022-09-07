# -*- coding: utf-8 -*-

import os
import torch
import os.path as osp
import torch.nn as nn

from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.relu = nn.ReLU()
        if add_linear:
            self.linear = nn.Linear(hidden_size * 2, 2048) if bidirectional else nn.Linear(hidden_size, 2048)
        else:
            self.linear = None

        self.use_residual = use_residual

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # Convert NTF to TNF
        y, _ = self.gru(x)
        if self.linear:
            y = self.relu(y)
            y = y.view(-1, y.size(-1))
            y = self.linear(y)
            y = y.view(t, n, f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x

        y = y.permute(1, 0, 2)  # Convert TNF to NTF
        return y


class VIBE(nn.Module):
    def __init__(
            self,
            seq_len,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            extract_features=False,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBE, self).__init__()

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        # From scratch to evaluate a video
        self.extract_features = extract_features
        checkpoint = torch.load(pretrained) if os.path.isfile(pretrained) else None

        if self.extract_features and os.path.isfile(pretrained):
            self.hmr = hmr()
            self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = checkpoint['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, input, J_regressor=None):
        # input size NTF
        if self.extract_features:
            batch_size, seq_len, nc, h, w = input.shape

            feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))
            feature = feature.reshape(batch_size, seq_len, -1)
            x = feature
        else:
            batch_size, seq_len = input.shape[:2]
            x = input

        feature = self.encoder(x)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seq_len, -1)
            s['verts'] = s['verts'].reshape(batch_size, seq_len, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seq_len, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seq_len, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seq_len, -1, 3, 3)

        return smpl_output
