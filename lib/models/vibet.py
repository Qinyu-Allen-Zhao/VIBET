# -*- coding: utf-8 -*-
import math
import os
import torch
import os.path as osp
import torch.nn as nn

from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size()[0], :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            seq_len=32,
            d_model=512,
            nhead=8,
            num_layers=6
    ):
        super(TemporalEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)  # => (seq, batch, feature)

        out = self.pos_encoder(x)
        out = self.transformer_encoder(out)
        out += x  # residual learning

        out = out.permute(1, 0, 2)
        return out


class SpatialEncoder(nn.Module):
    def __init__(
            self,
            input_size=256,
            hidden_layer=256,
            num_layers=3,
            output_size=64,
    ):
        super(SpatialEncoder, self).__init__()
        self.num_layers = num_layers

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_layer, bias=True),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_layer),
        )

        self.block = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer, bias=True),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_layer),
        )

        self.output_layer = nn.Linear(hidden_layer, output_size, bias=True)

    def forward(self, x):
        x = self.layers(x)
        for _ in range(self.num_layers):
            res = self.block(x)
            x += res

        x = self.output_layer(x)
        return x


class VIBET(nn.Module):
    def __init__(
            self,
            seq_len,
            batch_size=64,
            d_model=512,
            nhead=8,
            num_layers=6,
            extract_features=False,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBET, self).__init__()

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
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

            frame_feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))
            frame_feature = frame_feature.reshape(batch_size, seq_len, -1)
            x = frame_feature
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
