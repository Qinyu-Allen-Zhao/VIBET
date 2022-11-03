import os
import torch

from lib.dataset import ThreeDPW, MPII3D, SynVideos, ThreeDPWErase, ThreeDPWCut
from lib.models import VIBE, VIBET
from lib.core.function import evaluate, validate
from lib.core.config import parse_args
from torch.utils.data import DataLoader


def main(cfg):
    if cfg.MODEL.TEMPORAL_TYPE == 'gru':
        model = VIBE(
            n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            seq_len=cfg.DATASET.SEQLEN,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
            add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
            bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
            use_residual=cfg.MODEL.TGRU.RESIDUAL,
        ).to(cfg.DEVICE)
    elif cfg.MODEL.TEMPORAL_TYPE == 'transformer':
        model = VIBET(
            batch_size=cfg.TRAIN.BATCH_SIZE,
            seq_len=cfg.DATASET.SEQLEN,
            pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
            d_model=cfg.MODEL.TF.D_MODEL,
            nhead=cfg.MODEL.TF.NHEAD,
            num_layers=cfg.MODEL.TF.NUM_LAYERS,
        ).to(cfg.DEVICE)
    else:
        raise Exception()

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])
        print("Loaded pretrained model from {}".format(cfg.TRAIN.PRETRAINED))
        print('Performance on 3DPW test set '.format(best_performance))
    else:
        print('{} is not a pretrained model!!!!'.format(cfg.TRAIN.PRETRAINED))
        exit()

    for dataset in cfg.TEST.DATASETS:
        print(f'...Evaluating on {dataset} test set...')

        test_db = eval(dataset)(set='test', seq_len=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

        test_loader = DataLoader(
            dataset=test_db,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
        )

        evaluation_accumulators = validate(model=model, device=cfg.DEVICE, test_loader=test_loader)
        evaluate(evaluation_accumulators, dataset)


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)
