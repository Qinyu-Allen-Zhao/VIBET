import os
import torch

from lib.dataset import ThreeDPW
from lib.models import VIBE
from lib.core.function import evaluate, validate
from lib.core.config import parse_args
from torch.utils.data import DataLoader


def main(cfg):
    print(f'...Evaluating on {} test set...')

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

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])
        print("Loaded pretrained model from {}".format(cfg.TRAIN.PRETRAINED))
        print('Performance on 3DPW test set '.format(best_performance))
    else:
        print('{} is not a pretrained model!!!!'.format(cfg.TRAIN.PRETRAINED))
        exit()

    test_db = ThreeDPW(set='test', seq_len=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    evaluation_accumulators = validate(model=model, device=cfg.DEVICE, test_loader=test_loader)
    evaluate(evaluation_accumulators)


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)
