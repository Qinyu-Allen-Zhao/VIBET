# Adding the synthetic dataset
python train.py --cfg experiments/synthesis/train_with_syn.yaml
python eval.py --cfg experiments/synthesis/eval_with_syn.yaml

# Augment data by cutting
python train.py --cfg experiments/data_aug/train_cut.yaml
python eval.py --cfg experiments/data_aug/eval_cut.yaml

# Augment data by erasing
python train.py --cfg experiments/data_aug/train_erase.yaml
python eval.py --cfg experiments/data_aug/eval_erase.yaml