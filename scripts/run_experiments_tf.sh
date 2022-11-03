# Trainsformer 1 nhead 6 layers
#python train.py --cfg experiments/transformer/train_with_tf_1H_6L.yaml
python eval.py --cfg experiments/transformer/eval_with_tf_1H_6L.yaml

# Trainsformer 4 nheads 6 layers
#python train.py --cfg experiments/transformer/train_with_tf_4H_6L.yaml
python eval.py --cfg experiments/transformer/eval_with_tf_4H_6L.yaml

# Trainsformer 8 nheads 6 layers
#python train.py --cfg experiments/transformer/train_with_tf_8H_6L.yaml
python eval.py --cfg experiments/transformer/eval_with_tf_8H_6L.yaml

# Trainsformer 16 nheads 6 layers
#python train.py --cfg experiments/transformer/train_with_tf_16H_6L.yaml
python eval.py --cfg experiments/transformer/eval_with_tf_16H_6L.yaml