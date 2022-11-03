python train.py --cfg experiments/reproduce/config_w_3dpw.yaml
python eval.py --cfg experiments/reproduce/evaluate_ori.yaml

python train.py --cfg experiments/transformer/train_with_tf_simple.yaml
python eval.py --cfg experiments/transformer/eval_with_tf_simple.yaml

python lib/data_utils/h36m_utils.py --dir /home/qinyu/datasets/h36m
