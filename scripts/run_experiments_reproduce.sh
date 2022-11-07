# Training the vibe model with 3DPW
python train.py --cfg experiments/reproduce/config_w_3dpw.yaml
python eval.py --cfg experiments/reproduce/evaluate_ori.yaml

# Training the vibe model without 3DPW
python train.py --cfg experiments/reproduce/config_wo_3dpw.yaml
python eval.py --cfg experiments/reproduce/evaluate_ori_wo_3dpw.yaml
