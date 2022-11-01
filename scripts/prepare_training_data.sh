#!/usr/bin/env bash

mkdir -p ./data/vibe_db
export PYTHONPATH="./:$PYTHONPATH"

## AMASS
#python lib/data_utils/amass_utils.py --dir /home/qinyu/datasets/amass

## InstaVariety
## Comment this if you already downloaded the preprocessed file
#python lib/data_utils/insta_utils.py --dir ./data/insta_variety

## 3DPW
#python lib/data_utils/threedpw_utils.py --dir /home/qinyu/datasets/3dpw

## MPI-INF-3D-HP
python lib/data_utils/mpii3d_utils.py --dir /home/qinyu/datasets/mpi_inf_3dhp
#
## PoseTrack
#python lib/data_utils/posetrack_utils.py --dir /home/qinyu/datasets/posetrack
#
## PennAction
#python lib/data_utils/penn_action_utils.py --dir /home/qinyu/datasets/penn_action

# Synthesis videos
#python lib/data_synthesis/syn_videos_process.py --dir /home/qinyu/datasets/syn_videos
