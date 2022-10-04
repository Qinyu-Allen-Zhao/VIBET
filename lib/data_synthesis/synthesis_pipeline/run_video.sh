#!/bin/bash

# SET PATHS HERE
BLENDER_PATH=/root/blenderpath

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.90/python
# export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.4:${BUNDLED_PYTHON}/lib/python3.4/site-packages
# export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

for ((idx = 330; idx < 40000; idx++ )) # 40000
do
    # Main part 1
    $BLENDER_PATH/blender -b -t 1 -P main_part1-video.py -- --idx $idx --use_split train

    # Main part 2
#     python main_part2.py -- --idx $idx
done
