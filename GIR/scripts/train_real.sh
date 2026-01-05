#!/bin/bash 

nohup python train.py -s /mnt/disk1/codes/siggraph-exp/data/lucky_cat_indoor_high --eval --port 6100 --random_background -r 2 --hdr_rotation > log_lucky_cat_train.txt 2>&1 &

echo "All tasks are finished!" 