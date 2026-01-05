#!/bin/bash

nohup python render.py -m output/lucky_cat -w --save_name render --skip_train --hdr_rotation > log_lucky_cat_render.txt 2>&1 &

echo "All tasks are finished!"