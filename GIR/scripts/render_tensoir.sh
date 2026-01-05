#!/bin/bash  
  
# Get the number of available GPUs  
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)  
echo "GPU Count: $GPU_COUNT" 

# List of tasks  
scan_list=( "armadillo"  
            "ficus"  
            "hotdog"  
            "lego"
            )  
  
# Base directory  
base_dir="output" 
  
# Generate the bash commands for each GPU  
idx=1
while [ $idx -le ${#scan_list[@]} ]; do
    for ((i=0; i<$GPU_COUNT; i++))  do  
        scan_name=${scan_list[$idx-1]}  
        model_dir="${base_dir}/${scan_name}"
        log_dir="log_${scan_name}_render"
        CUDA_VISIBLE_DEVICES=$i nohup python render.py -m $model_dir --skip_train --save_name "render" -w --hdr_rotation > $log_dir 2>&1 &
        idx=$(($idx+1))
        if [ $idx -eq $(( ${#scan_list[@]} + 1)) ]; then 
            break 
        fi
    done
    wait
done
echo "All tasks are finished!" 