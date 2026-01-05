#!/bin/bash  
  
# Get the number of available GPUs  
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)  
echo "GPU Count: $GPU_COUNT" 
  
# List of tasks  
scan_list=( "toaster"  
            "car"  
            "helmet"  
            "coffee"  
            "teapot"  
            "ball"  
            )  
  
# Base directory  
base_dir="/mnt/disk1/codes/siggraph-exp/data/refnerf" 
  
# Generate the bash commands for each GPU  
idx=1
while [ $idx -le ${#scan_list[@]} ]; do
    for ((i=0; i<$GPU_COUNT; i++))  do  
        scan_name=${scan_list[$idx-1]}  
        data_dir="${base_dir}/${scan_name}"
        log_dir="log_${scan_name}_train"
        port_num=$(($i+6100))
        if [ "$scan_name" == "ball" ]  
        then  
            CUDA_VISIBLE_DEVICES=$i nohup python train.py -s $data_dir --eval --port $port_num --second_stage_step 100000 -w --hdr_rotation > $log_dir 2>&1 &
        else 
            CUDA_VISIBLE_DEVICES=$i nohup python train.py -s $data_dir --eval --port $port_num --random_background --hdr_rotation > $log_dir 2>&1 &
        fi  
        idx=$(($idx+1))
        if [ $idx -eq $(( ${#scan_list[@]} + 1)) ]; then 
            break 
        fi
    done
    wait
done
echo "All tasks are finished!" 