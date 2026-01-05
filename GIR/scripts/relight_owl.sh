#!/bin/bash  
  
# Get the number of available GPUs  
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)  
echo "GPU Count: $GPU_COUNT" 

# List of tasks  
scan_list=( "antman_blender"  
            "apple_blender"  
            "chest_blender"  
            "gamepad_blender"  
            "ping_pong_blender"  
            "porcelain_mug_blender"  
            "tpiece_blender"  
            "wood_bowl_blender"  
            )  

hdr_list=( "gt_env_512_rotated_0000.hdr" 
           "gt_env_512_rotated_0001.hdr"
           "gt_env_512_rotated_0002.hdr"
           "gt_env_512_rotated_0003.hdr"
           "gt_env_512_rotated_0004.hdr"
           "gt_env_512_rotated_0005.hdr"
           "gt_env_512_rotated_0006.hdr"
           "gt_env_512_rotated_0007.hdr"
           "gt_env_512_rotated_0008.hdr"
         )
  
# Base directory  
base_dir="output" 
base_hdr_dir="owl"  
  
# Generate the bash commands for each GPU  

for hdr_list_name in ${hdr_list[@]}; do
    idx=1
    hdr_dir="${base_hdr_dir}/${hdr_list_name}"
    while [ $idx -le ${#scan_list[@]} ]; do
        for ((i=0; i<$GPU_COUNT; i++))  do  
            scan_name=${scan_list[$idx-1]}  
            model_dir="${base_dir}/${scan_name}"
            log_dir="log_${scan_name}_relight_${hdr_list_name}.log"
            CUDA_VISIBLE_DEVICES=$i nohup python render.py -m $model_dir --skip_train -w --save_name ${hdr_list_name%.*} --environment_texture $hdr_dir --hdr_rotation --render_relight > $log_dir 2>&1 &
            idx=$(($idx+1))
            if [ $idx -eq $(( ${#scan_list[@]} + 1)) ]; then 
                break 
            fi
        done
        wait
    done
done
echo "All tasks are finished!" 