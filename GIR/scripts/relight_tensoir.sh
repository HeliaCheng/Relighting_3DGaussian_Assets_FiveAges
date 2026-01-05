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

hdr_list=( "bridge.hdr" 
           "city.hdr"
           "fireplace.hdr"
           "forest.hdr"
           "night.hdr"
         )
  
# Base directory  
base_dir="output" 
base_hdr_dir="high_res_envmaps_1k"  
  
# Generate the bash commands for each GPU  

for hdr_list_name in ${hdr_list[@]}; do
    idx=1
    hdr_dir="${base_hdr_dir}/${hdr_list_name}"
    while [ $idx -le ${#scan_list[@]} ]; do
        for ((i=0; i<$GPU_COUNT; i++))  do  
            scan_name=${scan_list[$idx-1]}  
            model_dir="${base_dir}/${scan_name}"
            log_dir="log_${scan_name}_relight_${hdr_list_name}"
            CUDA_VISIBLE_DEVICES=$i nohup python render.py -m $model_dir --skip_train -w --save_name ${hdr_list_name%.*} --hdr_rotation --environment_texture $hdr_dir --render_relight > $log_dir 2>&1 &
            idx=$(($idx+1))
            if [ $idx -eq $(( ${#scan_list[@]} + 1)) ]; then 
                break 
            fi
        done
        wait
    done
done
echo "All tasks are finished!" 