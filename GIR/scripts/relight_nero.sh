#!/bin/bash  
  
# Get the number of available GPUs  
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)  
echo "GPU Count: $GPU_COUNT" 

# List of tasks  
scan_list=( "angel_blender"  
            "bell_blender"  
            "cat_blender"  
            "horse_blender"  
            "luyu_blender"  
            "potion_blender"  
            "tbell_blender"  
            "teapot_blender"  
            )  

hdr_list=( "flower_road_no_sun_2k.hdr" 
           "lightroom_14b.hdr"
           "pillars_2k.hdr"
           "studio_small_02_2k.hdr"
           "syferfontein_18d_clear_2k.hdr"
           "the_sky_is_on_fire_2k.hdr"
         )
  
# Base directory  
base_dir="output" 
base_hdr_dir="hdri"  
  
# Generate the bash commands for each GPU  

for hdr_list_name in ${hdr_list[@]}; do
    idx=1
    hdr_dir="${base_hdr_dir}/${hdr_list_name}"
    while [ $idx -le ${#scan_list[@]} ]; do
        for ((i=0; i<$GPU_COUNT; i++))  do  
            scan_name=${scan_list[$idx-1]}  
            model_dir="${base_dir}/${scan_name}"
            log_dir="log_${scan_name}_relight_${hdr_list_name}"
            CUDA_VISIBLE_DEVICES=$i nohup python render.py -m $model_dir --skip_train -w --save_name ${hdr_list_name%.*} --environment_texture $hdr_dir --render_relight > $log_dir 2>&1 &
            idx=$(($idx+1))
            if [ $idx -eq $(( ${#scan_list[@]} + 1)) ]; then 
                break 
            fi
        done
        wait
    done
done
echo "All tasks are finished!" 