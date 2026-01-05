#!/bin/bash
base_hdr_dir="hdri"
hdr_list=( "flower_road_no_sun_2k.hdr" \
           "lightroom_14b.hdr" \
           "pillars_2k.hdr" \
           "studio_small_02_2k.hdr" \
           "syferfontein_18d_clear_2k.hdr" \
           "the_sky_is_on_fire_2k.hdr" \
         )

for hdr_list_name in ${hdr_list[@]}; do
    hdr_dir="${base_hdr_dir}/${hdr_list_name}"
    log_dir="log_lucky_relight_${hdr_list_name}"
    nohup python render.py -m output/sam_black --eval --skip_train -w --save_name ${hdr_list_name%.*} --hdr_rotation --environment_texture $hdr_dir --render_relight > $log_dir 2>&1 &
    wait
done

echo "All tasks are finished!"