#!/bin/bash

root_dir="./Synthetic4Relight/"
# list="air_baloons chair hotdog jugs"
list="air_baloons"
for i in $list
do
    # python train_ori.py --eval \
    #     -s ${root_dir}${i} \
    #     --model_path output/Syn4Relight/${i}/3dgs \
    #     --lambda_normal_render_depth 0.01 \
    #     --lambda_normal_smooth 0.02 \
    #     --lambda_mask_entropy 0.1 \
    #     --save_training_vis \
    #     --densify_grad_normal_threshold 1e-8 \
    #     --lambda_depth_var 1e-2\
    #     -t render
        

    # python eval_nvs.py --eval \
    #     --model_path output/Syn4Relight/${i}/3dgs \
    #     -c output/Syn4Relight/${i}/3dgs/chkpnt20000.pth\
    #     -t render


#     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5remove from below
#     ##########-c output/Syn4Relight/${i}/neilf/chkpnt50000.pth \

    # python train_ori.py --eval \
    #     -s ${root_dir}${i} \
    #     -m output/Syn4Relight/${i}/neilf \
    #     -c output/Syn4Relight/${i}/3dgs/chkpnt20000.pth \
    #     --save_training_vis \
    #     --position_lr_init 0 \
    #     --position_lr_final 0 \
    #     --normal_lr 0 \
    #     --sh_lr 0 \
    #     --opacity_lr 0 \
    #     --scaling_lr 0 \
    #     --rotation_lr 0 \
    #     --iterations 30000 \
    #     --lambda_base_color_smooth 1 \
    #     --lambda_roughness_smooth 0.5 \
    #     --lambda_light_smooth 1 \
    #     --lambda_light 0.01 \
    #     -t neilf --sample_num 64 \
    #     --save_training_vis_iteration 200 \
    #     --lambda_env_smooth 0.01
    
    # python eval_nvs.py --eval \
    #      --model_path output/Syn4Relight/${i}/neilf \
    #      -c output/Syn4Relight/${i}/neilf/chkpnt30000.pth \
    #      -t neilf

    python eval_relighting_syn4.py \
        -m output/Syn4Relight/air_baloons/neilf \
        -c output/Syn4Relight/air_baloons/neilf/chkpnt30000.pth \
        --sample_num 256
done