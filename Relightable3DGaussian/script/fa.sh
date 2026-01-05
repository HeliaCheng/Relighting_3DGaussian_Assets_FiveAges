# python train.py --eval \
#         -t render \
#         --model_path output/table/rendered \
#         --sample_num 128 \
#         --save_training_vis \
#         --save_training_vis_iteration 100 \
#         --iterations 16000 \
#         --lambda_depth_var 1e-2 \
#         --lambda_normal_render_depth 1 \
#         --lambda_normal_smooth 1 \
#         --lambda_mask_entropy 0.1 \
#         --lambda_depth_var 1e-2 \
#         -c output/table/rendered/chkpnt14000.pth \
        #        --is_ply True \

# python train.py --eval \
#         --model_path output/table/relighted \
#         --save_training_vis \
#         --position_lr_init 0 \
#         --position_lr_final 0 \
#         --normal_lr 0 \
#         --sh_lr 0 \
#         --opacity_lr 0.0 \
#         --scaling_lr 0.0 \
#         --rotation_lr 0.0 \
#         --iterations 35000 \
#         --lambda_base_color_smooth 1 \
#         --lambda_roughness_smooth 0.5 \
#         --lambda_light_smooth 1 \
#         --lambda_light 0.01 \
#         -t neilf --sample_num 128 \
#         --save_training_vis_iteration 100 \
#         --lambda_env_smooth 0.01\
#         -c output/table/rendered/chkpnt16000.pth

# python eval_nvs.py --eval \
#         --model_path output/table/relighted \
#         -c output/table/relighted/chkpnt23000.pth \
#         -t neilf

python relighting.py --sample_num 256 --ply_load "/home/chengwr/code/Relightable3DGaussian/output/table/relighted/iteration_35000.ply" --model_path "output/table/relighted" --video