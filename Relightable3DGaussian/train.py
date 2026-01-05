import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import GaussianModel,SceneFromGaussians
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from gui import GUI
from scene.direct_light_map import DirectLightMap
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
from scene.utils import save_render_orb, save_depth_orb, save_normal_orb, save_albedo_orb, save_roughness_orb
import imageio.v2 as iio
import cv2
from plyfile import PlyData
from utils.camera_utils import load_cam_img,camlist_from_infos,sorted_files_in_order
import imageio
import OpenEXR
import Imath
from utils.loss_utils import masked_l1,masked_lpips,masked_psnr,masked_ssim
# if torch.cuda.is_available():
#     # 低版本PyTorch设置默认张量类型为GPU（兼容所有版本）
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     # 绑定默认GPU卡（可选，多GPU场景指定0卡）
#     torch.cuda.set_device(0)
#     print("✅ 默认张量类型已改为 GPU (cuda:0)")
# else:
#     raise RuntimeError("❌ CUDA不可用，无法设置GPU默认张量类型")
# if torch.cuda.is_available():
#     print(f"GPU名称: {torch.cuda.get_device_name(0)}")
#     print(f"GPU算力架构: {torch.cuda.get_device_capability(0)}")
def training_report(tb_writer, iteration, tb_dict, scene: SceneFromGaussians, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=2.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False,dict_params=None,**kwargs):
    
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training, dict_params,
                                            **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)

                    # BRDF
                    base_color = torch.clamp(render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0)
                    roughness = torch.clamp(render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0)
                    image_pbr = render_pkg.get("pbr", torch.zeros_like(image))

                    grid = torchvision.utils.make_grid(
                        torch.stack([image, image_pbr, gt_image,
                                     opacity.repeat(3, 1, 1), depth.repeat(3, 1, 1), normal,
                                     base_color, roughness.repeat(3, 1, 1)], dim=0), nrow=3)

                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)
                    
                    
                    
                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()
                   

                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

        torch.cuda.empty_cache()

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs):
    os.makedirs(os.path.join(args.model_path, "visualize",f"ours_{opt.iterations}"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,scaling_modifier=2.0,
                                   opt=opt, is_training=False, dict_params=pbr_kwargs)

            visualization_list = [
                render_pkg["render"],
                viewpoint_cam.original_image.cuda(),
                visualize_depth(render_pkg["depth"]),
                (render_pkg["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1),
                render_pkg["opacity"].repeat(3, 1, 1),
                render_pkg["normal"] * 0.5 + 0.5,
                render_pkg["pseudo_normal"] * 0.5 + 0.5,
            ]

            if is_pbr:
                
                H, W = render_pkg["pbr"].shape[1:]
                env = F.interpolate(render_pkg['env'].permute(0, 3, 1, 2), (H, 2*W))
                env_0 = env[0, :, :, :W]
                env_1 = env[0, :, :, W:]
                visualization_list.extend([
                    render_pkg["base_color"],
                    render_pkg["roughness"].repeat(3, 1, 1),
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["diffuse"],
                    # render_pkg["lights"],
                    render_pkg["specular"],
                    # render_pkg["local_lights"],
                    render_pkg["global_lights"],
                    render_pkg["pbr"],
                    rgb_to_srgb(env_0),
                    rgb_to_srgb(env_1),
                ])

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            scale = grid.shape[-2] / 800
            grid = F.interpolate(grid[None], (int(grid.shape[-2]/scale), int(grid.shape[-1]/scale)))[0]
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))

def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
    if gaussians.use_pbr:
        os.makedirs(os.path.join(args.model_path, 'eval', 'base_color'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'lights'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'local'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'global'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'visibility'), exist_ok=True)

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, scaling_modifier=2.0 ,opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            if gaussians.use_pbr:
                image = results["pbr"]
            else:
                image = results["render"]

            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            
           
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()

            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}.png"))
            save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}.png"))
            save_image(results["normal"] * 0.5 + 0.5,
                       os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}.png"))
            if gaussians.use_pbr:
                save_image(results["base_color"],
                           os.path.join(args.model_path, 'eval', "base_color", f"{viewpoint.image_name}.png"))
                save_image(results["roughness"],
                           os.path.join(args.model_path, 'eval', "roughness", f"{viewpoint.image_name}.png"))
                save_image(results["lights"],
                           os.path.join(args.model_path, 'eval', "lights", f"{viewpoint.image_name}.png"))
                save_image(results["local_lights"],
                           os.path.join(args.model_path, 'eval', "local", f"{viewpoint.image_name}.png"))
                save_image(results["global_lights"],
                           os.path.join(args.model_path, 'eval', "global", f"{viewpoint.image_name}.png"))
                save_image(results["visibility"],
                           os.path.join(args.model_path, 'eval', "visibility", f"{viewpoint.image_name}.png"))

    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))

def training_new(is_ply:True,opt: OptimizationParams,pipe: PipelineParams,dataset:ModelParams,is_pbr=False):
     
    # ply_path="/root/autodl-tmp/Relightable3DGaussian/FA_scene/newoffice_point_cloud_slimmed_new_zero_corner_position.ply"
    # env_map= "/root/autodl-tmp/Relightable3DGaussian/FA_scene/HLDR/env_equi.hdr"
    # 初始化 Gaussians
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    
    # 加载 PLY
    if is_ply=="True": 
        ply_path="/home/chengwr/code/Relightable3DGaussian/FA_scene/so101_desktop.ply"
        gaussians.load_ply_safe(ply_path)
        print("loaded ply of table!")
    elif args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    else:
        raise ValueError("not loading scene!")
        创建场景对象
    base_dir="./FA_scene/table"
    train_cameras=camlist_from_infos(load_cam_img(cam_dir=os.path.join(base_dir,"fibo/cam"),img_dir=os.path.join(base_dir,"fibo/img"),mask_dir=os.path.join(base_dir,"fibo/masks")))
    test_cameras=camlist_from_infos(load_cam_img(cam_dir=os.path.join(base_dir,"ring/cam"),img_dir=os.path.join(base_dir,"ring/img"),mask_dir=os.path.join(base_dir,"ring/masks")))
    
    
    scene = SceneFromGaussians(args,gaussians,train_cameras,test_cameras)  # 需要定义一个专门接收已有 Gaussians 的 Scene

    # 设置训练优化器
    gaussians.training_setup(opt)

        # PBR 光照__%%%%%%什么是envmap ？？？？？
    pbr_kwargs = dict()
    if is_pbr:
        gaussians.update_visibility(pipe.sample_num)
        pbr_kwargs['sample_num'] = pipe.sample_num
        env_map = DirectLightMap(dataset.env_resolution, opt.light_init)
        # gt_env_path="./FA_scene/HLDR/confroom/env_64.exr"
        # env=load_hdr_to_tensor(gt_env_path)#1,3,h,w
        # env=env.unsqueeze(dim=0).permute(0,2,3,1)#1,h,w,3
        # env_map=DirectLightMap(H=32,env=env)
        pbr_kwargs = {"env_light": env_map, "sample_num": pipe.sample_num}
        env_map.training_setup(opt)


        # 渲染函数
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    empty_tensor = torch.tensor([])
    default_device = empty_tensor.device
    print(f"默认张量类型: {default_device}")
    #     # GUI
    windows = None
    if args.gui:
        cam = scene.getTrainCameras()[0]
        c2w = cam.c2w.detach().cpu().numpy()
        center = gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()
        render_kwargs = {"pc": gaussians, "pipe": pipe, "bg_color": background,
                            "opt": opt, "is_training": False, "dict_params": pbr_kwargs}
        windows = GUI(cam.image_height, cam.image_width, cam.FoVy,
                        c2w=c2w, center=center,
                        render_fn=render_fn, render_kwargs=render_kwargs,
                        mode='pbr')

    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        if windows is not None:
            windows.render()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        
        # Every 1000 update visibility
        if is_pbr and iteration % 500 == 0:
            gaussians.update_visibility(pipe.sample_num)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        # loss = torch.tensor(0.0, device="cuda")
        # print("len(viewpoint stack:)",len(viewpoint_stack))
        k=randint(0, len(viewpoint_stack) - 1)
        # print(viewpoint_stack[k].R,viewpoint_stack[k].T)
        viewpoint_cam = viewpoint_stack.pop(k)
        
        # print("k:%%%%%%%%%%",k)
        # print(viewpoint_cam.R,viewpoint_cam.T)
        
        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, 
                               opt=opt, is_training=True, dict_params=pbr_kwargs, iteration=iteration)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        #%%%%%%%%%%%%%%%%%%%%%%%%debug
        # loss.backward(retain_graph=True)
        # print("loss grad ok")

        # if torch.isnan(loss) or torch.isinf(loss):
        #     print("[DEBUG]LOSS ERROR:isnan or isinf", loss)
        #     exit()
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]
        loss.backward()

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(viewpoint_cam, gaussians, background, render_fn,
                                  pipe, opt, first_iter, iteration, pbr_kwargs)
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            if is_pbr:
                pbar_dict["light_mean"] = env_map.get_env.mean().item()
                pbar_dict["env"] = env_map.H
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            training_report(tb_writer, iteration, tb_dict,
                            scene, render_fn, pipe=pipe,scaling_modifier=2.0,
                            bg_color=background, dict_params=pbr_kwargs)

            # # densification(freeze,no densification step)
            
            # if iteration < opt.densify_until_iter:
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 
            #                                         render_pkg['weights'])
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #                                                             radii[visibility_filter])
                
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 15 if iteration > opt.opacity_reset_interval else None
            #         densify_grad_normal_threshold = opt.densify_grad_normal_threshold if iteration > opt.normal_densify_from_iter else 99999
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
            #                                     densify_grad_normal_threshold,weights_threshold=1e-4)

            #     if iteration % opt.opacity_reset_interval == 0 or (
            #             dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()
                    
            # Optimizer step
            gaussians.step()
            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass
            
            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

    if dataset.eval:
        eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)


def load_hdr_to_tensor(hdr_path):
    # 打开 EXR 文件
    exr = OpenEXR.InputFile(hdr_path)
    
    # 获取数据窗口
    header = exr.header()
    dw = header['dataWindow']
    
    # 获取图像宽高
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    # 定义像素类型
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # 读取 RGB 三个通道
    channels = [np.frombuffer(exr.channel(c, FLOAT), dtype=np.float32).reshape(h, w)
                for c in ("R", "G", "B")]

    # 将 RGB 三个通道堆叠到一起，并调整维度为 (1, 3, H, W)
    hdr = np.stack(channels, axis=0)

    # 将 NumPy 数组转换为 Torch 张量，并返回
    return torch.from_numpy(hdr)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='neilf')
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    #%%%%%%%%
    # parser.add_argument("--is_mask", type=str, choices=['True', 'False'], default='False')
    #%%%%%%%%
    parser.add_argument("--is_ply", type=str, choices=['True', 'False'], default='False')
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    # print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    is_pbr = args.type in ['neilf']
    # ply_path="./FA_scene/confroom.ply"
    # ply_path="/home/chengwr/code/Relightable3DGaussian/output/confroom/test_rli/iteration_5000.ply"

    # #&&&验证高斯点数量
    # plydata = PlyData.read(ply_path)
    # num_gaussians = len(plydata['vertex'].data)
    # print("Number of Gaussians in PLY:", num_gaussians)
    #&&
  
    training_new(is_ply=args.is_ply,opt=op.extract(args), dataset=lp.extract(args), pipe=pp.extract(args),is_pbr=is_pbr)

    # All done
    print("\nTraining complete.")
