import json
import os
import cv2
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
from scene import GaussianModel,SceneFromGaussians
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
from scene.envmap import EnvLight
from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from tqdm import tqdm
from utils.graphics_utils import rgb_to_srgb
from utils.camera_utils import camlist_from_infos,load_cam_img
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

def scene_composition(scene_dict: dict, dataset: ModelParams):
    gaussians_list = []
    for scene in scene_dict:
        gaussians = GaussianModel(dataset.sh_degree, render_type="neilf")
        gaussians.load_ply(scene_dict[scene]["path"])

        torch_transform = torch.tensor(scene_dict[scene]["transform"], device="cuda").reshape(4, 4)
        gaussians.set_transform(transform=torch_transform)

        gaussians_list.append(gaussians)

    gaussians_composite = GaussianModel.create_from_gaussians(gaussians_list, dataset)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    gaussians_composite._visibility_rest = (
        torch.nn.Parameter(torch.cat(
            [gaussians_composite._visibility_rest.data,
             torch.zeros(n, 5 ** 2 - 4 ** 2, 1, device="cuda", dtype=torch.float32)],
            dim=1).requires_grad_(True)))

    gaussians_composite._incidents_dc.data[:] = 0
    gaussians_composite._incidents_rest.data[:] = 0

    return gaussians_composite

def render_points(camera, gaussians):
    intrinsic = camera.get_intrinsics()
    w2c = camera.world_view_transform.transpose(0, 1)

    xyz = gaussians.get_xyz
    color = gaussians.get_base_color
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    xyz_cam = (xyz_homo @ w2c.T)[:, :3]
    z = xyz_cam[:, 2]
    uv_homo = xyz_cam @ intrinsic.T
    uv = uv_homo[:, :2] / uv_homo[:, 2:]
    uv = uv.long()

    valid_point = torch.logical_and(torch.logical_and(uv[:, 0] >= 0, uv[:, 0] < W),
                                    torch.logical_and(uv[:, 1] >= 0, uv[:, 1] < H))
    uv = uv[valid_point]
    z = z[valid_point]
    color = color[valid_point]

    depth_buffer = torch.full_like(render_pkg['render'][0], 10000)
    rgb_buffer = torch.full_like(render_pkg['render'], bg)
    while True:
        mask = depth_buffer[uv[:, 1], uv[:, 0]] > z
        if mask.sum() == 0:
            break
        uv_mask = uv[mask]
        depth_buffer[uv_mask[:, 1], uv_mask[:, 0]] = z[mask]
        rgb_buffer[:, uv_mask[:, 1], uv_mask[:, 0]] = color[mask].transpose(-1, -2)

    return rgb_buffer


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    # parser.add_argument('-co', '--config', default=None, required=True, help="the config root")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    # parser.add_argument('-e', '--envmap_path', default=None, help="Env map path")
    parser.add_argument('-bg', "--background_color", type=float, default=None,
                        help="If set, use it as background color")
    parser.add_argument('--bake', action='store_true', default=False, help="Bake the visibility and refine.")
    parser.add_argument('--video', action='store_true', default=False, help="If True, output video as well.")
    # parser.add_argument('--output', default="./capture_trace", help="Output dir.")
    parser.add_argument('--capture_list', default="pbr_env", help="what should be rendered for output.")
    parser.add_argument('--ply_load', type=str, default= None)
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    # load gaussians
    base_dir="./FA_scene/table"
    train_cameras=camlist_from_infos(load_cam_img(cam_dir=os.path.join(base_dir,"fibo/cam"),img_dir=os.path.join(base_dir,"fibo/img"),mask_dir=os.path.join(base_dir,"fibo/masks")))
    test_cameras=camlist_from_infos(load_cam_img(cam_dir=os.path.join(base_dir,"ring/cam"),img_dir=os.path.join(base_dir,"ring/img"),mask_dir=os.path.join(base_dir,"ring/masks")))

    gaussians = GaussianModel(dataset.sh_degree, render_type="neilf")
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)
    elif args.ply_load:
        print("create Gaussians from ply")
        gaussians.load_ply(args.ply_load)
    else:
        raise NotImplementedError
    
    scene = SceneFromGaussians(args, gaussians, train_cameras=train_cameras, test_cameras=test_cameras)

    #render
    task_dict = {
        "env6": {
            "capture_list": ["pbr",  "base_color","pbr_env"],
            "envmap_path": "env_map/envmap6.exr",
        },
        "env3": {
            "capture_list": ["pbr",  "base_color","pbr_env"],
            "envmap_path": "env_map/envmap3.exr",
        },
        "env_fa":{
            "capture_list": ["pbr", "base_color","pbr_env"],
            "envmap_path": "FA_scene/HLDR/confroom/env_equi.exr",
        },
        "env12": {
            "capture_list": ["pbr", "base_color","pbr_env"],
            "envmap_path": "env_map/envmap12.exr",
        },
        "night":{
            "capture_list":["pbr","base_color","pbr_env"],
            "envmap_path":"env_map/night.exr",
        },
         "pano_1":{
            "capture_list":["pbr","base_color","pbr_env"],
            "envmap_path":"env_map/inpainted_pano_1.exr",
        },
         "pano_7":{
            "capture_list":["pbr","base_color","pbr_env"],
            "envmap_path":"env_map/inpainted_pano_7.exr",
        },
         "doubao1":{
            "capture_list":["pbr","base_color","pbr_env"],
            "envmap_path":"env_map/doubao1.exr",
        },
    }
    
    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict['neilf']
    gaussians.update_visibility(args.sample_num)
    
    results_dir = os.path.join(args.model_path,"test_rli","twisted_35000_doubao")
    task_names = ["doubao1"]
    for task_name in task_names:
        task_dir = os.path.join(results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        light = EnvLight(path=task_dict[task_name]["envmap_path"], scale=1)

        render_kwargs = {
            "pc": gaussians,
            "pipe": pipe,
            "bg_color": background,
            "is_training": False,
            "scaling_modifier": 1.0,
            "dict_params": {
                "env_light": light,
                "sample_num": args.sample_num,
            },
        }
        
        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
            
        psnr_albedo = 0.0
        ssim_albedo = 0.0
        lpips_albedo = 0.0
            
        mse_roughness = 0.0
            
        capture_list = task_dict[task_name]["capture_list"]
        for capture_type in capture_list:
            capture_type_dir = os.path.join(task_dir, capture_type)
            os.makedirs(capture_type_dir, exist_ok=True)

        os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
    
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]
                
        for idx, frame in enumerate(tqdm(scene.getTestCameras(), leave=True)):
            cam=frame
            len_cam=len(scene.getTestCameras())
            gt_image=cam.original_image # (3,h,w)
            with torch.no_grad():
                render_pkg = render_fn(viewpoint_camera=cam, **render_kwargs)

            for capture_type in capture_list:
                if capture_type == "points":
                    render_pkg[capture_type] = render_points(cam, gaussians)
                elif capture_type == "normal":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
                elif capture_type in ["base_color", "roughness", "visibility"]:
                    render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
                elif capture_type in ["pbr", "pbr_env", "render"]:
                    render_pkg[capture_type] = render_pkg[capture_type]
                save_image(render_pkg[capture_type], f"{task_dir}/{capture_type}/frame_{idx}.png")
                    
            
            save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
                        
            with torch.no_grad():
                psnr_pbr += psnr(render_pkg['pbr'], gt_image).mean().double()
                ssim_pbr += ssim(render_pkg['pbr'], gt_image).mean().double()
                lpips_pbr += lpips(render_pkg['pbr'], gt_image, net_type='vgg').mean().double()

    
    psnr_pbr /= len_cam
    ssim_pbr /= len_cam
    lpips_pbr /= len_cam
        
        
    with open(os.path.join(task_dir, f"metric.txt"), "w") as f:
        f.write(f"psnr_pbr: {psnr_pbr}\n")
        f.write(f"ssim_pbr: {ssim_pbr}\n")
        f.write(f"lpips_pbr: {lpips_pbr}\n")

            
    print("\nEvaluating {}: PSNR_PBR {} SSIM_PBR {} LPIPS_PBR {}".format(task_name, psnr_pbr, ssim_pbr, lpips_pbr))




    # output as video
    if args.video:
        # progress_bar = tqdm(capture_list, desc="Outputting video")
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        results_dir = os.path.join(args.model_path, "test_rli")
        task_names = ["night","env12","env6"]
        for task_name in task_names:
            task_dir = os.path.join(results_dir, task_name)
            capture_type="pbr_env"
            video_path = f"{task_dir}/{capture_type}.mp4"
            image_names = [os.path.join(task_dir, capture_type, f"frame_{j}.png") for j in
                           range(100) ]
            media_writer = cv2.VideoWriter(video_path, fourcc, 60, (1024,1024))

            for image_name in image_names:
                img = cv2.imread(image_name)
                media_writer.write(img)
            media_writer.release()



    # # load configs
    # scene_config_file = f"{args.config}/transform.json"
    # traject_config_file = f"{args.config}/trajectory.json"
    # light_config_file = f"{args.config}/light_transform.json"

    # scene_dict = load_json_config(scene_config_file)
    # traject_dict = load_json_config(traject_config_file)
    # light_dict = load_json_config(light_config_file)

      # load gaussians
    # light = EnvLight(path=args.envmap_path, scale=1)
    # # gaussians_composite = scene_composition(scene_dict, dataset)
    # gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    # ply_path="/output/Syn4Relight/point_cloud/iteration_30000.ply"
    # gaussians.load_ply(ply_path)
    # # scene = SceneFromGaussians(gaussians=gaussians,test_cameras=test_cameras)  # 需要定义一个专门接收已有 Gaussians 的 Scene
    # # update visibility
    # gaussians.update_visibility(args.sample_num)

    # # rendering
    # capture_dir = args.output
    # os.makedirs(capture_dir, exist_ok=True)
    # capture_list = [str.strip() for str in args.capture_list.split(",")]
    # for capture_type in capture_list:
    #     capture_type_dir = os.path.join(capture_dir, capture_type)
    #     os.makedirs(capture_type_dir, exist_ok=True)

    # bg = args.background_color
    # if bg is None:
    #     bg = 1 if dataset.white_background else 0
    # background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    # render_fn = render_fn_dict['neilf']

    # render_kwargs = {
    #     "pc": gaussians,
    #     "pipe": pipe,
    #     "bg_color": background,
    #     "is_training": False,
    #     "scaling_modifier":2.0,
    #     "dict_params": {
    #         "env_light": light,
    #         "sample_num": args.sample_num,
    #     },
    #     "bake": args.bake
    # }

    # H = 1024
    # W = 1024

    # fovx = 90
    # fovy = focal2fov(fov2focal(fovx, W), H)

    # # progress_bar = tqdm(, desc="Rendering")

    # psnr_test = 0.0
    # ssim_test = 0.0
    # lpips_test = 0.0
        # for idx, cam_info in progress_bar:
    #         w2c = np.array(cam_info, dtype=np.float32).reshape(4, 4)

    #         R = w2c[:3, :3].T
    #         T = w2c[:3, 3]
    #         custom_cam = Camera(colmap_id=0, R=R, T=T,
    #                             FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
    #                             image=torch.zeros(3, H, W), image_name=None, uid=0)
    #         if light_dict is not None:
    #             light.transform = torch.tensor(light_dict["transform"][idx], dtype=torch.float32, device="cuda").reshape(3, 3)

    #         with torch.no_grad():
    #             render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)

    #         for capture_type in capture_list:
    #             if capture_type == "points":
    #                 render_pkg[capture_type] = render_points(custom_cam, gaussians_composite)
    #             elif capture_type == "normal":
    #                 render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
    #                 render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
    #             elif capture_type in ["base_color", "roughness", "visibility"]:
    #                 render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
    #             elif capture_type in ["pbr", "pbr_env", "render"]:
    #                 render_pkg[capture_type] = render_pkg[capture_type]
    #             save_image(render_pkg[capture_type], f"{capture_dir}/{capture_type}/frame_{idx}.png")


    