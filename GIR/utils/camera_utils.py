#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from PIL import Image
import os
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)


    gt_image = resized_image_rgb

    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, exposure=cam_info.exposure)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


###########nenenwnennewnnenenwnenw:added%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sorted_files_in_order(directory):
    # 获取目录中的所有文件
    files = os.listdir(directory)
    # 对文件进行排序，假设文件格式为 'cam_XXX.txt'，按数字升序排序
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return files

def load_cam_img(cam_dir,img_dir,mask_dir):
    R_set=[]
    T_set=[]
    gt_images=[]
    gt_mask=[]
    sorted_cam_name=sorted_files_in_order(cam_dir)
    sorted_img_name=sorted_files_in_order(img_dir)
    sorted_mask_name=sorted_files_in_order(mask_dir)

    for ms in sorted_mask_name:
        mask_path = os.path.join(mask_dir,ms)
        # ms= Image.open(mask_path)  # 返回 PIL.Image 对象
        ms = Image.open(mask_path).convert("L")   # 转成灰度，确保单通道
        img_array = np.array(ms).astype(np.float32) / 255.0   # 0~255 → 0~1
        img_tensor=torch.from_numpy(img_array)
        # print(img_tensor.shape)
        gt_mask.append(img_tensor)
    # print(sorted_cam_name[:10])
    for camtxt in sorted_cam_name:
        txt_path = os.path.join(cam_dir, camtxt)
        mat_4x4 = np.loadtxt(txt_path, delimiter=None, dtype=np.float32)
        assert mat_4x4.shape == (4, 4), f"{camtxt} 不是 4×4 矩阵，实际形状是 {mat_4x4.shape}！"
        
        # 提取 R（前 3×3 矩阵）和 T（第 4 列前 3 个元素）
        R = mat_4x4[:3, :3]  # 3×3 旋转矩阵
        T = mat_4x4[:3, 3]   # 1×3 平移向量（shape: (3,)）

        R_set.append(R)
        T_set.append(T)

    for img in sorted_img_name:
        img_path = os.path.join(img_dir,img)
        img= Image.open(img_path)  # 返回 PIL.Image 对象
        img_array = np.array(img)  # 转换为 NumPy 数组（形状：(h, w, 3)，RGB 格式）
        img_array_chw = img_array.transpose(2, 0, 1)  # (3,h,w)
        img_tensor=torch.from_numpy(img_array_chw)
        gt_images.append(img_tensor)
   

    return R_set,T_set,gt_images,gt_mask


def camlist_from_infos(infos):
    
    if len(infos)==3:
        R_set,T_set,gt_images=infos
    if len(infos)==4:
        R_set,T_set,gt_images,gt_mask=infos
    assert len(gt_images)==len(R_set)
    camlist=[]
    for idx, gi in enumerate(gt_images):
        R_w2c=R_set[idx]
        T_w2c=T_set[idx]
        gm=gt_mask[idx]
        cam=Camera(
            colmap_id=-1,
            uid=idx,
            R=R_w2c,
            T=T_w2c,
            FoVx=90,
            FoVy=90,
            image=gi,
            image_mask=gm,
            image_name=f"img_{idx:3d}",
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            data_device="cuda",
            gt_alpha_mask=None,
        )
        camlist.append(cam)
    return camlist