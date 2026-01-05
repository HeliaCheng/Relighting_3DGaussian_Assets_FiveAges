import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from PIL import Image
import os

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_h, orig_w = cam_info.image.shape[:2]

    if args.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * args.resolution
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

        scale = global_down * resolution_scale
    resolution = (int(orig_h / scale), int(orig_w / scale))

    image = torch.from_numpy(cam_info.image).float().permute(2, 0, 1)
    if scale == 1:
        resized_image_rgb = image
    else:
        resized_image_rgb = torchvision.transforms.Resize(resolution, antialias=True)(image)
    gt_image = resized_image_rgb

    resized_depth = None
    if cam_info.depth is not None:
        depth = torch.from_numpy(cam_info.depth).float().unsqueeze(0)
        resized_depth = torchvision.transforms.Resize(
            resolution, interpolation=InterpolationMode.NEAREST)(depth)

    resized_normal = None
    if cam_info.normal is not None:
        normal = torch.from_numpy(cam_info.normal).float().permute(2, 0, 1)
        resized_normal = torchvision.transforms.Resize(
            resolution, interpolation=InterpolationMode.NEAREST)(normal)

    resized_image_mask = None
    if cam_info.image_mask is not None:
        image_mask = torch.from_numpy(cam_info.image_mask).float().unsqueeze(0)
        resized_image_mask = torchvision.transforms.Resize(
            resolution, interpolation=InterpolationMode.NEAREST)(image_mask)

    # change the fx and fy
    scale_cx = cam_info.cx
    scale_cy = cam_info.cy
    scale_fx = cam_info.fx
    scale_fy = cam_info.fy
    if cam_info.cx is not None and cam_info.cy is not None:
        scale_cx /= scale
        scale_cy /= scale
        scale_fx /= scale
        scale_fy /= scale

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, fx=scale_fx, fy=scale_fy, cx=scale_cx, cy=scale_cy,
                  image=gt_image, depth=resized_depth, normal=resized_normal, image_mask=resized_image_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos, desc="resolution scale: {}".format(resolution_scale), leave=False)):
        # for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    if camera.cx is None:
        camera_entry = {
            'id': id,
            'img_name': camera.image_name,
            'width': camera.width,
            'height': camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'FoVx': camera.FovX,
            'FoVy': camera.FovY,
        }
    else:
        camera_entry = {
            'id': id,
            'img_name': camera.image_name,
            'width': camera.width,
            'height': camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fx': camera.fx,
            'fy': camera.fy,
            'cx': camera.cx,
            'cy': camera.cy,
        }
    return camera_entry


def JSON_to_camera(json_cam):
    rot = np.array(json_cam['rotation'])
    pos = np.array(json_cam['position'])
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos
    W2C[3, 3] = 1
    Rt = np.linalg.inv(W2C)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]
    H, W = json_cam['height'], json_cam['width']
    if 'cx' not in json_cam:
        if 'fx' in json_cam:
            FovX = focal2fov(json_cam["fx"], W)
            FovY = focal2fov(json_cam["fy"], H)
        else:
            FovX = json_cam["FoVx"]
            FovY = json_cam["FoVy"]
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, fx=None, fy=None, cx=None, cy=None,
                        image=None, image_name=json_cam['img_name'], uid=json_cam['id'],
                        data_device='cuda', height=H, width=W)
    else:
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=None, FoVy=None, fx=json_cam["fx"], fy=json_cam["fy"],
                        cx=json_cam["cx"], cy=json_cam["cy"], image=None, image_name=json_cam['img_name'],
                        uid=json_cam['id'], data_device='cuda', height=H, width=W)
    return camera

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
            depth=None,
            normal=None,
            fx=512,
            fy=512,
            cx=512,
            cy=512
        )
        camlist.append(cam)
    return camlist

