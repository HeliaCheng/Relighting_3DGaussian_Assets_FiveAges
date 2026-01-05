import numpy as np
from plyfile import PlyData, PlyElement
import os
import torch
from torch import nn
from utils.graphics_utils import BasicPointCloud
from scene import GaussianModel
import cv2
from gaussian_renderer import render
from scene.cameras import Camera
import torch.nn.functional as F  
import math
from PIL import Image
import cv2
from math import pi, atan2, asin
import imageio.v3 as iio
import torch
import numpy as np
import os
from PIL import Image
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R
from arguments import OptimizationParams, ParamGroup, PipelineParams
import argparse
from utils.graphics_utils import getWorld2View2, getWorld2View, getProjectionMatrix
from tqdm import tqdm

class GaussianRenderer:
    def __init__(self, opt, scene_dir="./FA_scene", sh_degree=0):
        """
        :param opt: OptimizationParams or other config object
        :param scene_dir: 文件夹包含 ply 模型
        :param sh_degree: 球谐阶数（保留）
        """
        self.opt = opt
        self.scene_dir = scene_dir
        self.sh_degree = sh_degree
        self.scene_list = []
        self.loaded = False
        self.all_points = None
        self.space = None

    def load_models(self, target="confroom_hl.ply"):
            """加载指定目录下包含 target 的 ply 模型，并旋转顶点使Up从+Z转向-Y"""
            for filename in os.listdir(self.scene_dir):
                file_path = os.path.join(self.scene_dir, filename)
                if os.path.isfile(file_path) and target in filename:
                    gaussian_model = GaussianModel(sh_degree=self.sh_degree)
                    
                    gaussian_model.load_ply_safe(file_path)
                   
                    self.scene_list.append(gaussian_model)

            if not self.scene_list:
                raise ValueError("No Gaussian models loaded. scene_list is empty.")
            self.loaded = True
            print(f"Successfully loaded {len(self.scene_list)} valid Gaussian models")
            return self

    def render_cam_and_img(self,
                           fibo_views: int,
                           output_base_dir: str,
                           pipe,
                           radius: float = None,
                           ring_count: int = 100):
        """
        生成两组视角并渲染，分别存储到不同文件夹：
          1) Fibonacci 球面采样（数量 fibo_views）
          2) 环绕中点高度的一圈采样（数量 ring_count，默认100）
        
        :param fibo_views: 球面均匀采样视角数量（可为0跳过）
        :param output_base_dir: 基础输出目录，将在其中创建子文件夹
        :param pipe: pipeline params 给 render 使用
        :param radius: 若 None，会自动设为场景 bbox 最大边长的 0.8 倍；否则用该 r
        :param ring_count: 环绕采样数量（默认100）
        """
        # 创建输出目录结构
        fibo_img_dir = os.path.join(output_base_dir, "fibo", "img_bl")
        fibo_cam_dir = os.path.join(output_base_dir, "fibo", "cam")
        ring_img_dir = os.path.join(output_base_dir, "ring", "img_bl")
        ring_cam_dir = os.path.join(output_base_dir, "ring", "cam")
        
        os.makedirs(fibo_img_dir, exist_ok=True)
        os.makedirs(fibo_cam_dir, exist_ok=True)
        os.makedirs(ring_img_dir, exist_ok=True)
        os.makedirs(ring_cam_dir, exist_ok=True)

        # compute bbox & center
        self.space = self._compute_scene_bounding_box()
        center = np.array([
            (self.space["L"] + self.space["R"]) / 2.0,
            (self.space["D"] + self.space["U"]) / 2.0,
            (self.space["B"] + self.space["F"]) / 2.0
        ], dtype=np.float32)
        print(f"Scene center: {center}")

        # auto radius estimate
        bbox_diag = np.sqrt((self.space["R"] - self.space["L"])**2 +
                            (self.space["U"] - self.space["D"])**2 +
                            (self.space["F"] - self.space["B"])**2)
        if radius is None:
            r = max(0.2, 0.55 * bbox_diag)
        else:
            r = float(radius)
        print(f"Camera radius: {r}")

        # 渲染模型
        model = self.scene_list[0]
        bg_color_tensor = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

        # 1) Fibonacci sphere sampling
        if fibo_views > 0:
            print(f"\n📸 Rendering Fibonacci sphere views ({fibo_views})...")
            fibo_pts = self._fibonacci_sphere_points(center, r, fibo_views)
            for i, pos in enumerate(tqdm(fibo_pts, desc="Fibonacci views")):
                cam = self._make_camera_from_pos(pos, center, uid=i)
                
                # 保存相机外参
                R_c2w = cam.R  # 已经是 3x3
                T_w2c = cam.T  # 3x1
                extrinsics=np.eye(4)
                extrinsics[:3,:3]=R_c2w
                extrinsics[:3,3]=T_w2c
                cam_path = os.path.join(fibo_cam_dir, f"cam_{i:03d}.txt")
                np.savetxt(cam_path, extrinsics, fmt="%.6f", delimiter=" ")
                
                # 渲染图像
                render_output = self._render_single_view(cam, model, bg_color_tensor, pipe)
                img_path = os.path.join(fibo_img_dir, f"img_{i:03d}.png")
                self._save_image(render_output, img_path)

        # 2) Ring sampling
        if ring_count > 0:
            print(f"\n📸 Rendering ring views ({ring_count})...")
            # 改动：传递radius=r到_ring_points，用于45度角计算
            ring_pts = self._ring_points(center, r, ring_count)
            for i, pos in enumerate(tqdm(ring_pts, desc="Ring views")):
                cam = self._make_camera_from_pos(pos, center, uid=i + fibo_views)
                
                # 保存相机外参
                R_c2w = cam.R  # 已经是 3x3
                T_w2c = cam.T  # 3x1
                extrinsics=np.eye(4)
                extrinsics[:3,:3]=R_c2w
                extrinsics[:3,3]=T_w2c
                cam_path = os.path.join(ring_cam_dir, f"cam_{i:03d}.txt")
                np.savetxt(cam_path, extrinsics, fmt="%.6f", delimiter=" ")
                
                # 渲染图像
                render_output = self._render_single_view(cam, model, bg_color_tensor, pipe)
                img_path = os.path.join(ring_img_dir, f"img_{i:03d}.png")
                self._save_image(render_output, img_path)

        print(f"\n✅ All images saved. Total views: Fibonacci={fibo_views}, Ring={ring_count}")
        print(f"   Fibonacci views saved to: {fibo_img_dir}")
        print(f"   Ring views saved to: {ring_img_dir}")

    def _render_single_view(self, cam, model, bg_color, pipe):
        """渲染单张视图"""
        render_output = render(
            viewpoint_camera=cam,
            pc=model,
            bg_color=bg_color,
            pipe=pipe,
            scaling_modifier=1,
            iteration=1,
        )
        
        # 根据实际的render输出键名调整
        if "render" in render_output:
            rgb_tensor = render_output["render"]
        elif "image" in render_output:
            rgb_tensor = render_output["image"]
        else:
            print("Render output keys:", render_output.keys())
            raise KeyError("Can't find image tensor in render output")
        
        return rgb_tensor

    def _save_image(self, rgb_tensor, img_path):
        """保存图像"""
        rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)
        img_np = (rgb_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(img_path)

    def _compute_scene_bounding_box(self):
        all_points = []
        for gaussian_data in self.scene_list:
            pts = None
            # 尝试常见属性名
            if hasattr(gaussian_data, "_xyz"):
                pts = gaussian_data._xyz
            elif hasattr(gaussian_data, "xyz"):
                pts = gaussian_data.xyz
            elif isinstance(gaussian_data, dict) and "xyz" in gaussian_data:
                pts = gaussian_data["xyz"]

            if pts is None:
                continue

            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()

            if pts.size == 0:
                continue
            all_points.append(pts)

        if not all_points:
            raise ValueError("self.scene_list 中无有效点坐标数据")

        all_points = np.vstack(all_points)
        min_x, min_y, min_z = all_points.min(axis=0)
        max_x, max_y, max_z = all_points.max(axis=0)
        self.all_points = all_points
        return {
            "U": float(max_y),
            "D": float(min_y),
            "L": float(min_x),
            "R": float(max_x),
            "F": float(max_z),
            "B": float(min_z)
        }

    def _fibonacci_sphere_points(self, center: np.ndarray, r: float, n: int):
        """
        生成 n 个在球面半径 r 上均匀分布的点（Fibonacci sphere）
        返回 numpy array 列表 shape (n,3)
        """
        points = []
        if n <= 0:
            return points

        # 黄金角
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            z = 1.0 - (i / float(n - 1)) * 2.0  # y 从 1 到 -1
            radius = np.sqrt(max(0.0, 1.0 - z * z))
            theta = phi * i
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            # 单位球坐标 (x,y,z) -> 缩放 r，平移中心
            p = center + r * np.array([x, y, z], dtype=np.float32)
            points.append(p)
        return points

    def _ring_points(self, center: np.ndarray, r: float, n: int):
        """
        生成与地面夹角45度的环绕采样点（俯视物体）
        :param center: (3,) 场景中心
        :param r: 相机到场景中心的欧式距离（保持不变）
        :param n: 点数量
        """
        points = []
        if n <= 0:
            return points
        
        # 45度角几何计算：
        # 设相机到场景中心的连线与XZ平面（地面）夹角为45°
        # 则：y轴高度差 = 水平距离（XZ平面）
        # 由勾股定理：(水平距离)^2 + (高度差)^2 = r^2 → 2*(高度差)^2 = r^2 → 高度差 = r/√2
        height_offset = r / np.sqrt(2)  # 45度角对应的高度差
        cz = center[2] + height_offset  # 相机y坐标（在场景中心上方height_offset处）

        for i in range(n):
            theta = 2.0 * np.pi * (i / float(n))
            # 计算XZ平面上的偏移（水平距离 = r/√2）
            x_offset = (r / np.sqrt(2)) * np.cos(theta)
            y_offset = (r / np.sqrt(2)) * np.sin(theta)
            # 相机位置：场景中心X/Z + 水平偏移，Y + 高度偏移
            x = center[0] + x_offset
            y = center[1] + y_offset
            z=cz
            points.append(np.array([x, y, z], dtype=np.float32))
        return points

    def _make_camera_from_pos(self, pos, center, uid=0):
        """
        基于世界坐标 pos 和物体中心 center 构造 3DGS/OpenGL 风格相机
        坐标系约定（OpenGL right-hand, -Z 看向前方）：
            camera x = right
            camera y = up
            camera z = -forward（camera 看向 -Z）
        """

        pos = np.asarray(pos, dtype=np.float32)
        print(pos,"pos")
        center = np.asarray(center, dtype=np.float32)
        #RDF
        # 1. 相机朝向
        forward = center - pos  # 指向目标
        forward /= np.linalg.norm(forward)

        # world up
        world_up = np.array([0, 0, -1], dtype=np.float32)
        if abs(np.dot(world_up, forward)) > 0.99:
            world_up = np.array([1, 0, 0], dtype=np.float32)

    
        right = np.cross(world_up,forward)
        right /= np.linalg.norm(right)

      
        up = np.cross(forward,right)
        up /= np.linalg.norm(up)

        # 2. 构造旋转矩阵 c2w（列堆叠）
        
        R_c2w = np.stack([right, up, forward], axis=1).astype(np.float32)
        R_w2c = R_c2w.T

        # 3. 平移
        T_w2c = -R_w2c @ pos

        # 4. 构造 4x4 矩阵
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = T_w2c

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = pos

        # 5. Camera 对象
        fov_x = fov_y = 90
        img_w = img_h = 1024
        # fx = fy = img_w / 2.0
        # cx = cy = img_w / 2.0

        cam = Camera(
            colmap_id=uid,
            R=R_c2w,
            T=T_w2c,
            FoVx=fov_x,
            FoVy=fov_y,
            gt_alpha_mask=None,
            image_mask=torch.zeros(1, img_h, img_w),
            image=torch.zeros(3, img_h, img_w, device="cuda"),
            image_name=f"view_{uid}",
            uid=uid,
            data_device="cuda"
        )
        camera_center = torch.tensor(pos, dtype=torch.float32, device="cuda")
        trans=[0,0,0]
        #getWorld2View2:input R_c2w(column-first),T_w2c----output:w2c(row-first)
        cam.world_view_transform = torch.tensor(getWorld2View2(R_c2w, T_w2c, trans, scale=1)).transpose(0, 1).cuda()
        cam.projection_matrix = getProjectionMatrix(
                            cam.znear, cam.zfar,cam.FoVx,cam.FoVy).transpose(0, 1).cuda()
        cam.full_proj_transform = (
                        cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        cam.c2w = cam.world_view_transform.transpose(0, 1).inverse()
        # cam.intrinsics = cam.get_intrinsics()
        # cam.extrinsics = cam.get_extrinsics()
        # cam.proj_matrix = cam.get_proj_matrix()
                    
        assert torch.allclose(cam.camera_center, camera_center, atol=1e-5), \
                        f"[Err] camera_center mismatch {cam.camera_center} vs {camera_center}"
        

        return cam



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    opt = OptimizationParams(parser)
    pipe = PipelineParams(parser)

    renderer = GaussianRenderer(opt=opt, scene_dir="../Relightable3DGaussian/FA_scene/gs_results", sh_degree=0)
    renderer.load_models(target="chair.ply")
    
    sample_fibo = 200  # 球面视角数
    output_base_dir = "../Relightable3DGaussian/FA_scene/gs_results/chair_bl"  # 基础输出目录
    
    renderer.render_cam_and_img(
        fibo_views=sample_fibo,
        output_base_dir=output_base_dir,
        pipe=pipe,
        radius=None,    # 自动估计半径
        ring_count=100  # 环绕采样100张
    )