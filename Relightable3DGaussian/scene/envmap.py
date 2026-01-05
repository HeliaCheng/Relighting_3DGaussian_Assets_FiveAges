
import torch
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
import imageio
import pyexr
from utils.graphics_utils import srgb_to_rgb
# imageio.plugins.freeimage.download()
# print(imageio.plugins.freeimage.FREEIMAGE_INSTALL_DIR)
class EnvLight(torch.nn.Module):
    def __init__(self, path=None, scale=1.0):
        super().__init__()
        self.device = "cuda"  # only supports cuda
        self.scale = scale  # scale of the hdr values
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")

        self.envmap = self.load(path, scale=self.scale, device=self.device)
        self.transform = None

    @staticmethod
    def load(envmap_path, scale, device):
        if not envmap_path.endswith(".exr"):
            image = srgb_to_rgb(imageio.imread(envmap_path)[:, :, :3] / 255)
        else:
            # load latlong env map from file
            image = pyexr.open(envmap_path).get()[:, :, :3]

        image = image * scale

        env_map_torch = torch.tensor(image, dtype=torch.float32, device=device, requires_grad=False)

        return env_map_torch

    def direct_light(self, dirs, transform=None):
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)

        if transform is not None:
            dirs = dirs @ transform.T
        elif self.transform is not None:
            dirs = dirs @ self.transform.T

        envir_map =  self.envmap.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        # print(envir_map.shape,"%%%%%%%%%%%%%envir_map.shape")#yes  【1，3，2048，4096】
        # phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6  #
        # theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)  #
         # phi：天顶角，范围[0, π]，-Y轴（顶）→ phi=0，Y轴（下）→ phi=π
        phi = torch.arccos(-dirs[:, 1]).reshape(-1) - 1e-6  # 关键修改：-dirs[:,1]（顶为-Y）
        # theta：方位角，范围[-π, π]，X轴正方向（右）→ theta=0，逆时针旋转增大
        theta = torch.atan2(dirs[:, 2], dirs[:, 0]).reshape(-1)  # 关键修改：atan2(z, x)（RDF水平基准为X-Z平面）
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        # print(grid.shape,"%%%%%%%%%%%grid.shape")#[1, 1, 1536000, 2]
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        # print(light_rgbs.shape,"%%%%%%%%%light_rgbs.shape")#[1536000, 3]
        # print(light_rgbs.reshape(*shape).shape,"%%%%%%%%%%%%%%%%returned ")
        return light_rgbs.reshape(*shape)
