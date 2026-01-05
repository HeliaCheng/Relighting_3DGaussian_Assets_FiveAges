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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import laplacian, spatial_gradient
from lpipsPyTorch import lpips


def masked_ssim(img1, img2, mask):
    """
    img1, img2: (B,C,H,W) 或 (C,H,W), 范围[0,1]
    mask: (H,W) 或 (1,H,W), 需要归一化到[0,1]
    """
    # 关键修复：归一化mask
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    # 保证 batch 维
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)  # [1,1,H,W]

    ssim_map = ssim(img1, img2)  # shape [B,H,W] or [B,C,H,W]
    if ssim_map.dim() == 4:
        ssim_map = ssim_map.mean(1)  # 平均通道

    masked_val = (ssim_map * mask[:,0]).sum(dim=(1,2)) / (mask[:,0].sum(dim=(1,2)) + 1e-8)
    return masked_val.mean()

def masked_lpips(img1, img2, mask):
    """
    img1, img2: [B,3,H,W], range [-1,1]
    mask: [B,1,H,W], 需要归一化到[0,1]
    """
    _lpips_net = lpips(net_type='vgg').cuda()
    
    # 关键修复：归一化mask
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        mask = mask.unsqueeze(0)
    
    # 注意：LPIPS需要输入在[-1,1]范围，但您的图像是[0,1]
    # 如果需要，先转换图像范围
    if img1.min() >= 0 and img1.max() <= 1:
        img1 = img1 * 2 - 1  # [0,1] -> [-1,1]
        img2 = img2 * 2 - 1  # [0,1] -> [-1,1]
    
    B = img1.shape[0]
    vals = []
    for b in range(B):
        idxs = mask[b,0] > 0
        if idxs.sum() == 0:
            vals.append(torch.tensor(0.0, device=img1.device))
            continue
        img1_masked = img1[b:b+1] * mask[b:b+1]
        img2_masked = img2[b:b+1] * mask[b:b+1]
        val = _lpips_net(img1_masked, img2_masked)
        vals.append(val)
    return torch.stack(vals).mean()

def masked_l1(img, gt, mask):
    """
    img, gt: [3,H,W] 或 [B,3,H,W], 范围[0,1]
    mask: [1,H,W] 或 [B,1,H,W], 需要归一化到[0,1]
    """
    # 关键修复：归一化mask
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    if img.dim() == 3:
        img = img.unsqueeze(0)
        gt = gt.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    mask_exp = mask.expand_as(img)
    per_pixel_l1 = F.l1_loss(img, gt, reduction='none')
    return (per_pixel_l1 * mask_exp).sum() / (mask_exp.sum() + 1e-8)

def masked_psnr(img1, img2, mask):
    """
    img1, img2: [B, 3, H, W] 或 [3, H, W]
    mask: [B, 1, H, W] 或 [1, H, W], 1表示有效像素
    """
    # print("=== PSNR诊断信息 ===")
    # print(f"img1 (GT) 形状: {img1.shape}, 像素范围: [{img1.min():.6f}, {img1.max():.6f}]")
    # print(f"img2 (rendered) 形状: {img2.shape}, 像素范围: [{img2.min():.6f}, {img2.max():.6f}]")
    # print(f"mask 形状: {mask.shape}, 像素范围: [{mask.min():.6f}, {mask.max():.6f}]")
    # print(f"mask 中1的数量(有效像素): {(mask > 0.5).sum().item()}")
    # print(f"mask 中0的数量(被masked): {(mask <= 0.5).sum().item()}")
    # 确保输入是 [B, 3, H, W] 形状
    if mask.max() > 1.0:
        # print(f"检测到mask范围是[0,255]，正在归一化到[0,1]")
        mask = mask / 255.0
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)  # [1, 1, H, W]

    mask_exp = mask.expand(img1.shape[0], 3, *mask.shape[2:])  # 扩展 mask，使其与 img1 形状一致

    diff = (img1 - img2) ** 2
    mse = (diff * mask_exp).sum() / (mask_exp.sum() + 1e-8)
    
    # 计算 PSNR
    psnr_value = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr_value


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def cal_gradient(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient


def bilateral_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad).exp() * mask).mean()

    return smooth_loss


def second_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs() * torch.exp(-10*spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()


def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def first_order_edge_aware_norm_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].norm(dim=1, keepdim=True))).sum(1).mean()

def first_order_loss(data):
    return spatial_gradient(data[None], order=1)[0].abs().sum(1).mean()

def tv_loss(depth):
    # return spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs().sum(1).mean()
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    return h_tv + w_tv
