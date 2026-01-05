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

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

# def searchForMaxIteration(folder):
#     saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
#     return max(saved_iters)
def searchForMaxIteration(folder):
    """查找文件夹中最大的迭代数（修复文件名解析逻辑）"""
    saved_iters = []
    for fname in os.listdir(folder):
        # 步骤1：过滤非检查点文件（只处理.pth文件）
        if not fname.endswith(".pth"):
            continue
        # 步骤2：提取迭代数（兼容 chkpnt_30000.pth / chkpnt30000.pth 等格式）
        # 先去掉后缀，再提取数字
        fname_no_ext = os.path.splitext(fname)[0]  # 去掉.pth → chkpnt30000
        # 提取所有数字字符并拼接
        num_str = ''.join([c for c in fname_no_ext if c.isdigit()])
        if num_str:  # 确保提取到数字
            saved_iters.append(int(num_str))
    # 步骤3：返回最大迭代数（无文件时返回0）
    if saved_iters:
        return max(saved_iters)
    else:
        return 0