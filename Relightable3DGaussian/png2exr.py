import OpenEXR
import Imath
import numpy as np
from PIL import Image
import os

def load_png(path, linearize=True):
    """Load PNG and convert to float32 numpy array (H,W,3)"""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    if linearize:
        # sRGB to linear
        arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return arr

def save_exr(path, arr):
    """Save float32 numpy array (H,W,3) as EXR"""
    H, W, C = arr.shape
    assert C == 3, "Only 3-channel images supported"
    
    header = OpenEXR.Header(W, H)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exr = OpenEXR.OutputFile(path, header)
    
    # OpenEXR expects channel-wise data as bytes
    R = arr[:, :, 0].astype(np.float32).tobytes()
    G = arr[:, :, 1].astype(np.float32).tobytes()
    B = arr[:, :, 2].astype(np.float32).tobytes()
    
    exr.writePixels({'R': R, 'G': G, 'B': B})
    exr.close()

def convert_png_to_exr(png_path, exr_path, linear=True):
    arr = load_png(png_path, linearize=linear)
    save_exr(exr_path, arr)
    print(f"Saved EXR to {exr_path}")

if __name__ == "__main__":
    # 直接指定输入输出路径
    input_png = "/home/chengwr/code/Relightable3DGaussian/env_map/doubao1.png"
    output_exr = "/home/chengwr/code/Relightable3DGaussian/env_map/doubao1.exr"
    
    convert_png_to_exr(input_png, output_exr, linear=True)

