from PIL import Image
import os.path as osp
import cv2
import numpy as np

from ...utils import scandir
from ...utils.registry import IMG_PROCESS_REGISTRY
from .base_operation import BaseOperation

# 将 1 张图片裁成 crop*crop 大小的若干张小图片
# 提取子图像
@IMG_PROCESS_REGISTRY.register()
def extract_subimages(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    filename_tmpl = opt['filename_tmpl']
    # 被裁剪后的图片的尺寸
    crop_size = opt['crop_size']
    # 裁剪时不同块之间的步长大小
    step = opt['step']
    # 阈值size， 比 thresh_size 小的补丁的会被舍弃
    thresh_size = opt['thresh_size']
    # 如 'test.txt' -> 'test', '.txt'
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # cv2.imshow('img', img)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    sub_img_paths = []
    for x in h_space:
        for y in w_space:
            index += 1
            sub_img = img[x:x + crop_size, y:y + crop_size, ...]
            # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
            sub_img = np.ascontiguousarray(sub_img)
            basename = f'{img_name}_s{index:03d}'
            sub_img_name = f'{filename_tmpl.format(basename)}{extension}'
            sub_img_path = osp.join(opt['save_folder'], sub_img_name)
            sub_img_paths.append(sub_img_path)
            # 有损压缩格式（如.jpg）只能有损保存图片，可选损失程度
            # 无损压缩格式（如.png）就能无损保存图片，可选压缩程度
            # cv2.CV_IMWRITE_JPEG_QUALITY：设置 .jpeg/.jpg 格式的图片质量，取值为 0-100（默认值 95），数值越大则图片质量越高；
            # cv2.CV_IMWRITE_PNG_COMPRESSION：设置 .png 格式图片的压缩比，取值为 0-9（默认值 3），数值越大则无损压缩比越大，0表示不压缩直接储存。默认值为1（最佳速度设置）
            cv2.imwrite(
                sub_img_path, sub_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    
    return sub_img_paths