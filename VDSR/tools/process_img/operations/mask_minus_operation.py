import os
import os.path as osp
import PIL.Image as pil_image
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ...utils import scandir
from ...utils.registry import IMG_PROCESS_REGISTRY
from .base_operation import BaseOperation

def dilateMask(mask_gray, n=3, structure_element = cv2.MORPH_CROSS):
    '''
    n: 结构元大小
    structure_element: 结构元类型
    '''
    # 得到一个结构元，实际就是numpy矩阵 
    # MORPH_CROSS 十字交叉结构元  MORPH_ELLIPSE 椭圆结构元
    kernel = cv2.getStructuringElement(structure_element, (n, n)) 
    erode_res = cv2.erode(mask_gray, kernel)  # 腐蚀结果
    dilate_res = cv2.dilate(mask_gray, kernel)  # 膨胀结果
    return erode_res, dilate_res

def convertImage(mask_path, origin_img, n):
    # read img
    mask_img = pil_image.open(mask_path).convert('RGB')
    # show info
    print("origin size (W H):", origin_img.size)
    print("mask size (W H):", mask_img.size)
    # resize
    new_mask_width = int(origin_img.width)
    new_mask_height = int(origin_img.height)
    mask2_img = mask_img.resize((new_mask_width, new_mask_height), pil_image.BICUBIC)
    # numpy
    mask2_np = np.array(mask2_img).astype(np.uint8)
    origin_np = np.array(origin_img).astype(np.uint8)
    origin_height = origin_np.shape[0]
    origin_width = origin_np.shape[1]
    # out_np = np.array([[origin_np[y, x] if (mask2_np[y,x]!=[68, 0, 83]).all() else [0, 0, 0] for x in range(origin_width)] for y in range(origin_height)]).astype(np.uint8)
    out_np = np.zeros((origin_height, origin_width, 3), dtype=np.uint8)
    # 颜色空间转换，读取灰度图
    mask2_gray = cv2.cvtColor(mask2_np, cv2.COLOR_RGB2GRAY)
    _, dilate_res1 = dilateMask(mask2_gray, n, cv2.MORPH_ELLIPSE)
    erode_res2, _ = dilateMask(erode_res1, 2*n, cv2.MORPH_ELLIPSE)

    for y in range(origin_height):
        for x in range(origin_width):
            if (erode_res2[y,x]<=40):
                out_np[y, x] = origin_np[y, x]
            else:
                out_np[y, x] = [0, 0, 0]
    out_img = pil_image.fromarray(out_np)
    mask2_d_img = pil_image.fromarray(dilate_res1)
    mask2_de_img = pil_image.fromarray(erode_res2)
    return out_img, mask2_e_img, mask2_ed_img

@IMG_PROCESS_REGISTRY.register()
def mask_minus(path, opt):
    # eg. filename_tmpl='{}' -> lr_pic_name={basename}{ext}
    # eg. filename_tmpl='in_{}' -> lr_pic_name=in_{basename}{ext}
    filename_tmpl = opt['filename_tmpl']
    save_folder = opt['save_folder']
    mask_foder = opt['mask_foder']
    mask_save_folder = opt['mask_save_folder']
    mask_name_tmpl = opt.get('mask_name_tmpl')
    mask_names = opt.get('mask_names')
    out_mask_name = opt['out_mask_name']
    structure_element_kernel_size = opt['structure_element_kernel_size']

    basename, ext = osp.splitext(osp.basename(path))
    out_name = f'{filename_tmpl.format(basename)}{ext}'
    out_path = osp.join(save_folder, out_name)
    mask2_path = osp.join(mask_save_folder, out_mask_name)
    print("path", path)
    print("out_path", out_path)
    print("mask2_path", mask2_path)
    # the name of mask
    if mask_name_tmpl != None:
        mask_names = f'{mask_name_tmpl.format(basename)}{ext}'

    out_img = pil_image.open(path).convert('RGB')
    print("out_img ok", mask_names)
    for mask_name in mask_names:
        print("mask_path", mask_path)
        mask_path = osp.join(mask_foder, mask_name)
        # out_img, _, mask2_img = convertImage(mask_path, out_img, n=structure_element_kernel_size)
    print("mask_name ok", out_img.size)

    # save
    out_img.save(out_path)
    # mask2_img.save(mask2_path)

    return out_path
