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

def convertImage(mask_path, origin_path, n_e, n_d):
    # read img
    mask_img = pil_image.open(mask_path).convert('RGB')
    origin_img = pil_image.open(origin_path).convert('RGB')
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
    out_np_g = np.zeros((origin_height, origin_width, 3), dtype=np.uint8)
    # 颜色空间转换，读取灰度图
    mask2_gray = cv2.cvtColor(mask2_np, cv2.COLOR_RGB2GRAY)
    _, mask2_bi = cv2.threshold(mask2_gray, 40, 255, cv2.THRESH_BINARY)
    # cv2.imshow(mask2_bi, "bi")
    # cv2.waitKey(0)
    erode_res1, _ = dilateMask(mask2_bi, n_e, cv2.MORPH_ELLIPSE)
    _, dilate_res2 = dilateMask(erode_res1, n_e+n_d, cv2.MORPH_ELLIPSE)
    erode_res3, _ = dilateMask(dilate_res2, n_d, cv2.MORPH_ELLIPSE)
    _, dilate_res2 = cv2.threshold(dilate_res2, 40, 255, cv2.THRESH_BINARY)
    _, erode_res1 = cv2.threshold(erode_res1, 40, 255, cv2.THRESH_BINARY)
    _, erode_res3 = cv2.threshold(erode_res3, 40, 255, cv2.THRESH_BINARY)

    for y in range(origin_height):
        for x in range(origin_width):
            if (dilate_res2[y,x]>40):
                out_np[y, x] = origin_np[y, x]
                out_np_g[y,x] = [0, 0, 0]
            else:
                out_np[y, x] = [0, 0, 0]
                out_np_g[y,x] = origin_np[y, x]
    out_img = pil_image.fromarray(out_np)
    out_img_g = pil_image.fromarray(out_np_g)
    mask2_e_img = pil_image.fromarray(erode_res1)
    mask2_ed_img = pil_image.fromarray(dilate_res2)
    mask2_ede_img = pil_image.fromarray(erode_res3)
    return out_img, mask2_e_img, mask2_ed_img, mask2_ede_img, out_img_g

@IMG_PROCESS_REGISTRY.register()
def masklhy(path, opt):
    # eg. filename_tmpl='{}' -> lr_pic_name={basename}{ext}
    # eg. filename_tmpl='in_{}' -> lr_pic_name=in_{basename}{ext}
    filename_tmpl = opt['filename_tmpl']
    save_folder = opt['save_folder']
    mask_name_tmpl = opt['mask_name_tmpl']
    mask_name = opt['mask_name']
    mask_save_name = opt['mask_save_name']
    mask_foder = opt['mask_foder']
    mask_save_folder = opt['mask_save_folder']
    structure_element_kernel_e_size = opt['structure_element_kernel_e_size']
    structure_element_kernel_d_size = opt['structure_element_kernel_d_size']

    basename, ext = osp.splitext(osp.basename(path))
    # the name of mask
    if mask_name_tmpl != None:
        mask_name = f'{mask_name_tmpl.format(basename)}{ext}'
    mask_path = osp.join(mask_foder, mask_name)
    out_name = f'{filename_tmpl.format(basename)}{ext}'
    out_path = osp.join(save_folder, out_name)
    mask2_path = osp.join(mask_save_folder, mask_save_name)
    mask3_save_name = 'e_' + mask_save_name
    mask3_path = osp.join(mask_save_folder, mask3_save_name)

    # convert
    out_img, _, mask2_img, mask3_img, out_img_g = convertImage(mask_path, path, n_e=structure_element_kernel_e_size, n_d=structure_element_kernel_d_size)

    # save
    out_img.save(out_path)
    mask2_img.save(mask2_path)
    mask3_img.save(mask3_path)
    out_img_g.save(path)
    return out_path
