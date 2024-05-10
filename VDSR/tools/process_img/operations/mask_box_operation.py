import os
import os.path as osp
import PIL.Image as pil_image
import cv2
import json
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

def convertImage(mask_img, origin_img, n_e, n_d):
    # show info
    # print("origin size (W H):", origin_img.size)
    # print("mask size (W H):", mask_img.size)
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
    # cv2.imshow(mask2_bi, "bi")
    # cv2.waitKey(0)
    _, dilate_res = dilateMask(mask2_np, n_d, cv2.MORPH_ELLIPSE)
    _, dilate_res = cv2.threshold(dilate_res, 40, 255, cv2.THRESH_BINARY)

    for y in range(origin_height):
        for x in range(origin_width):
            if (dilate_res[y,x]>40):
                out_np[y, x] = origin_np[y, x]
            else:
                out_np[y, x] = [0, 0, 0]
    out_img = pil_image.fromarray(out_np)
    mask2_img = pil_image.fromarray(mask2_np)
    mask2_d_img = pil_image.fromarray(dilate_res)
    return out_img, mask2_img, mask2_d_img

@IMG_PROCESS_REGISTRY.register()
def mask_box(path, opt, origin_img=None):
    # eg. filename_tmpl='{}' -> lr_pic_name={basename}{ext}
    # eg. filename_tmpl='in_{}' -> lr_pic_name=in_{basename}{ext}
    filename_tmpl = opt['filename_tmpl']
    save_folder = opt['save_folder']
    mask_name_tmpl = opt['mask_name_tmpl']
    mask_name = opt.get('mask_name')
    mask_foder = opt['mask_foder']
    mask_save_name = opt['mask_save_name']
    mask_save_folder = opt.get('mask_save_folder')
    json_path = opt['json_path']
    structure_element_kernel_e_size = opt['structure_element_kernel_e_size']
    structure_element_kernel_d_size = opt['structure_element_kernel_d_size']

    basename, ext = osp.splitext(osp.basename(path))
    # the name of mask
    if mask_name == None:
        mask_name = f'{mask_name_tmpl.format(basename)}{ext}'
    mask_path = osp.join(mask_foder, mask_name)
    out_name = f'{filename_tmpl.format(basename)}{ext}'
    out_path = osp.join(save_folder, out_name)
    # set mask paths
    if mask_save_folder == None:
        mask_save_folder = f'{mask_save_folder_tmpl.format(basename)}{ext}'
    mask_basename, ext = osp.splitext(osp.basename(mask_save_name))
    mask_path = osp.join(mask_save_folder, mask_save_name)
    mask_d_save_name = mask_basename + '_d' + ext
    mask_d_path = osp.join(mask_save_folder, mask_d_save_name)

    # read img
    if origin_img == None:
        origin_img = pil_image.open(path).convert('RGB')
    
    with open(json_path) as json_file:
        file_contents = json_file.read()
    parsed_json = json.loads(file_contents)
    boxs = []
    for box_item in parsed_json:
        if 'box' in box_item:
            boxs.append(box_item['box'])
    mask_img = np.zeros((origin_img.size[1], origin_img.size[0])) # 二维数组
    for box in boxs:
        box = [int(x) for x in box]
        mask_img[box[1]:box[3], box[0]:box[2]] = 255
    mask_img = mask_img.astype(np.uint8)
    mask_img = pil_image.fromarray(mask_img)  

    # convert
    out_img, mask_img, mask_d_img = convertImage(mask_img, origin_img, n_e=structure_element_kernel_e_size, n_d=structure_element_kernel_d_size)

    # save
    out_img.save(out_path)
    mask_img.save(mask_path)
    mask_d_img.save(mask_d_path)

    return out_path, out_img
