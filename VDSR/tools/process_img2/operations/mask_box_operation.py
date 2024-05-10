import os
import os.path as osp
import PIL.Image as pil_image
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from ...utils import scandir
from .base_operation import BaseOperation

class MaskBoxOperation(BaseOperation):
    ''' 
    根据 json_path 得到 mask box 的范围并生成 mask -> mask_img
    将 mask_img 膨胀 -> mask_d_img
    使用 mask_d_img 分割原图 -> out_img
    '''
    def __init__(self, opt):
        super(MaskBoxOperation, self).__init__(opt)
        self.mask_save_imgname_tmpl = opt['mask_save_imgname_tmpl']
        self.mask_save_dir = opt['mask_save_dir']
        self.json_path = osp.join(self.work_dir, opt['json_path'])
        self.structure_element_kernel_d_size = opt['structure_element_kernel_d_size']

    def operate(self, path, img=None):
        ''' 
        img = (origin_img, mask_box)
        mask_box  存储 mask 信息
        ''' 
        if img == None:
            origin_img = None
            mask_box = None
        else:
            origin_img, mask_box = img
            if mask_box == []:
                # return out_path, [out_img, mask_img, mask_d_img, boxs, boxs_d]
                return None, [None, None, None, [], []]

        
        basename, ext = osp.splitext(osp.basename(path))
        # set out img name
        out_name = f'{self.save_imgname_tmpl.format(basename)}{ext}'
        out_path = osp.join(self.save_dir, out_name)
        # set out mask name
        out_mask_name = f'{self.mask_save_imgname_tmpl.format(basename)}{ext}'
        mask_basename, ext = osp.splitext(osp.basename(out_mask_name))
        mask_save_path = osp.join(self.mask_save_dir, out_mask_name)
        mask_d_save_name = mask_basename + '_d' + ext
        mask_d_save_path = osp.join(self.mask_save_dir, mask_d_save_name)    # 膨胀

        # read img
        if origin_img == None:
            origin_img = pil_image.open(path).convert('RGB')
        
        if mask_box == None:
            with open(self.json_path) as json_file:
                mask_box = json_file.read()
        mask_img = np.zeros((origin_img.size[1], origin_img.size[0])) # 二维数组
        mask_d_img = np.zeros((origin_img.size[1], origin_img.size[0])) # 二维数组
        boxs = []
        boxs_d = []
        if isinstance(mask_box, list):
            # x_min, y_min, x_max, y_max
            boxs = mask_box
        else:
            parsed_json = json.loads(mask_box)
            for box_item in parsed_json:
                if 'box' in box_item:
                    boxs.append([int(x) for x in box_item['box']])
        for box in boxs:
            box_d = self._dilateMaskBox(
                box, 
                n=self.structure_element_kernel_d_size, 
                x_constraint=(0, origin_img.size[0]), 
                y_constraint=(0, origin_img.size[1]))
            boxs_d.append(box_d)
            mask_img[box[1]:box[3], box[0]:box[2]] = 255
            mask_d_img[box_d[1]:box_d[3], box_d[0]:box_d[2]] = 255
        mask_img = pil_image.fromarray(mask_img.astype(np.uint8))  
        mask_d_img = pil_image.fromarray(mask_d_img.astype(np.uint8))  

        # convert
        out_img = self._convertImage2(mask_d_img, origin_img)

        # save
        if self.save:
            out_img.save(out_path)
            mask_img.save(mask_save_path)
            mask_d_img.save(mask_d_save_path)

        return out_path, [out_img, mask_img, mask_d_img, boxs, boxs_d]

    def _dilateMask(self, mask_gray, n=3, structure_element = cv2.MORPH_CROSS):
        ''' 膨胀
        n: 结构元大小
        structure_element: 结构元类型
        '''
        # 得到一个结构元，实际就是numpy矩阵 
        # MORPH_CROSS 十字交叉结构元  MORPH_ELLIPSE 椭圆结构元
        kernel = cv2.getStructuringElement(structure_element, (n, n))
        dilate_res = cv2.dilate(mask_gray, kernel)  # 膨胀结果
        return dilate_res

    def _erodeMask(self, mask_gray, n=3, structure_element = cv2.MORPH_CROSS):
        ''' 腐蚀
        n: 结构元大小
        structure_element: 结构元类型
        '''
        # 得到一个结构元，实际就是numpy矩阵 
        # MORPH_CROSS 十字交叉结构元  MORPH_ELLIPSE 椭圆结构元
        kernel = cv2.getStructuringElement(structure_element, (n, n)) 
        erode_res = cv2.erode(mask_gray, kernel)  # 腐蚀结果
        return erode_res

    def _dilateMaskBox(self, mask_box, n=3, x_constraint=None, y_constraint=None):
        # x_min, y_min, x_max, y_max
        # x_constraint eg: (0, 1000)
        mask_box_d = [x for x in mask_box]
        if x_constraint:
            mask_box_d[0] = max(mask_box[0]-n, x_constraint[0])
            mask_box_d[2] = min(mask_box[2]+n, x_constraint[1])
        else:
            mask_box_d[0] = mask_box[0]-n
            mask_box_d[2] = mask_box[2]+n
        if y_constraint:
            mask_box_d[1] = max(mask_box[1]-n, y_constraint[0])
            mask_box_d[3] = min(mask_box[3]+n, y_constraint[1])
        else:
            mask_box_d[1] = mask_box[1]-n
            mask_box_d[3] = mask_box[3]+n
        
        return mask_box_d

    def _convertImage(self, mask_img, origin_img):
        # numpy
        mask_np = np.array(mask_img).astype(np.uint8)
        origin_np = np.array(origin_img).astype(np.uint8)
        origin_height = origin_np.shape[0]
        origin_width = origin_np.shape[1]
        # out_np = np.array([[origin_np[y, x] if (mask_np[y,x]!=[68, 0, 83]).all() else [0, 0, 0] for x in range(origin_width)] for y in range(origin_height)]).astype(np.uint8)
        out_np = np.zeros((origin_height, origin_width, 3), dtype=np.uint8)
        # dilate the mask
        dilate_res = self._dilateMask(mask_np, self.structure_element_kernel_d_size, cv2.MORPH_ELLIPSE)
        _, dilate_res = cv2.threshold(dilate_res, 40, 255, cv2.THRESH_BINARY)

        for y in range(origin_height):
            for x in range(origin_width):
                if (dilate_res[y,x]>40):
                    out_np[y, x] = origin_np[y, x]
                else:
                    out_np[y, x] = [0, 0, 0]
        out_img = pil_image.fromarray(out_np)
        mask_d_img = pil_image.fromarray(dilate_res)
        return out_img, mask_d_img
    
    def _convertImage2(self, mask_img, origin_img):
        # numpy
        mask_np = np.array(mask_img).astype(np.uint8)
        origin_np = np.array(origin_img).astype(np.uint8)
        origin_height = origin_np.shape[0]
        origin_width = origin_np.shape[1]
        # out_np = np.array([[origin_np[y, x] if (mask_np[y,x]!=[68, 0, 83]).all() else [0, 0, 0] for x in range(origin_width)] for y in range(origin_height)]).astype(np.uint8)
        out_np = np.zeros((origin_height, origin_width, 3), dtype=np.uint8)
        # dilate the mask
        _, mask_np = cv2.threshold(mask_np, 40, 255, cv2.THRESH_BINARY)

        for y in range(origin_height):
            for x in range(origin_width):
                if (mask_np[y,x]>40):
                    out_np[y, x] = origin_np[y, x]
                else:
                    out_np[y, x] = [0, 0, 0]
        out_img = pil_image.fromarray(out_np)
        return out_img
