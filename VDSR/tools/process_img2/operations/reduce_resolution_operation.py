from PIL import Image
import os.path as osp

from ...utils import scandir
from .base_operation import BaseOperation


class ReduceResolutionOperation(BaseOperation):
    def __init__(self, opt):
        super(ReduceResolutionOperation, self).__init__(opt)
        self.scaling_factor = opt['reduce_scaling_factor']
        self.crop = opt['rr_crop']


    def _generate_lr_pic(self, hr_img, scaling_factor, crop=False):
        if crop:
            cropped_width = int(hr_img.width / scaling_factor) * scaling_factor
            cropped_height = int(hr_img.height / scaling_factor) * scaling_factor
            if cropped_width!=hr_img.width or cropped_height!=hr_img.height:
                hr_img = hr_img.crop((0, 0, cropped_width, cropped_height))
        else:
            # 所有图像必须保持固定的分辨率以此保证能够整除放大比例
            assert hr_img.width % scaling_factor == 0, "尺寸宽不能被比例因子整除!"
            assert hr_img.height % scaling_factor == 0, "尺寸高不能被比例因子整除!"
        # 下采样（双三次差值）
        lr_img = hr_img.resize((int(hr_img.width / scaling_factor),
                                        int(hr_img.height / scaling_factor)),
                                    Image.BICUBIC)
        # 安全性检查
        assert hr_img.width == lr_img.width * scaling_factor and hr_img.height == lr_img.height * scaling_factor
        return lr_img


    def operate(self, path, img=None):
        basename, ext = osp.splitext(osp.basename(path))
        lr_img_name = f'{self.save_imgname_tmpl.format(basename)}{ext}'
        lr_img_path = osp.join(self.save_dir, lr_img_name)

        if img == None:
            img = Image.open(path, mode='r')
        img = img.convert('RGB')
        lr_img = self._generate_lr_pic(img, self.scaling_factor, self.crop)
        
        if self.save:
            lr_img.save(lr_img_path)

        return lr_img_path, lr_img
