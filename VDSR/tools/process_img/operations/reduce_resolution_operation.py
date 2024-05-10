from PIL import Image
import os.path as osp

from ...utils import scandir
from ...utils.registry import IMG_PROCESS_REGISTRY
from .base_operation import BaseOperation


@IMG_PROCESS_REGISTRY.register()
class ReduceResolutionOperation(BaseOperation):
    def __init__(self, opt):
        super(ReduceResolutionOperation, self).__init__(opt)
        self.scaling_factor = opt['reduce_scaling_factor']
        self.crop = opt['rr_crop']


    def _generate_lr_pic(hr_img, scaling_factor, crop=False):
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


    def operate(path, opt):
        basename, ext = osp.splitext(osp.basename(path))
        # eg. filename_tmpl='{}' -> lr_img_name={basename}{ext}
        # eg. filename_tmpl='in_{}' -> lr_img_name=in_{basename}{ext}
        lr_img_name = f'{self.filename_tmpl.format(basename)}{ext}'
        lr_img_path = osp.join(self.save_folder, lr_img_name)

        hr_img = Image.open(path, mode='r')
        hr_img = hr_img.convert('RGB')
        lr_img = _generate_lr_pic(hr_img, self.scaling_factor, self.crop)
        lr_img.save(lr_img_path)

        return lr_img_path


def generate_lr_pic(hr_img, scaling_factor, crop=False):
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

@IMG_PROCESS_REGISTRY.register()
def reduce_resolution(path, opt, img=None):
    filename_tmpl = opt['filename_tmpl']
    save_folder = opt['save_folder']
    scaling_factor = opt['reduce_scaling_factor']
    crop = opt['rr_crop']

    basename, ext = osp.splitext(osp.basename(path))
    # eg. filename_tmpl='{}' -> lr_img_name={basename}{ext}
    # eg. filename_tmpl='in_{}' -> lr_img_name=in_{basename}{ext}
    lr_img_name = f'{filename_tmpl.format(basename)}{ext}'
    lr_img_path = osp.join(save_folder, lr_img_name)

    if img == None:
        img = Image.open(path, mode='r')
    img = img.convert('RGB')
    lr_img = generate_lr_pic(img, scaling_factor, crop)
    lr_img.save(lr_img_path)

    return lr_img_path, lr_img