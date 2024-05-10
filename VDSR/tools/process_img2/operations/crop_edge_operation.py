from PIL import Image
import os.path as osp

from ...utils import scandir
from .base_operation import BaseOperation

class CropEdgeOperation(BaseOperation):
    def __init__(self, opt):
        super(CropEdgeOperation, self).__init__(opt)
        self.base_size = opt.get('base_size')
        self.crop_width = opt.get('crop_width')
        self.crop_height = opt.get('crop_height')
        self.center = opt['center']

    def operate(self, path, img=None):
        basename, ext = osp.splitext(osp.basename(path))
        # eg. self.save_imgname_tmpl='{}' -> lr_pic_name={basename}{ext}
        # eg. self.save_imgname_tmpl='in_{}' -> lr_pic_name=in_{basename}{ext}
        cropped_img_name = f'{self.save_imgname_tmpl.format(basename)}{ext}'
        cropped_img_path = osp.join(self.save_dir, cropped_img_name)

        if img == None:
            img = Image.open(path, mode='r')
        img = img.convert('RGB')
        if self.base_size == None:
            cropped_width = min(self.crop_width, int(img.width))
            cropped_height = min(self.crop_height, int(img.height))
        else:
            cropped_width = int(img.width / self.base_size) * self.base_size
            cropped_height = int(img.height / self.base_size) * self.base_size
        if cropped_width!=img.width or cropped_height!=img.height:
            if self.center:
                crop_w_start = (img.width-cropped_width) / 2
                crop_h_start = (img.height-cropped_height) / 2
                img = img.crop((crop_w_start, crop_h_start, crop_w_start+cropped_width, crop_h_start+cropped_height))
            else:
                img = img.crop((0, 0, cropped_width, cropped_height))

        if self.save:
            img.save(cropped_img_path)

        return cropped_img_path, img

