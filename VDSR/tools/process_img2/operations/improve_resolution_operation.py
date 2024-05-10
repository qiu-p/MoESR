from PIL import Image
import os.path as osp

from ...utils import scandir
from .base_operation import BaseOperation


class ImproveResolutionOperation(BaseOperation):
    def __init__(self, opt):
        super(ImproveResolutionOperation, self).__init__(opt)
        self.scaling_factor = opt['improve_scaling_factor']

    def operate(self, path, img=None):
        basename, ext = osp.splitext(osp.basename(path))
        hr_pic_name = f'{self.save_imgname_tmpl.format(basename)}{ext}'
        hr_pic_path = osp.join(self.save_dir, hr_pic_name)

        if img == None:
            img = Image.open(path, mode='r')
        img = img.convert('RGB')
        # (BICUBIC)
        hr_img = img.resize((int(img.width*self.scaling_factor), int(img.height*self.scaling_factor)), 
                            Image.BICUBIC)

        if self.save:
            hr_img.save(hr_pic_path)

        return hr_pic_path, hr_img