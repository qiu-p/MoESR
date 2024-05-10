from PIL import Image
import os.path as osp

from ...utils import scandir
from ...utils.registry import IMG_PROCESS_REGISTRY
from .base_operation import BaseOperation

@IMG_PROCESS_REGISTRY.register()
class CropEdgeOperation(BaseOperation):
    def __init__(self, opt):
        super(CropEdgeOperation, self).__init__(opt)
        self.base_size = opt['base_size']
        self.center = opt['center']

    def operate(path):
        basename, ext = osp.splitext(osp.basename(path))
        # eg. filename_tmpl='{}' -> lr_pic_name={basename}{ext}
        # eg. filename_tmpl='in_{}' -> lr_pic_name=in_{basename}{ext}
        cropped_pic_name = f'{self.filename_tmpl.format(basename)}{ext}'
        cropped_pic_path = osp.join(self.save_folder, cropped_pic_name)

        img = Image.open(path, mode='r')
        img = img.convert('RGB')
        cropped_width = int(img.width / self.base_size) * self.base_size
        cropped_height = int(img.height / self.base_size) * self.base_size
        if cropped_width!=img.width or cropped_height!=img.height:
            if self.center:
                crop_w_start = (img.width-cropped_width) / 2
                crop_h_start = (img.height-cropped_height) / 2
                img = img.crop((crop_w_start, crop_h_start, crop_w_start+cropped_width, crop_h_start+cropped_height))
            else:
                img = img.crop((0, 0, cropped_width, cropped_height))

        img.save(cropped_pic_path)

        return cropped_pic_path


@IMG_PROCESS_REGISTRY.register()
def crop_edge(path, opt, img=None):
    """cope the edge of img
    - Return:
      - new_path
      - new_img
    """
    filename_tmpl = opt['filename_tmpl']
    save_folder = opt['save_folder']
    base_size = opt.get('base_size')
    crop_width = opt.get('crop_width')
    crop_height = opt.get('crop_height')
    center = opt['center']

    basename, ext = osp.splitext(osp.basename(path))
    # eg. filename_tmpl='{}' -> lr_pic_name={basename}{ext}
    # eg. filename_tmpl='in_{}' -> lr_pic_name=in_{basename}{ext}
    cropped_img_name = f'{filename_tmpl.format(basename)}{ext}'
    cropped_img_path = osp.join(save_folder, cropped_img_name)

    if img == None:
        img = Image.open(path, mode='r')
    img = img.convert('RGB')
    if base_size == None:
        cropped_width = min(crop_width, int(img.width))
        cropped_height = min(crop_height, int(img.height))
    else:
        cropped_width = int(img.width / base_size) * base_size
        cropped_height = int(img.height / base_size) * base_size
    if cropped_width!=img.width or cropped_height!=img.height:
        if center:
            crop_w_start = (img.width-cropped_width) / 2
            crop_h_start = (img.height-cropped_height) / 2
            img = img.crop((crop_w_start, crop_h_start, crop_w_start+cropped_width, crop_h_start+cropped_height))
        else:
            img = img.crop((0, 0, cropped_width, cropped_height))

    img.save(cropped_img_path)

    return cropped_img_path, img
