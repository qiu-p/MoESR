from PIL import Image
import os.path as osp

from ...utils import scandir
from ...utils.registry import IMG_PROCESS_REGISTRY
from .base_operation import BaseOperation


@IMG_PROCESS_REGISTRY.register()
def improve_resolution(path, opt, img=None):
    filename_tmpl = opt['filename_tmpl']
    save_folder = opt['save_folder']
    scaling_factor = opt['improve_scaling_factor']

    basename, ext = osp.splitext(osp.basename(path))
    # eg. filename_tmpl='{}' -> lr_pic_name={basename}{ext}
    # eg. filename_tmpl='in_{}' -> lr_pic_name=in_{basename}{ext}
    hr_pic_name = f'{filename_tmpl.format(basename)}{ext}'
    hr_pic_path = osp.join(save_folder, hr_pic_name)

    if img == None:
        img = Image.open(path, mode='r')
    img = img.convert('RGB')
    # (BICUBIC)
    hr_img = img.resize((int(img.width*scaling_factor), int(img.height*scaling_factor)), 
                        Image.BICUBIC)

    hr_img.save(hr_pic_path)

    return hr_pic_path, hr_img