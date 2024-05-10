import sys
sys.path.append("./BasicSR_master/")

from seg_rec import Seg_RecPic
from VDSR.pic_SRBlend import interface_all
import os

from VDSR.tools.psnr import psnr

# gt_dir = './test_4/'
# lr_dir = './test_bicubic4/'
gt_dir = './dataset/originGT/'
lr_dir = './dataset/originLRx4/'
work_dir = './dataset/set3/'
save_dir = './dataset/set3/sr_result/'
model_path = "./yolov8x-oiv7.pt"

if __name__ == "__main__":
    imgs_list = {"car" : [],"plane" : [],"dog" : [],"cat" : [], "people" : []}
    for filename in os.listdir(lr_dir):
        print(filename)
        testpicture = Seg_RecPic(lr_dir+filename, model_path, gt_dir+filename)
        testpicture.seg_pic()
        for keys,vals in testpicture.img_dic.items():
            imgs_list[keys].append(vals)

    if work_dir.endswith('/'):
        work_dir = work_dir[:-1]
    if lr_dir.endswith('/'):
        lr_dir = lr_dir[:-1]
    if gt_dir.endswith('/'):
        gt_dir = gt_dir[:-1]
    if save_dir.endswith('/'):
        save_dir = save_dir[:-1]
              
    interface_all(imgs_list, 4, work_dir, save_dir, lr_dir, gt_dir, is_gpu=True, is_mask_gt=False, model_kind='EDSR')
