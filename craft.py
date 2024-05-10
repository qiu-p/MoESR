import PIL.Image as pil_image
import os.path as osp
import os

from seg_rec import Seg_RecPic

from VDSR.tools.psnr import psnr
from VDSR.my_test import sr_img_dif_tag, sr_img_dif_tags

from BasicSR_master.my_test_BSR import sr_img
from BasicSR_master.my_test_BSR import sr_img_dif_tag_bsr

is_gpu = True
img_input = 'dataset/set01/originLR/0.png'
model_dir = 'VDSR/model'
out_path = 'dataset/set01/0_gpu.png'

gt_dir = './dataset/set02/origin0GT'
lr_dir = './dataset/set02/origin0LR'
sr_dir = './dataset/set02/originLR_sr'
origin_imgs = ['./dataset/set02/origin0LR/00.png', './dataset/set02/origin0LR/01.png', './dataset/set02/origin0LR/02.png', './dataset/set02/origin0LR/03.png', './dataset/set02/origin0LR/04.png', './dataset/set02/origin0LR/05.png', './dataset/set02/origin0LR/06.png', './dataset/set02/origin0LR/07.png', './dataset/set02/origin0LR/08.png', './dataset/set02/origin0LR/09.png']
blended_img_paths = ['./sr_result/blend_00.png', './sr_result/blend_01.png', './sr_result/blend_02.png', './sr_result/blend_03.png', './sr_result/blend_04.png', './sr_result/blend_05.png', './sr_result/blend_06.png', './sr_result/blend_07.png', './sr_result/blend_08.png', './sr_result/blend_09.png']


def test_bsr():
    sr_img_dif_tag_bsr(
        tag = 'origin',
        img_input='BasicSR_master/datasets/set01/lr/0.png', 
        model_dir='BasicSR_master/experiments/pretrained_models/my', 
        save_path='ysh_testpic2.png')
    
def test_psnr():
    # PSNR
    process_img_num = len(origin_imgs)
    print('PSNR...')
    print('origin_img_paths: {}'.format(origin_imgs))
    print('blended_img_paths: {}'.format(blended_img_paths))
    for i in range(0, process_img_num):
        origin_picname = os.path.basename(origin_imgs[i])
        gt_img_path = osp.join(gt_dir, origin_picname)
        sred_img_path = osp.join(sr_dir, origin_picname)
        
        # psnr calculation
        # 原图与blend
        psnr_blend = psnr(
            img1=gt_img_path,
            img2=blended_img_paths[i])
        # 原图与仅mixture
        psnr_no = psnr(
            img1=gt_img_path,
            img2=sred_img_path)
        print('PSNR of {}: \t\tpsnr blend: {} \t\tpsnr total: {}'.format(origin_picname, psnr_blend, psnr_no))

if __name__ == '__main__':
    print('===== START =====')
    # out = sr_img_dif_tag('origin', img_input, model_dir, is_gpu)
    # out.save(out_path)
    
    # test_bsr()
    # test_psnr()
    
    test_type = 'cat'
    img_num = '06'
    gt_img_path = 'dataset/set02/maskedGT/{}_masked/{}.png'.format(test_type, img_num)
    img_input = 'dataset/set02/{}_masked/{}.png'.format(test_type, img_num)
    tags = ['origin', 'car', 'cat', 'dog', 'people', 'plane']
    img_outputs = sr_img_dif_tags(tags, img_input, model_dir, is_gpu)
    print(img_outputs)
    print('test img: {} {}'.format(test_type, img_num))
    for tag, img_output in img_outputs.items():
        psnr_get = psnr(
            img1=gt_img_path,
            img2=img_output)
        print('tag: {} \t\t psnr: {}'.format(tag, psnr_get))
           
    
    
    print('=====  END  =====')
    