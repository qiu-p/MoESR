import argparse
import random
import yaml
from collections import OrderedDict
import os
import time
from os import path as osp
import cv2
import PIL.Image as pil_image
from copy import deepcopy
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .tools.utils import get_logger
from .tools.utils import make_dir, is_file_exist
from .tools.utils import parse_options, dict2str, parse_yamls
from .tools.process_img2 import process_imgs
from .tools.process_img2 import blend, blend_img
from .tools.psnr import psnr
from .tools.ssim import calculate_ssim

from .my_test import sr_img_dif_tag

from BasicSR_master.my_test_BSR import sr_img_dif_tag_bsr
from preblend import blend_img

def trcrop(origin_img,xnum,ynum,improve_scale,borderV):
    h = origin_img.size[1]
    w = origin_img.size[0]
    y_step = int(h/ynum) + 1
    x_step = int(w/xnum) + 1
    x1 = 0
    img_bg = np.ones((h*improve_scale,w*improve_scale, 3), dtype=np.uint8) * 255
    img_to_blend = [img_bg]
    while x1 <= w:
        x2 = min(x1+x_step, w)
        y1 = 0
        while y1 <= h:
            y2 = min(y1+y_step, h)
            crop_img = origin_img.crop((max(x1-borderV,0), max(y1-borderV,0), min(x2+borderV,w), min(y2+borderV,h)))
            cordi = [max(x1-borderV,0)*improve_scale, max(y1-borderV,0)*improve_scale, \
                     min(x2+borderV,w)*improve_scale, min(y2+borderV,h)*improve_scale]
            img_to_blend.append([crop_img,cordi])
            y1 = y1 + y_step
        x1 = x1 + x_step
    return img_to_blend
        
def trprocess_imgs(imgs_list, origin_imgs):
    processed_imgs={
        "people" : [],
        "car" : [],
        "plane" : [],
        "dog" : [],
        "cat" : [],
        "origin" : []
    }
    type_list = ['people','car','plane','dog','cat']
    for seq,ori_img in enumerate(origin_imgs):
        origin_img = pil_image.open(ori_img).convert('RGB')
        processed_imgs['origin'].append(origin_img)
        for process_key in type_list:
            if not imgs_list[process_key][seq][1]:
                seg_piclist = None
            else:
                seg_piclist = []
                for cordi in imgs_list[process_key][seq][1]:
                    seg_pic = origin_img.crop(tuple(cordi))
                    seg_piclist.append(seg_pic)
            processed_imgs[process_key].append(seg_piclist)
    return processed_imgs

def interface_all(imgs_list, improve_scale, work_dir, fsave_dir, lr_dir, gt_dir, borderV,is_gpu=False, is_mask_gt=False, model_kind='EDSR'):
    if model_kind!='VDSR' and model_kind!='EDSR' and model_kind!='RCAN':
        print('model_kind error')
        exit()
    if work_dir==None:#如果未指定工作目录，则选择当前所在目录作为工作目录
        work_dir = "./vdsr"
    
    make_dir(work_dir, keep=True)
    
    # [origin_img1, origin_img2, ...]
    # this is lr imgs
    origin_imgs = [values[0] for values in list(imgs_list.values())[0]]
    print('origin_imgs: {}'.format(origin_imgs))
    process_img_num = len(origin_imgs)

    parser = argparse.ArgumentParser(description="PyTorch VDSR")
    parser.add_argument("--mode", type=str, default='', help="mode")
    args = parse_options(parser)
    
    logger = get_logger(log_file='main.log')

    # process_img
    
    processed_imgs = trprocess_imgs(imgs_list, origin_imgs)
    
    # SR
    #logger.info('sr...')
    print('sr...')
    # sred_imgs = {
    # 'origin': [img1],
    # 'dog': [img1],
    # ...}
    sred_imgs = {}
    save_dirs = {
        'origin': "originLR_sr",
        'car': "car_sr",
        'cat': "cat_sr",
        'dog': "dog_sr",
        'people': "people_sr",
        'plane': "plane_sr"
    }
    for process_key, imgs in processed_imgs.items():
        print('sring... {}'.format(process_key))
        sred_imgs[process_key] = []
        save_dir = os.path.join(work_dir, save_dirs[process_key])
        sr_dir = os.path.join(work_dir, save_dirs['origin'])
        make_dir(save_dir, delete=True)
        for i, img in enumerate(imgs):
            if img == None:
                sred_imgs[process_key].append(None)
                continue
            else:
                origin_picname = os.path.basename(origin_imgs[i])
                if process_key == "origin":
                    sub_sr_list = []
                    #logger.info('\tsr {} {}...'.format(process_key, origin_picname))
                    print('\tsr {} {}...'.format(process_key, origin_picname))
                    save_name = '{}_{}.png'.format(origin_picname.split(".")[0], process_key)
                    # save_name = '{}.png'.format(origin_picname.split(".")[0])
                    save_path = os.path.join(save_dir, save_name)
                    crop_imglist = trcrop(img,4,4,improve_scale,borderV)
                    for sseq,subimgl in enumerate(crop_imglist):
                        if sseq == 0:
                            continue
                        subimg = subimgl[0]
                        if model_kind == 'VDSR':
                            subimg_sr = sr_img_dif_tag(process_key, subimg, model_dir='VDSR/model', is_gpu=is_gpu)
                        elif model_kind == 'EDSR':
                            subimg_sr = sr_img_dif_tag_bsr(process_key, 'EDSR', subimg, model_dir='BasicSR_master/experiments/pretrained_models/EDSR')
                        elif model_kind == 'RCAN':
                            subimg_sr = sr_img_dif_tag_bsr(process_key, 'RCAN', subimg, model_dir='BasicSR_master/experiments/pretrained_models/RCAN')
                        subimgl[0] = subimg_sr
                    originLR_sr = blend_img(crop_imglist,improve_scale,borderV)
                    originLR_sr = pil_image.fromarray(originLR_sr)
                    # originLR_sr = pil_image.fromarray(cv2.cvtColor(originLR_sr,cv2.COLOR_BGR2RGB))
                    originLR_sr.save(save_path)
                    sred_imgs[process_key].append(originLR_sr)
                else:    
                    img_sr_list = []
                    for seq,seg_img in enumerate(img):
                        sub_sr_list = []
                        #logger.info('\tsr {} {}_{}...'.format(process_key, origin_picname,seq))
                        print('\tsr {} {}_{}...'.format(process_key, origin_picname,seq))
                        save_name = '{}_{}_{}.png'.format(origin_picname.split(".")[0], process_key,seq)
                        # save_name = '{}.png'.format(origin_picname.split(".")[0])
                        save_path = os.path.join(save_dir, save_name)
                        if seg_img.size[0] >= 200 and seg_img.size[1] >= 200:
                            crop_imglist = trcrop(seg_img,2,2,improve_scale,borderV)
                            for sseq,subimgl in enumerate(crop_imglist):
                                if sseq == 0:
                                    continue
                                subimg = subimgl[0]
                                if model_kind == 'VDSR':
                                    subimg_sr = sr_img_dif_tag(process_key, subimg, model_dir='VDSR/model', is_gpu=is_gpu)
                                elif model_kind == 'EDSR':
                                    subimg_sr = sr_img_dif_tag_bsr(process_key, 'EDSR', subimg, model_dir='BasicSR_master/experiments/pretrained_models/EDSR')
                                elif model_kind == 'RCAN':
                                    subimg_sr = sr_img_dif_tag_bsr(process_key, 'RCAN', subimg, model_dir='BasicSR_master/experiments/pretrained_models/RCAN')
                                subimgl[0] = subimg_sr
                            seg_img_sr = blend_img(crop_imglist,improve_scale,borderV)
                            seg_img_sr = pil_image.fromarray(seg_img_sr)
                        else:
                            if model_kind == 'VDSR':
                                seg_img_sr = sr_img_dif_tag(process_key, seg_img, model_dir='VDSR/model', is_gpu=is_gpu)
                            elif model_kind == 'EDSR':
                                seg_img_sr = sr_img_dif_tag_bsr(process_key, 'EDSR', seg_img, model_dir='BasicSR_master/experiments/pretrained_models/EDSR')
                            elif model_kind == 'RCAN':
                                seg_img_sr = sr_img_dif_tag_bsr(process_key, 'RCAN', seg_img, model_dir='BasicSR_master/experiments/pretrained_models/RCAN')
                        #cv2.imwrite(save_path,seg_img_sr)
                        img_sr_list.append(seg_img_sr)
                    sred_imgs[process_key].append(img_sr_list)

    # blend
    #logger.info('blend...')
    print('blend...')
    blended_imgs = []
    blended_img_paths = []
    make_dir(fsave_dir, rename=True)
    # if os.path.isdir(fsave_dir):
    #     logger.info('save_file already exists.')
    #     print('save_file already exists.')
    #     os._exit(0)
    # else:
    #     os.makedirs(fsave_dir)
    if 'origin' in processed_imgs:
        processed_imgs.pop('origin')
    
    for i in range(0, process_img_num):
        origin_picname = os.path.basename(origin_imgs[i])
        background_img = sred_imgs['origin'][i]
        img_to_blend = [np.array(background_img)]
        blend_save_path = os.path.join(fsave_dir, 'blend_{}'.format(origin_picname))
        print('blend_{}'.format(origin_picname))
        for process_key, imgs in processed_imgs.items():
            if sred_imgs[process_key][i] == None:
                continue
            # logger.info('blend {} {}...'.format(origin_picname, process_key))
            # print('blend {} {}...'.format(origin_picname, process_key))
            for seq,seg_img in enumerate(sred_imgs[process_key][i]):
                img_to_blend.append([np.array(seg_img),[xy*improve_scale for xy in imgs_list[process_key][i][1][seq]]])
            blendimg = pil_image.fromarray(blend_img(img_to_blend,1,borderV))
            blendimg.save(blend_save_path)
        blended_imgs.append(blendimg)
        blended_img_paths.append(blend_save_path)
        
    # PSNR
    #logger.info('PSNR/SSIM...')
    print('PSNR/SSIM...')
    psnr_all_total = 0
    psnr_all_blend = 0
    ssim_all_total = 0
    ssim_all_blend = 0
    for i in range(0, process_img_num):
        origin_picname = os.path.basename(origin_imgs[i])
        gt_img_path = osp.join(gt_dir, origin_picname)
        sred_img_path = osp.join(sr_dir, "{}_origin.png".format(origin_picname.split(".")[0]))
        
        # psnr calculation
        if model_kind == "VDSR":
            psnr_blend = psnr(img1=gt_img_path,
                img2=blended_img_paths[i])
            ssim_blend = calculate_ssim(np.array(pil_image.open(gt_img_path)), np.array(pil_image.open(blended_img_paths[i])))
            psnr_sr = psnr(gt_img_path,
                sred_img_path)
            ssim_sr = calculate_ssim(np.array(pil_image.open(gt_img_path)), np.array(pil_image.open(sred_img_path)))
        elif model_kind == "EDSR":
            psnr_blend = psnr(gt_img_path,
            sred_img_path)
            ssim_blend = calculate_ssim(np.array(pil_image.open(gt_img_path)), np.array(pil_image.open(sred_img_path)))
            psnr_sr = psnr(img1=gt_img_path,
            img2=blended_img_paths[i])
            ssim_sr = calculate_ssim(np.array(pil_image.open(gt_img_path)), np.array(pil_image.open(blended_img_paths[i])))
        elif model_kind == "RCAN":
            psnr_blend = psnr(gt_img_path,
            sred_img_path)
            ssim_blend = calculate_ssim(np.array(pil_image.open(gt_img_path)), np.array(pil_image.open(sred_img_path)))
            psnr_sr = psnr(img1=gt_img_path,
            img2=blended_img_paths[i])
            ssim_sr = calculate_ssim(np.array(pil_image.open(gt_img_path)), np.array(pil_image.open(blended_img_paths[i])))
        psnr_all_total = psnr_all_total+psnr_sr
        psnr_all_blend = psnr_all_blend+psnr_blend
        ssim_all_total = ssim_all_total+ssim_sr
        ssim_all_blend = ssim_all_blend+ssim_blend
        print('PSNR of {}: \t\tpsnr blend: {} \t\tpsnr total: {}'.format(origin_picname, psnr_blend, psnr_sr))
        print('SSIM of {}: \t\tssim blend: {} \t\tssim total: {}'.format(origin_picname, ssim_blend, ssim_sr))
        #logger.info('PSNR of {}: \t\tpsnr blend: {} \t\tpsnr total: {}'.format(origin_picname, psnr_blend, psnr_sr))
        #logger.info('SSIM of {}: \t\tssim blend: {} \t\tssim total: {}'.format(origin_picname, ssim_blend, ssim_sr))
    logger.info("borderV:{}".format(borderV))
    logger.info("psnr_mean_total{},psnr_mean_blend{},ssim_mean_total{},ssim_mean_blend{}".format(psnr_all_total/process_img_num,psnr_all_blend/process_img_num,ssim_all_total/process_img_num,ssim_all_blend/process_img_num))
    return 