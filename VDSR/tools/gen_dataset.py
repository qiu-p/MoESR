import PIL.Image as pil_image
import os.path as osp
import numpy as np
import cv2 
import os
import sys
from multiprocessing import Pool
from tqdm import tqdm

from .utils import scandir, make_dir

def get_default_opt():
    opt = {}
    opt['n_thread'] = 20

    opt['input_folder'] = 'data/datasets/BSRN_train/car_train'
    opt['save_foder_input'] = 'data/datasets/BSRN_train/car_train_input'
    opt['save_foder_label'] = 'data/datasets/BSRN_train/car_train_label'

    opt['scale_factors'] = [4] # [2,3,4]          # scale factors
    opt['downsizes'] = [1] # [1,0.7,0.5]          # downsizing
    opt['flips'] = [0, 1, 2]                # 0: no; 1: 水平翻转; 2: 垂直翻转
    opt['rotate_degrees'] = [0, 1, 2, 3]    # rotate degree, 1=90 degree

    opt['size_input'] = 256 # 41
    opt['size_label'] = 256
    opt['stride'] = 256
    opt['thresh_size'] = 256 # 阈值size 比 thresh_size 小的补丁的会被舍弃

    return opt

def gen(opt):
    input_folder = opt['input_folder']  
    save_foder_input = opt['save_foder_input']
    save_foder_label = opt['save_foder_label']

    make_dir(save_foder_input, delete=True)
    make_dir(save_foder_label, delete=True)

    # generate data
    input_paths = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(input_paths), unit='image', desc='Process')
    pool = Pool(opt['n_thread'])
    for input_path in input_paths:
        pool.apply_async(worker, args=(input_path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(input_path, opt):
    save_foder_input = opt['save_foder_input']
    save_foder_label = opt['save_foder_label']

    size_input = opt['size_input']
    size_label = opt['size_label']
    stride = opt['stride']
    thresh_size = opt['thresh_size']

    scale_factors = opt['scale_factors']      
    downsizes = opt['downsizes']       
    flips = opt['flips']             
    rotate_degrees = opt['rotate_degrees'] 

    image_name, extension = osp.splitext(osp.basename(input_path))
    
    for flip in flips:
        for rotate_degree in rotate_degrees:
            for scale_factor in scale_factors:
                for downsize in downsizes:
                    tag = f'f{flip}_r{rotate_degree}_sc{scale_factor}_d{downsize}'
                    image = pil_image.open(input_path, mode='r') # size: (width, height)

                    # 变换
                    if flip == 1:
                        image = image.transpose(pil_image.FLIP_LEFT_RIGHT)
                    elif flip == 2:
                        image = image.transpose(pil_image.FLIP_TOP_BOTTOM)
                    image = image.rotate(90 * rotate_degree)
                    image = image.resize((int(image.width*downsize), int(image.height*downsize)), pil_image.BICUBIC)

                    # crop edge
                    small_width = int(image.width / scale_factor)
                    small_height = int(image.height / scale_factor)
                    cropped_width = small_width * scale_factor
                    cropped_height = small_height * scale_factor
                    if cropped_width!=image.width or cropped_height!=image.height:
                        crop_w_start = (image.width-cropped_width) / 2
                        crop_h_start = (image.height-cropped_height) / 2
                        image_label = image.crop((crop_w_start, crop_h_start, crop_w_start+cropped_width, crop_h_start+cropped_height))
                    else:
                        image_label = image
                    
                    # resize 生成低分辨率图片
                    image_small = image_label.resize((small_width, small_height), pil_image.BICUBIC)
                    image_input = image_small.resize((cropped_width, cropped_height), pil_image.BICUBIC)

                    # 转换为 numpy (H W C)
                    image_label = cv2.cvtColor(np.asarray(image_label), cv2.COLOR_RGB2BGR)
                    image_input = cv2.cvtColor(np.asarray(image_input), cv2.COLOR_RGB2BGR)

                    # 得到 sub image 的位置坐标
                    h_space = np.arange(0, cropped_height - size_input + 1, stride)
                    if cropped_height - (h_space[-1] + size_input) > thresh_size:
                        h_space = np.append(h_space, cropped_height - size_input)
                    w_space = np.arange(0, cropped_width - size_input + 1, stride)
                    if cropped_width - (w_space[-1] + size_input) > thresh_size:
                        w_space = np.append(w_space, cropped_width - size_input)
                    
                    # 提取 sub image
                    index = 0
                    for x in h_space:
                        for y in w_space:
                            sub_image_input = image_input[x : x+size_input-1, y : y+size_input-1]
                            sub_image_label = image_label[x : x+size_label-1, y : y+size_label-1]

                            # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
                            sub_image_input = np.ascontiguousarray(sub_image_input)
                            sub_image_label = np.ascontiguousarray(sub_image_label)
                            basename_input = f'{image_name}_{tag}_s{index:03d}'
                            basename_label = f'{image_name}_{tag}_s{index:03d}'
                            index = index + 1
                            sub_image_input_name = f'{basename_input}{extension}'
                            sub_image_label_name = f'{basename_label}{extension}'
                            sub_image_input_path = osp.join(save_foder_input, sub_image_input_name)
                            sub_image_label_path = osp.join(save_foder_label, sub_image_label_name)
                            cv2.imwrite(sub_image_input_path, sub_image_input)
                            cv2.imwrite(sub_image_label_path, sub_image_label)
    
    process_info = f'Processing {image_name}{extension} ...'
    return process_info
