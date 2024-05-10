import PIL.Image as pil_image
import os.path as osp
import numpy as np

import cv2 

from .utils import scandir

def get_default_opt():
    opt = {}
    opt['input_folder'] = 'E:/work/DaChuang/final/misc/AA_test'
    opt['save_path'] = 'train.h5'
    opt['scale_factors'] = [2,3,4]          # scale factors
    opt['downsizes'] = [1,0.7,0.5]          # downsizing
    opt['flips'] = [0, 1, 2]                # 0: no; 1: 水平翻转; 2: 垂直翻转
    opt['rotate_degrees'] = [0, 1, 2, 3]    # rotate degree, 1=90 degree

    return opt

def gen(opt):
    input_folder = opt['input_folder']
    save_path = opt['save_path']

    size_input = 500 # 41
    size_label = 500
    stride = 500
    thresh_size = size_input # 阈值size， 比 thresh_size 小的补丁的会被舍弃

    scale_factors = opt['scale_factors']      
    downsizes = opt['downsizes']       
    flips = opt['flips']             
    rotate_degrees = opt['rotate_degrees']   

    # initialization
    data = np.zeros((1, size_input, size_input, 1))
    label = np.zeros((1, size_label, size_label, 1))

    count = 0
    margain = 0

    # generate data
    input_paths = scandir(input_folder, full_path=True)

    for input_path in input_paths:
        for flip in flips:
            for rotate_degree in rotate_degrees:
                for scale_factor in scale_factors:
                    for downsize in downsizes:
                        image = pil_image.open(input_path, mode='r') # size: (width, height)

                        # 变换
                        if flip == 1:
                            image = image.transpose(pil_image.FLIP_LEFT_RIGHT)
                        elif flip == 2:
                            image = image.transpose(pil_image.FLIP_TOP_BOTTOM)
                        image = image.rotate(90 * rotate_degree)
                        image = image.resize((image.width*downsize, image.height*downsize), pil_image.BICUBIC)

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
                        print(input_path + " OK")

                        # 转换为 numpy (H W C)
                        image_label = np.array(image_label.convert("YCbCr"))[:,:,0].astype(float)
                        image_input = np.array(image_input.convert("YCbCr"))[:,:,0].astype(float)
                        # 归一化
                        image_label = image_label/255
                        image_input = image_input/255

                        # 得到 sub image 的位置坐标
                        h_space = np.arange(0, cropped_height - size_input + 1, stride)
                        if cropped_height - (h_space[-1] + size_input) > thresh_size:
                            h_space = np.append(h_space, h - size_input)
                        w_space = np.arange(0, cropped_width - size_input + 1, stride)
                        if cropped_width - (w_space[-1] + size_input) > thresh_size:
                            w_space = np.append(w_space, w - size_input)
                        
                        # 提取 sub image
                        for x in h_space:
                            for y in w_space:
                                subimage_input = image_input[x : x+size_input-1, y : y+size_input-1]
                                subimage_label = image_label[x : x+size_label-1, y : y+size_label-1]
                                
                                count = count + 1

                                data[count, :, :, 0] = subimage_input
                                label[count, :, :, 0] = subimage_label
    
    order = np.random.permutation(count)
    data = data[order, :, :, 0]     # (N H W C)
    label = label[order, :, :, 0]

    data = np.transpose(data, (0, 3, 1, 2)) # (N C H W)
    label = np.transpose(label, (0, 3, 1, 2)) # (N C H W)

    # writing to HDF5
    # chunksz = 64
    # created_flag = false
    # totalct = 0

    # for batchno in range(int(count/chunksz)):
    #     print('batchno:', batchno)
    #     last_read = batchno * chunksz
    #     batchdata = data[:, :, 0, last_read:last_read+chunksz]
    #     batchlabs = label[:, :, 0, last_read:last_read+chunksz]

    #     startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1])
    #     curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz)
    #     created_flag = true
    #     totalct = curr_dat_sz(end)

