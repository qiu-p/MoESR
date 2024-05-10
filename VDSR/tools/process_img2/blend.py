import cv2
import numpy as np
import os

from ..utils import make_dir

# black: 0;  white: 255
# white & white = white
# white & black = black

def blend_img(obj_mask, obj_img, obj_mask_smaller, background_img, erode_kernel_size = 10):
    '''
    obj_mask: the mask of the img
    obj_img: the img to be blended
    obj_mask_smaller: the mask of the img(smaller)
    background_img: the img that is the background of the blend
    output_path: save path
    erode_kernel_size: 若为0 则不腐蚀
    '''
        
    if not isinstance(obj_img, np.ndarray):
        obj_img = cv2.cvtColor(np.asarray(obj_img), cv2.COLOR_RGB2BGR)
    if not isinstance(background_img, np.ndarray):
        background_img = cv2.cvtColor(np.asarray(background_img), cv2.COLOR_RGB2BGR)
        
    if isinstance(obj_mask, list):
        print('obj_mask: {}'.format(obj_mask))
        mask_img = np.zeros((background_img.shape[0], background_img.shape[1]))
        for box in obj_mask:
            mask_img[box[1]:box[3], box[0]:box[2]] = 255
        obj_mask = mask_img
        obj_mask = obj_mask.astype(np.uint8)
    elif not isinstance(obj_mask, np.ndarray):
        obj_mask = cv2.cvtColor(np.asarray(obj_mask), cv2.COLOR_RGB2BGR)
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_BGR2GRAY)
    else:
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_BGR2GRAY)
    if isinstance(obj_mask_smaller, list):
        print('obj_mask_smaller: {}'.format(obj_mask_smaller))
        mask_img = np.zeros((background_img.shape[0], background_img.shape[1]))
        for box in obj_mask_smaller:
            mask_img[box[1]:box[3], box[0]:box[2]] = 255
        obj_mask_smaller = mask_img
        obj_mask_smaller = obj_mask_smaller.astype(np.uint8)
    elif not isinstance(obj_mask_smaller, np.ndarray):
        obj_mask_smaller = cv2.cvtColor(np.asarray(obj_mask_smaller), cv2.COLOR_RGB2BGR)
        obj_mask_smaller = cv2.cvtColor(obj_mask_smaller, cv2.COLOR_BGR2GRAY)
    else:
        obj_mask_smaller = cv2.cvtColor(obj_mask_smaller, cv2.COLOR_BGR2GRAY)
        
    
    if erode_kernel_size>0:
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        obj_mask = cv2.erode(obj_mask, kernel, iterations=1)
        obj_mask_smaller = cv2.erode(obj_mask_smaller, kernel, iterations=1)
    
    obj_mask_smaller_reverse = ~obj_mask_smaller
    mask2 = obj_mask & obj_mask_smaller_reverse # mask2 为一个白圈
    
    # 膨胀轮廓，使轮廓向内收缩5个像素点
    #kernel = np.ones((18, 15), np.uint8)
    #mask = cv2.erode(mask, kernel, iterations=1)
    # 将非黑色部分从第一张图复制到第二张图，处理mask1的图像
    background_img[obj_mask_smaller > 0] = obj_img[obj_mask_smaller > 0]
    
    # 处理mask2的图像
    # 通过alpha blending算法融合图像,首先计算~mask2到mask1和obj_mask
    # 计算每个像素到离它最近的零值像素（黑色像素）的距离
    alpha1 = cv2.distanceTransform(obj_mask, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5, dstType=cv2.CV_32F)
    alpha2 = cv2.distanceTransform(obj_mask_smaller_reverse, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5, dstType=cv2.CV_32F)
    total = alpha1 + alpha2
    alpha = alpha1
    alpha[total > 0] = alpha1[total > 0] / (total[total > 0])
    alpha = alpha.astype(float)
    alpha = alpha[:, :, np.newaxis]
    alpha = np.repeat(alpha, 3, axis=2)

    blended = obj_img * alpha  + background_img * (1- alpha)
    blended = blended.astype(np.uint8)
    
    background_img[mask2 > 0] = blended[mask2 > 0]
    # background_img[mask2 > 0] = 0
    
    return background_img
    

def blend():
    output_dir = 'data/datasets/seg_test/set01/blend'
    output_name = 'blend.png'
    make_dir(output_dir, rename=True)

    print("===== blend... =====")
    mask_path = "data/datasets/seg_test/set01/people_mask/mask_people_d.jpg"
    img_path = "data/datasets/seg_test/set01/people_masked_sr/img01.png"
    obj_mask_smaller_path = "data/datasets/seg_test/set01/people_mask/mask_people.jpg"
    background_path = "data/datasets/seg_test/set01/origin_cropEdge_rrx4_irx4_sr/img01.png"
    mask = cv2.imread(mask_path)
    img = cv2.imread(img_path)
    obj_mask_smaller = cv2.imread(obj_mask_smaller_path)
    background_img = cv2.imread(background_path)
    out_img = blend_img(mask, img, obj_mask_smaller, background_img)
    
    print("===== blend... =====")
    mask_path = "data/datasets/seg_test/set01/dog_mask/mask_dog_d.jpg"
    img_path = "data/datasets/seg_test/set01/dog_masked_sr/img01.png"
    obj_mask_smaller_path = "data/datasets/seg_test/set01/dog_mask/mask_dog.jpg"
    mask = cv2.imread(mask_path)
    img = cv2.imread(img_path)
    obj_mask_smaller = cv2.imread(obj_mask_smaller_path)
    background_img = cv2.imread(background_path)
    out_img = blend_img(mask, img, obj_mask_smaller, out_img)

    print("===== save... =====")
    cv2.imwrite(os.path.join(output_dir, output_name), out_img)