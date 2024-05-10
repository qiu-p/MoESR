import cv2
import numpy as np
import os 

# 创造一张分割下来图片的蒙版
def mask_generation(img, height, width):
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    pic_isr = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask[img[1][1]:img[1][3], img[1][0]:img[1][2]] = 255
    pic_isr[img[1][1]:img[1][3], img[1][0]:img[1][2]] = img[0]
    
    return pic_isr, mask

# 创造一张背景蒙版
def mask_generation_base(img_to_blend, height, width,improve_scale,borderV):
    mask = np.ones((height, width, 1), dtype=np.uint8) * 255
    i = 0
    for box in img_to_blend:
        if i  > 0 :
            x_min = 0 if (box[1][0] == 0) else (box[1][0] + borderV*improve_scale)
            y_min = 0 if(box[1][1] == 0) else(box[1][1] + borderV*improve_scale)
            x_max = width if(box[1][2] == width) else(box[1][2] - borderV*improve_scale)
            y_max = height if(box[1][3] == height) else(box[1][3] - borderV*improve_scale)
            if x_max > x_min and y_max > y_min:
                mask[y_min:y_max, x_min:x_max] = 0
        i = i + 1   
    return mask


# 图像融合
def blend_pic(base_isr, base_mask, pic_isr, pic_mask,improve_scale,borderV):
    # 图像拼接后的蒙版
    base_mask_next = base_mask
    base_mask_next[pic_mask > 0] = 255
    
    base_mask = ~base_mask
    kernel = np.ones((borderV*improve_scale, borderV*improve_scale), np.uint8)
    pic_mask = cv2.erode(pic_mask, kernel, iterations=1)
    base_mask = cv2.erode(base_mask, kernel, iterations=0)
    mask1 = pic_mask & base_mask
    mask2 = pic_mask & (~base_mask)
    
    base_isr[mask1 > 0] = pic_isr[mask1 > 0]

    
    # base_isr[mask2 > 0] = pic_isr[mask2 > 0]
    '''
    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2', cv2.WINDOW_NORMAL)
    cv2.imshow('1', mask1)
    cv2.imshow('2', mask2)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    '''
    # 处理mask2的图像
    # 通过alpha blending算法融合图像,首先计算~mask2到mask1和mask_obj
    # 计算每个像素到离它最近的零值像素（黑色像素）的距离
    
    
    alpha1 = cv2.distanceTransform(pic_mask, cv2.DIST_L2, 5)
    alpha2 = cv2.distanceTransform(base_mask, cv2.DIST_L2, 5)
    alpha1 = alpha1.astype(float)
    alpha2 = alpha2.astype(float)
    total = alpha1 + alpha2
    alpha = alpha1
    alpha[total > 0] = alpha1[total > 0] / (total[total > 0])
    alpha = alpha.astype(float)
    alpha = alpha[:, :, np.newaxis]
    alpha = np.repeat(alpha, 3, axis=2)
    
    blended = pic_isr * alpha  + base_isr * (1- alpha)
    
    base_isr[mask2 > 0] = blended[mask2 > 0]
    
    
    # 保存结果
    return base_mask_next, base_isr

def blend_img(img_to_blend,improve_scale,borderV):
    i = 0
    for img in img_to_blend:
        if i == 0:
            # 第一张图片要是整图（即用作背景的图）
            # print(file_name)
            height, width = img.shape[:2]
            base_mask = mask_generation_base(img_to_blend, height, width,improve_scale,borderV)
            base_isr = img
        else:
            pic_isr, pic_mask = mask_generation(img, height, width)
            base_mask, base_isr = blend_pic(base_isr, base_mask, pic_isr, pic_mask,improve_scale,borderV)
        i = i + 1
    return base_isr
    