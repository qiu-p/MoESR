
## operation 选项
**依据mask裁剪**
```yml
mode: mask
filename_tmpl: '{}'
mask_name_tmpl: ~
# mask 的文件名 （优先依据 mask_name_tmpl ）
mask_name: 'img1_people_mask.jpg'
# 处理后的 mask 的文件名
mask_save_name: 'mask2_people.jpg'
# 得到的 masked 的 img 的存放地址
save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\BSRN-main\datasets\20231214_2\people_masked
# mask 的存放地址
mask_foder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\BSRN-main\datasets\20231214_2\mask
# 处理后的 mask 的存放地址
mask_save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\BSRN-main\datasets\20231214_2\people_mask2 
# 蒙版腐蚀蒙版膨胀的 kernel 的大小
structure_element_kernel_size: 30
```

**裁剪边使图片大小符合 x 的倍数**
mode: crop_edge
filename_tmpl: '{}'
save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_cropEdge
base_size: 4
center: True

**降低分辨率**
mode: reduce_resolution
filename_tmpl: '{}'
save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_cropEdge_rrx4
reduce_scaling_factor: 4
rr_crop: True

**提高分辨率**
mode: improve_resolution
filename_tmpl: '{}'
save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_irx4
improve_scaling_factor: 4

**提取子图片**
mode: extract_subimages
filename_tmpl: '{}'
save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_sub
crop_size: 240 # the size of sub images: crop_size*crop_size
step: 240 # the step to crop the image
thresh_size: 240 # thresh size， the patches smaller than thresh size will be dropped
compression_level: 3 # (int): for cv2.IMWRITE_PNG_COMPRESSION.  The compression ratio of the picture


## others

```
  operation_1:
    mode: crop_edge
    filename_tmpl: '{}'
    save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_cropEdge
    base_size: 4
    center: True
  operation_2:
    mode: reduce_resolution
    filename_tmpl: '{}'
    save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_cropEdge_rrx4
    reduce_scaling_factor: 4
    rr_crop: True
  operation_3:
    mode: improve_resolution
    filename_tmpl: '{}'
    save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_irx4
    improve_scaling_factor: 4
  operation_4:
    mode: extract_subimages
    filename_tmpl: '{}'
    save_folder: E:\Jupyter\OpenCV\20231002_1008\my_train_BSRN\my_tools\test\bird_sub
    crop_size: 240
    step: 240
    thresh_size: 240
    compression_level: 3
```