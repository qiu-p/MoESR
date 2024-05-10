from PIL import Image
import numpy as np


def psnr(img1, img2, auto_resize=False):
    # 读取原始图像和压缩后的图像
    # compressed_image= Image.open("D:/ISR_test/out1/out3.png")
    if isinstance(img1, str):
        original_image = Image.open(img1)
    elif isinstance(img1, np.ndarray):
        original_image = Image.fromarray(img1.astype(np.uint8), mode='RGB')
    else:
        original_image = img1
    if isinstance(img2, str):
        compressed_image = Image.open(img2)
    elif isinstance(img2, np.ndarray):
        compressed_image = Image.fromarray(img2.astype(np.uint8), mode='RGB')
    else:
        compressed_image = img2
    # compressed_image = Image.open("E:/work/DaChuang/final/final1/outputs/out3.png")

    # 确保两幅图像具有相同的大小
    if original_image.size != compressed_image.size:
        if auto_resize:
            original_image = original_image.resize(compressed_image.size)
        else:
            assert original_image.size == compressed_image.size, "size not match"

    # 将图像转换为NumPy数组以进行计算
    original_data = np.array(original_image)
    compressed_data = np.array(compressed_image)

    # 计算均方误差（Mean Squared Error，MSE）
    mse = np.mean((original_data - compressed_data) ** 2)

    # 如果 MSE 为 0，表示两幅图像完全相同，PSNR 定义为无穷大
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel_value = 255.0  # 像素值的最大值
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    
    return psnr



