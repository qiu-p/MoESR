# Load the package you are going to use
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

from .tools.utils import scandir, make_dir
from .tools.utils import get_logger

# 添加 module 搜索路径
sys.path.insert(0, 'VDSR')

# Here is the function for PSNR calculation
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


# Define the colorization function
# We'll reuse the Cb and Cr channels from bicubic interpolation
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


def test1(model, img_label_path, img_input_path):
    # Load the groundtruth image and the low-resolution image (downscaled with a factor of 4)
    img_label = Image.open(img_label_path).convert("RGB")
    img_input = Image.open(img_input_path).convert("RGB")

    # Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
    img_label_ycbcr = np.array(img_label.convert("YCbCr"))
    img_input_ycbcr = np.array(img_input.convert("YCbCr"))
    img_label_y = img_label_ycbcr[:,:,0].astype(float)
    img_input_y = img_input_ycbcr[:,:,0].astype(float)
    # Calculate the PNSR for bicubic interpolation
    # For best PSNR score, you will have to use Matlab for color space transformation, 
    # since it is different from Python's implementation
    psnr_bicubic = PSNR(img_label_y, img_input_y)
    print('psnr for bicubic is {}dB'.format(psnr_bicubic))
    
    # Prepare for the input, a pytorch tensor
    im_input = img_input_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).\
        view(1, -1, im_input.shape[0], im_input.shape[1])
    # Let's try the network feedforward in cpu mode
    model = model.cpu()
    # Let's see how long does it take for processing
    start_time = time.time()
    out = model(im_input)
    elapsed_time = time.time() - start_time
    print("It takes {}s for processing in cpu mode".format(elapsed_time))
    # Get the output image
    out = out.cpu()
    im_h_y = out.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h_y = im_h_y[0,:,:]
    # Calculate the PNSR for vdsr prediction
    psnr_predicted = PSNR(img_label_y, im_h_y)
    print('psnr for vdsr is {}dB'.format(psnr_predicted))
    # Calculate the PNSR different between bicubic interpolation and vdsr prediction
    print("PSNR improvement is {}dB".format(psnr_predicted - psnr_bicubic))

    # Colorize the grey-level image and convert into RGB mode
    img_output = colorize(im_h_y, img_input_ycbcr)
    img_label = Image.fromarray(img_label_ycbcr, "YCbCr").convert("RGB")
    img_input = Image.fromarray(img_input_ycbcr, "YCbCr").convert("RGB")


def test_img(model, img_input_path, img_label_path):
    # Load the groundtruth image and the low-resolution image (downscaled with a factor of 4)
    img_label = Image.open(img_label_path).convert("RGB")
    img_input = Image.open(img_input_path).convert("RGB")

    # Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
    img_label_ycbcr = np.array(img_label.convert("YCbCr"))
    img_input_ycbcr = np.array(img_input.convert("YCbCr"))
    img_label_y = img_label_ycbcr[:,:,0].astype(float)
    img_input_y = img_input_ycbcr[:,:,0].astype(float)
    # Calculate the PNSR for bicubic interpolation
    # For best PSNR score, you will have to use Matlab for color space transformation, 
    # since it is different from Python's implementation
    psnr_bicubic = PSNR(img_label_y, img_input_y)
    
    # Prepare for the input, a pytorch tensor
    im_input = img_input_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).\
        view(1, -1, im_input.shape[0], im_input.shape[1])
    # Let's try the network feedforward in cpu mode
    model = model.cpu()
    out = model(im_input)
    # Get the output image
    out = out.cpu()
    im_h_y = out.data[0].numpy().astype(np.float32)
    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    im_h_y = im_h_y[0,:,:]
    # Calculate the PNSR for vdsr prediction
    psnr_predicted = PSNR(img_label_y, im_h_y)

    # Colorize the grey-level image and convert into RGB mode
    img_output = colorize(im_h_y, img_input_ycbcr)

    return psnr_bicubic, psnr_predicted, img_output


def sr_img(model, img_input, is_gpu=False):
    with torch.no_grad():
        '''
        img_input: Image.PIL or path
        '''
        # Load the groundtruth image and the low-resolution image (downscaled with a factor of 4)
        if isinstance(img_input, str):
            img_input = Image.open(img_input)
        img_input = img_input.convert("RGB")

        # Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
        img_input_ycbcr = np.array(img_input.convert("YCbCr"))
        img_input_y = img_input_ycbcr[:,:,0].astype(float)

        # Prepare for the input, a pytorch tensor
        im_input = img_input_y/255.
        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
        # Let's try the network feedforward
        if is_gpu:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()
        out = model(im_input)
        # Get the output image
        out = out.cpu()
        im_h_y = out.data[0].numpy().astype(np.float32)
        im_h_y = im_h_y * 255.
        im_h_y[im_h_y < 0] = 0
        im_h_y[im_h_y > 255.] = 255.
        im_h_y = im_h_y[0,:,:]

        # Colorize the grey-level image and convert into RGB mode
        img_output = colorize(im_h_y, img_input_ycbcr)

        return img_output


def test_imgs(model, img_input_dir, img_label_dir, save_dir=None):
    paths = list(scandir(img_input_dir, full_path=False))
    # print(paths)

    psnr_bicubic_total = 0
    psnr_predicted_total = 0
    pbar = tqdm(total=len(paths), unit='image', desc='Process')
    for path in paths:
        psnr_bicubic, psnr_predicted, img_output = test_img(model, os.path.join(img_input_dir, path), os.path.join(img_label_dir, path))
        psnr_predicted_total += psnr_predicted
        psnr_bicubic_total += psnr_bicubic
        if save_dir!=None:
            img_output.save(os.path.join(save_dir, path))
        pbar.update(1)
    pbar.close()
    
    return psnr_bicubic_total/5.0, psnr_predicted_total/5.0


def sr_imgs(model, img_input_dir, save_dir=None):
    if save_dir!=None:
        make_dir(save_dir, delete=True)
    paths = list(scandir(img_input_dir, full_path=False))
    # print(paths)

    psnr_bicubic_total = 0
    psnr_predicted_total = 0
    pbar = tqdm(total=len(paths), unit='image', desc='Process')
    for path in paths:
        img_input = Image.open(os.path.join(img_input_dir, path))
        img_output = sr_img(model, img_input)
        if save_dir!=None:
            img_output.save(os.path.join(save_dir, path))
        pbar.update(1)
    pbar.close()


def test_mod1():
    tag = 'mixure'
    logfile = './log/test/test_' + tag + '.log'
    logger = get_logger(log_file=logfile)

    model_path = "model/" + tag + "/model_" + tag + ".pth"
    model = torch.load(model_path, map_location=torch.device('cpu'))["model"]
    logger.info('model: {}'.format(tag))

    img_label_dir = "data/datasets/test/car_test_cropEdge"
    img_input_dir = "data/datasets/test/car_test_cropEdge_rrx4_irx4"
    psnr_bicubic, psnr_predicted = test_imgs(model, img_input_dir, img_label_dir)
    logger.info('test data: {}'.format('car'))
    logger.info('psnr for bicubic is {} dB'.format(psnr_bicubic))
    logger.info('psnr for vdsr is {} dB'.format(psnr_predicted))

    img_label_dir = "data/datasets/test/cat_test_cropEdge"
    img_input_dir = "data/datasets/test/cat_test_cropEdge_rrx4_irx4"
    psnr_bicubic, psnr_predicted = test_imgs(model, img_input_dir, img_label_dir)
    logger.info('test data: {}'.format('cat'))
    logger.info('psnr for bicubic is {} dB'.format(psnr_bicubic))
    logger.info('psnr for vdsr is {} dB'.format(psnr_predicted))

    img_label_dir = "data/datasets/test/dog_test_cropEdge"
    img_input_dir = "data/datasets/test/dog_test_cropEdge_rrx4_irx4"
    psnr_bicubic, psnr_predicted = test_imgs(model, img_input_dir, img_label_dir)
    logger.info('test data: {}'.format('dog'))
    logger.info('psnr for bicubic is {} dB'.format(psnr_bicubic))
    logger.info('psnr for vdsr is {} dB'.format(psnr_predicted))

    img_label_dir = "data/datasets/test/people_test_cropEdge"
    img_input_dir = "data/datasets/test/people_test_cropEdge_rrx4_irx4"
    psnr_bicubic, psnr_predicted = test_imgs(model, img_input_dir, img_label_dir)
    logger.info('test data: {}'.format('people'))
    logger.info('psnr for bicubic is {} dB'.format(psnr_bicubic))
    logger.info('psnr for vdsr is {} dB'.format(psnr_predicted))

    img_label_dir = "data/datasets/test/plane_test_cropEdge"
    img_input_dir = "data/datasets/test/plane_test_cropEdge_rrx4_irx4"
    psnr_bicubic, psnr_predicted = test_imgs(model, img_input_dir, img_label_dir)
    logger.info('test data: {}'.format('plane'))
    logger.info('psnr for bicubic is {} dB'.format(psnr_bicubic))
    logger.info('psnr for vdsr is {} dB'.format(psnr_predicted))

    
def test_mod2():
    def _test(model_path, img_input_dir, save_dir, tag=None):
        if tag:
            print('test {}:'.format(tag))
        model = torch.load(model_path, map_location=torch.device('cpu'))["model"]
        sr_imgs(model, img_input_dir, save_dir=save_dir)

    _test(
        model_path="model\dog\model_dog.pth",
        img_input_dir="data\datasets\seg_test\set01\dog_masked",
        save_dir="data\datasets\seg_test\set01\dog_masked_sr"
    )
    _test(
        model_path="model\people\model_people.pth",
        img_input_dir="data\datasets\seg_test\set01\people_masked",
        save_dir="data\datasets\seg_test\set01\people_masked_sr"
    )
    _test(
        model_path="model\mixure\model_mixure.pth",
        img_input_dir="data\datasets\seg_test\set01\origin_cropEdge_rrx4_irx4",
        save_dir="data\datasets\seg_test\set01\origin_cropEdge_rrx4_irx4_sr",
        tag='origin'
    )


def sr_img_dif_tag(tag, img_input, model_dir=None, is_gpu=False):
    if model_dir == None:
        model_dir = "model"
    if tag == 'origin':
        model_path = os.path.join(model_dir, "model_mixture.pth")
    elif tag == 'car':
        model_path = os.path.join(model_dir, "model_car.pth")
    elif tag == 'cat':
        model_path = os.path.join(model_dir, "model_cat.pth")
    elif tag == 'dog':
        model_path = os.path.join(model_dir, "model_dog.pth")
    elif tag == 'people':
        model_path = os.path.join(model_dir, "model_people.pth")
    elif tag == 'plane':
        model_path = os.path.join(model_dir, "model_plane.pth")

    model = torch.load(model_path, map_location=torch.device('cpu'))["model"]
    img_output = sr_img(model, img_input, is_gpu)
    return img_output



def sr_img_dif_tags(tags, img_input, model_dir=None, is_gpu=False):
    if model_dir == None:
        model_dir = "model"
    
    img_outputs = {}
    for tag in tags:
        img_outputs[tag] = sr_img_dif_tag(tag, img_input, model_dir=model_dir, is_gpu=is_gpu)

    return img_outputs


if __name__ == '__main__':
    test_list = []
    test_mod2()

    