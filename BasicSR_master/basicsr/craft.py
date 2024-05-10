import torch
import PIL.Image as pil_image
import numpy as np
from os import path as osp
# import logging

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

is_gpu = False
img_input_path = '../datasets/set01/lr/0.png'
model_path = '../experiments/pretrained_models/my/net_mixture_200000.pth'

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print('===== START =====')# create model
    opt, _ = parse_options(root_path, is_train=False)

    model = build_model(opt)

    img_input = np.array(pil_image.open(img_input_path).convert("RGB"))
    img_input = img_input.astype(np.float32) / 255  # change to float32 and norm to [0, 1]
    # BGR to RGB, HWC to CHW, numpy to tensor
    img_input = torch.from_numpy(img_input.transpose(2, 0, 1)).float()
    test_data = {
        'lq': img_input
    }
    model.feed_data(test_data)
    model.test()
    out = model.output.detach().cpu()
    print(out)
    exit()

    model = torch.load(model_path, map_location=torch.device('cpu'))
    data = {
        'lq': '',
        'gt': ''
    }
    #out = model()
    print(model)

    exit()
    with torch.no_grad():
        '''
        img_input: Image.PIL or path
        '''
        # Load the groundtruth image and the low-resolution image (downscaled with a factor of 4)
        if isinstance(img_input, str):
            img_input = pil_image.open(img_input)
        img_input = img_input.convert("RGB")

        # Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
        img_input_ycbcr = np.array(img_input.convert("YCbCr"))
        img_input_y = img_input_ycbcr[:,:,0].astype(float)

        # Prepare for the input, a pytorch tensor
        im_input = img_input_y/255.
        im_input = torch.from_numpy(im_input).float().view(1, -1, im_input.shape[0], im_input.shape[1])
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


    print('=====  END  =====')