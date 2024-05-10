import torch, gc
import PIL.Image as pil_image
import numpy as np
import os
from os import path as osp
import cv2
import math
# import logging

from basicsr.data import build_dataloader
from basicsr.data import build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # squeeze: 如果dim指定的维度的值为1，则将该维度删除，若指定的维度值不为1，则返回原来的tensor
        # clamp_: input：输入张量；min：限制范围下限；max：限制范围上限；out：输出张量。
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            return None
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def sr_img(img_input, arch, opt=None, model_path=None, save_path=None):
    with torch.no_grad():
        '''
        img_input: str or pil.image
        '''
        if opt == None:
            if arch == 'EDSR':
                opt = {
                    'name': 'EDSR',
                    'model_type': 'SRModel',
                    'scale': 4,
                    'num_gpu': 1,  # set num_gpu: 0 for cpu mode
                    'manual_seed': 0,
                    'network_g': {
                        'type': 'EDSR',
                        'num_in_ch': 3,
                        'num_out_ch': 3,
                        'num_feat': 256,
                        'num_block': 32,
                        'upscale': 4,
                        'res_scale': 0.1,
                        'img_range': 255.0,
                        'rgb_mean': [0.4488, 0.4371, 0.4040]
                    },
                    'path': {
                        'pretrain_network_g': 'experiments/pretrained_models/EDSR/net_mixture_200000.pth',
                        'strict_load_g': True
                    },
                    'dist': False,
                    'rank': 0,
                    'world_size': 1,
                    'is_train': False
                }
            elif arch == 'RCAN':
                opt = {
                    'name': 'RCAN',
                    'model_type': 'SRModel',
                    'scale': 4,
                    'num_gpu': 1,  # set num_gpu: 0 for cpu mode
                    'manual_seed': 0,
                    'network_g': {
                          'type': 'RCAN',
                          'num_in_ch': 3,
                          'num_out_ch': 3,
                          'num_feat': 64,
                          'num_group': 10,
                          'num_block': 20,
                          'squeeze_factor': 16,
                          'upscale': 4,
                          'res_scale': 1,
                          'img_range': 255.0,
                          'rgb_mean': [0.4488, 0.4371, 0.4040]
                    },
                    'path': {
                        'pretrain_network_g': 'experiments/pretrained_models/RCAN/net_mixture.pth',
                        'strict_load_g': True
                    },
                    'dist': False,
                    'rank': 0,
                    'world_size': 1,
                    'is_train': False
                }
            else:
                print('SR arch error')
                exit()
        if model_path:
            opt['path']['pretrain_network_g'] = model_path
        model = build_model(opt)

        if isinstance(img_input, str):
            img_input = pil_image.open(img_input)
        img_input = np.array(img_input.convert("RGB"))
        img_input = img_input.astype(np.float32) / 255  # change to float32 and norm to [0, 1]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_input = torch.from_numpy(img_input.transpose(2, 0, 1)).float()
        test_data = {
            'lq': img_input
        }
        model.feed_data(test_data)
        model.test()
        img_out_tensor = model.output.detach().cpu()
        del model.output
        del model.lq

        img_out_np = tensor2img(img_out_tensor, rgb2bgr=False)
        img_out = pil_image.fromarray(img_out_np)
        if save_path:
            img_out.save(save_path)

        gc.collect()
        torch.cuda.empty_cache()
        del model
        del img_input
        del test_data
        return img_out
    
def sr_img_dif_tag_bsr(tag, arch, img_input, opt=None, model_dir=None, save_path=None):
    if model_dir == None:
        model_dir = "model"
    if tag == 'origin':
        model_path = os.path.join(model_dir, "net_mixture.pth")
    elif tag == 'car':
        model_path = os.path.join(model_dir, "net_car.pth")
    elif tag == 'cat':
        model_path = os.path.join(model_dir, "net_cat.pth")
    elif tag == 'dog':
        model_path = os.path.join(model_dir, "net_dog.pth")
    elif tag == 'people':
        model_path = os.path.join(model_dir, "net_people.pth")
    elif tag == 'plane':
        model_path = os.path.join(model_dir, "net_plane.pth")

    img_output = sr_img(img_input, arch, opt=opt, model_path=model_path, save_path=save_path)
    return img_output
    
is_gpu = False
img_input_path = 'datasets/set01/lr/0.png'
model_path = 'experiments/pretrained_models/EDSR/net_mixture_200000.pth'

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print('===== START =====')# create model
    # opt, _ = parse_options(root_path, is_train=False)
    opt = {
        'name': 'EDSR_Lx4_f256b32_DIV2K_official',
        'model_type': 'SRModel',
        'scale': 4,
        'num_gpu': 0,  # set num_gpu: 0 for cpu mode
        'manual_seed': 0,
        'network_g': {
            'type': 'EDSR',
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 256,
            'num_block': 32,
            'upscale': 4,
            'res_scale': 0.1,
            'img_range': 255.0,
            'rgb_mean': [0.4488, 0.4371, 0.4040]
        },
        'path': {
            'pretrain_network_g': 'experiments/pretrained_models/my/net_mixture_200000.pth',
            'strict_load_g': True
        },
        'dist': False,
        'rank': 0,
        'world_size': 1,
        'is_train': False
    }
    
    sr_img(
        opt, 
        img_input_path='BasicSR_master/datasets/set01/lr/0.png', 
        model_path='BasicSR_master/experiments/pretrained_models/my/net_mixture_200000.pth', 
        save_path='testpic.png')


    print('=====  END  =====')