import argparse
from PIL import Image
import os.path as osp
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from tqdm import tqdm
import importlib

from ..utils import scandir, make_dir
from ..utils import dict2str, parse_options, parse_yaml

operation_module = importlib.import_module(f'..operations', package=sys.modules[__name__].__name__)

class OperationList():
    def __init__(self, opt, operation_module, work_dir=None):
        if work_dir:
            self.work_dir = work_dir
        else:
            self.work_dir = opt['work_dir']
        opt['input_dir'] = osp.join(self.work_dir, opt['input_dir'])
        self.existed_dir = opt['existed_dir']
        if 'save_dir_prefix' not in opt:
            self.save_dir_prefix = opt['input_dir']
        else:
            opt['save_dir_prefix'] = osp.join(self.work_dir, opt['save_dir_prefix'])
            self.save_dir_prefix = opt['save_dir_prefix']
        self.operation_opt_list = opt['operation_list']

        self.operation_list = []
        self.operation_module = operation_module

        # get operations list
        self._get_oprtation_list()

    def _get_oprtation_list(self):
        for operation_tag, operation_opt in self.operation_opt_list.items():
            # decide how to deal with existed dir
            if 'existed_dir' not in operation_opt:
                operation_opt['existed_dir'] = self.existed_dir
            if operation_opt.get('work_dir') == None:
                operation_opt['work_dir'] = self.work_dir
            
            operation = getattr(self.operation_module, operation_opt['mode'])(operation_opt)
            # get the path of save dir
            self.save_dir_prefix = operation.make_save_dirs(self.save_dir_prefix)
            # add operation
            self.operation_list.append(operation)

    def operate(self, path, img=None):
        for operation in self.operation_list:
            path, imgs = operation.operate(path, img)
            if isinstance(imgs, list):
                img = imgs[0]
            else:
                img = imgs
        return imgs


def _get_process_list(opt, yaml_dir=None, imgs_list=None, work_dir=None):
    process_lists = []         # (process_key, process_opt['input_dir'], process_opt['input_dir_recursive'])
    operation_list_dict = {}   # process_key: operations
    for process_key, yaml_path in opt['process_list'].items():
        if yaml_dir:
            yaml_path = osp.join(yaml_dir, osp.basename(yaml_path))
        # get processes that imgs_list need
        if imgs_list!=None:
            if not process_key in imgs_list:
                continue

        process_opt = parse_yaml(yaml_path)
        # get operation_list
        operation_list_dict[process_key] = OperationList(process_opt, operation_module, work_dir)
        process_lists.append((process_key, process_opt['input_dir'], process_opt['input_dir_recursive'], process_opt['img_from_outer']))
    return process_lists, operation_list_dict


def process_imgs(opt, imgs_list=None, work_dir=None):
    ''' 
    imgs_list = {
        process_key1: [img1, img2, ...],
        process_key2: [[img1, mask_json_file_contents], ...],
        ...
        'dog': [(img, mask)],
        'people': [[img, mask2]],
        ...
    }
    '''
    process_lists, operation_list_dict = _get_process_list(
        opt = opt[0], 
        yaml_dir = opt[1], 
        imgs_list=imgs_list, 
        work_dir=work_dir)

    # processed_imgs={
    # process_key1: [imgs1, imgs2, ...],
    # ...}
    processed_imgs = {}
    for process_key, input_dir, input_dir_recursive, img_from_outer in process_lists:        
        print('process list: {} \t img_from_outer: {}'.format(process_key, img_from_outer))
        processed_imgs[process_key] = []
        if img_from_outer == True:
            img_list = []
            for i, img in enumerate(imgs_list[process_key]):
                img_path = 'img_{}_{}.png'.format(process_key, i)
                if (isinstance(img, list) and isinstance(img[0], str)):
                    img_path = img[0]
                    img[0] = Image.open(img_path).convert('RGB')
                elif isinstance(img, str):
                    img_path = img
                    img = Image.open(img_path).convert('RGB')
                img_list.append((img_path, img))
            pbar = tqdm(total=len(img_list), unit='image', desc='Process')
            # pool = Pool(opt['n_thread'])
            for path, img in img_list:
                # pool.apply_async(worker, args=(path, operation_list_dict[process_key]), callback=lambda arg: pbar.update(1))
                process_info, img = worker(operation_list_dict[process_key], path, img)
                processed_imgs[process_key].append(img)
                pbar.update(1)
            # pool.close()
            # pool.join()
            pbar.close()
        else:
            # 绝对路径
            img_list = list(scandir(
                dir_path=input_dir, 
                full_path=True, 
                recursive=input_dir_recursive)) # 完整路径
            pbar = tqdm(total=len(img_list), unit='image', desc='Process')
            # pool = Pool(opt['n_thread'])
            for path in img_list:
                # pool.apply_async(worker, args=(path, operation_list_dict[process_key]), callback=lambda arg: pbar.update(1))
                process_info, img = worker(operation_list_dict[process_key], path)
                processed_imgs[process_key].append(img)
                pbar.update(1)
            # pool.close()
            # pool.join()
            pbar.close()

    print('All processes done.')
    return processed_imgs


def worker(operation_list, path, img=None):
    basename, ext = osp.splitext(osp.basename(path))

    img = operation_list.operate(path, img)

    process_info = f'Processing {basename}{ext} ...'
    return process_info, img

