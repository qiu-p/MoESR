import argparse
from PIL import Image
import os.path as osp
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from tqdm import tqdm

from .operations import get_operation
from ..utils import scandir, make_dir
from ..utils import dict2str, parse_options, parse_yaml

def _make_save_dirs(save_folder_key, save_folder_tmpl_key, operation_opt, save_folder_prefix):
    # get the path of save folder
    if (save_folder_key not in operation_opt) or (operation_opt.get(save_folder_key)==None):
        save_folder_tmpl = operation_opt.get(save_folder_tmpl_key)
        if save_folder_tmpl == None:
            save_folder_tmpl = '{}_' + operation_opt['mode']
        operation_opt[save_folder_key] = save_folder_tmpl.format(save_folder_prefix)
    save_folder_prefix = operation_opt[save_folder_key]
    
    save_folder = operation_opt[save_folder_key]

    def _make_dir(save_folder, existed_folder='raise'):
        if existed_folder == 'raise':
            print(f'mkdir {save_folder} ...')
            make_dir(save_folder)
        elif existed_folder == 'rename':
            print(f'mkdir {save_folder} ...')
            if make_dir(save_folder, rename=True):
                print(f'rename origin {save_folder} ...')
        elif existed_folder == 'delete':
            print(f'mkdir {save_folder} ...')
            if make_dir(save_folder, delete=True):
                print(f'delete origin {save_folder} ...')

    existed_folder = operation_opt['existed_folder']
    _make_dir(save_folder, existed_folder)

    return save_folder_prefix

def _get_process_list(opt):
    process_list = []
    operations_list_dict = {}
    for process_key, yaml_path in opt['process_list'].items():
        process_opt = parse_yaml(yaml_path)
        # get some options
        input_folder = process_opt['input_folder']
        recursive = process_opt['input_folder_recursive']
        if 'save_folder_prefix' not in process_opt:
            process_opt['save_folder_prefix'] = process_opt['input_folder']
        # get operations list
        operations_list = []
        save_folder_prefix = process_opt['save_folder_prefix']
        for operation_tag, operation_opt in process_opt['operations_list'].items():
            # decide how to deal with existed folder
            if 'existed_folder' not in operation_opt:
                operation_opt['existed_folder'] = process_opt['existed_folder']
            # get the path of save folder
            if (operation_opt.get('mask_save_folder')!=None) or (operation_opt.get('mask_save_folder_tmpl')!=None):
                _make_save_dirs(
                    save_folder_key = 'mask_save_folder',
                    save_folder_tmpl_key = 'mask_save_folder_tmpl',
                    operation_opt = operation_opt,
                    save_folder_prefix = save_folder_prefix)
            save_folder_prefix = _make_save_dirs(
                save_folder_key = 'save_folder',
                save_folder_tmpl_key = 'save_folder_tmpl',
                operation_opt = operation_opt,
                save_folder_prefix = save_folder_prefix)
            # add operation
            operations_list.append((get_operation(operation_opt), operation_opt))
        operations_list_dict[process_key] = operations_list
        # 绝对路径
        img_list = list(scandir(input_folder, full_path=True, recursive=recursive)) # 完整路径
        for img_path in img_list:
            process_list.append((img_path, process_key))
    return process_list, operations_list_dict


def process_imgs(opt):
    process_list, operations_list_dict = _get_process_list(opt)
    
    pbar = tqdm(total=len(process_list), unit='image', desc='Process')
    pool = Pool(opt['n_thread'])
    for path, process_key in process_list:
        pool.apply_async(worker, args=(path, operations_list_dict[process_key]), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, operations_list):
    basename, ext = osp.splitext(osp.basename(path))
    # print('='*10, '\n', path, '\n', operations_list)

    img = None
    for operation, operation_opt in operations_list:
        path, img = operation(path, operation_opt, img)

    process_info = f'Processing {basename}{ext} ...'
    return process_info

