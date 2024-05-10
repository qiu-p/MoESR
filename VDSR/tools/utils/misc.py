import numpy as np
import os
import shutil
import random
import time
import torch
from os import path as osp



def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def is_file_exist(file_path, rename=False, delete=False):
    '''
    - Args:
      - file_path (str): file path
      - rename (bool): if path exists, whether to rename it (rename it with timestamp)
      - delete (bool): if path exists, whether to delete it
    - Return:
      - (bool): whether the file exit
    '''
    if osp.exists(file_path):
        if rename:
            file_path_no_ext, ext = osp.splitext(file_path)
            new_name = file_path_no_ext + '_archived_' + get_time_str() + ext
            print(f'File already exists. Rename it to {new_name}', flush=True)
            os.rename(file_path, new_name)
        elif delete:
            os.remove(file_path)
        return True
    else:
        return False

def make_dir(path, rename=False, delete=False, keep=False):
    """mkdirs

    Args:
        path (str): Folder path.
        rename (bool): if path exists, whether to rename it (rename it with timestamp and create a new one)
        delete (bool): if path exists, whether to delete it
        keep (bool): if path exists, whether to keep it and do not make a new one
    Return:
        is_exist (bool): whether the dir has existed
    """
    is_exist = False
    if osp.exists(path):
        is_exist = True
        if rename:
            new_name = path + '_archived_' + get_time_str()
            print(f'Path already exists. Rename it to {new_name}', flush=True)
            os.rename(path, new_name)
        elif delete:
            shutil.rmtree(path)
        elif keep:
            return is_exist
    # exist_ok：是否在目录存在时触发异常。
    # exist_ok = False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
    # exist_ok = True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
    os.makedirs(path)
    return is_exist

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    # exist_ok：是否在目录存在时触发异常。
    # exist_ok = False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
    # exist_ok = True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
    os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'

def get_from_dict(dict, key, replace_none = None):
    """ 
    get value from dictionary
    Aegs:
        dict (dict):
        key (int, str, ...):
        replace_none: the value to replace the return if the original value is None
    Return:
        (bool): whethe the key exists
        (value): value 
    """
    if key not in dict:
        return False, None
    else:
        value = dict.get(key)
        if value == None:
            value = replace_none
        return True, value


def change_filename_with_tmpl(path, filename_tmpl):
    '''
    Args:
        path: eg. AAA/BBB/CC.txt
        filename_tmpl: eg. {}_test
    Return:
        eg. AAA/BBB/CC_test.txt
    '''
    dir = osp.dirname(path)
    file_name = osp.basename(path)
    basename, ext = osp.splitext(file_name)
    out_name = f'{filename_tmpl.format(basename)}{ext}'
    out_path = osp.join(dir, out_name)

    return out_path


def add_suffix_to_filename(path, index):
    '''
    Args:
        path: eg. AAA/BBB/CC.txt
        index (int): eg. 1
    Return:
        eg. AAA/BBB/CC_1.txt
    '''
    filename_tmpl = '{}_' + str(index)
    change_filename_with_tmpl(path, filename_tmpl)
