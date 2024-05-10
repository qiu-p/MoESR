import importlib
from copy import deepcopy
from os import path as osp
import os
import sys

from ...utils import scandir
from ...utils.registry import IMG_PROCESS_REGISTRY

__all__ = ['get_operation']

# automatically scan and import modules for registry
# scan all the files under current folder and collect files ending with
# '_operation.py'
folder = osp.dirname(osp.abspath(__file__))
filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(folder) if v.endswith('_operation.py')]
# import all the operation modules
mod = sys.modules[__name__]
_operation_modules = [importlib.import_module(f'.{file_name}', package=mod.__name__) for file_name in filenames]


def get_operation(opt):    
    opetation = IMG_PROCESS_REGISTRY.get(opt['mode'])
    return opetation