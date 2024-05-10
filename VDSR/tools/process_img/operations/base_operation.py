import os.path as osp
import os
import sys

class BaseOperation():
    def __init__(self, opt):
        self.opt = opt
        self.filename_tmpl = opt['filename_tmpl']
        self.save_folder = opt['save_folder']

        # if not osp.exists(self.save_folder):
        #     os.makedirs(self.save_folder)
        #     print(f'mkdir {self.save_folder} ...')
        # else:
        #     print(f'Folder {self.save_folder} already exists. Exit.')
        #     sys.exit(1)