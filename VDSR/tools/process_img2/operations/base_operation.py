import os.path as osp
import os
import sys

from ...utils import scandir, make_dir

class BaseOperation():
    def __init__(self, opt, work_dir=None):
        self.mode = opt['mode']
        self.existed_dir = opt['existed_dir']
        if work_dir:
            self.work_dir = work_dir
        else:
            self.work_dir = opt['work_dir']
        self.save = opt['save']

        self.save_imgname_tmpl = opt.get('save_imgname_tmpl')
        self.save_dir = opt.get('save_dir')
        self.save_dir_tmpl = opt.get('save_dir_tmpl')
        self.mask_save_dir = opt.get('mask_save_dir')
        self.mask_save_dir_tmpl = opt.get('mask_save_dir_tmpl')

    def _make_save_dir(self, save_dir):
        def _make_dir(save_dir, existed_dir='raise'):
            if existed_dir == 'raise':
                print(f'mkdir {save_dir} ...')
                make_dir(save_dir)
            elif existed_dir == 'rename':
                print(f'mkdir {save_dir} ...')
                if make_dir(save_dir, rename=True):
                    print(f'rename origin {save_dir} ...')
            elif existed_dir == 'delete':
                print(f'mkdir {save_dir} ...')
                if make_dir(save_dir, delete=True):
                    print(f'delete origin {save_dir} ...')

        _make_dir(save_dir, self.existed_dir)
    
    def make_save_dirs(self, save_dir_prefix):
        '''
        save dir
        mask_save_dir
        '''
        # turn tmpl -> dir_name
        if self.save_dir == None:
            if self.save_dir_tmpl == None:
                self.save_dir_tmpl = '{}_' + self.mode
            self.save_dir = self.save_dir_tmpl.format(save_dir_prefix)
        if (self.mask_save_dir==None) and (self.mask_save_dir_tmpl!=None):
            self.mask_save_dir = self.mask_save_dir_tmpl.format(save_dir_prefix)

        save_dir_prefix = self.save_dir
        if self.save == False:
            return save_dir_prefix
            
        if (self.mask_save_dir!=None) or (self.mask_save_dir_tmpl!=None):
            self._make_save_dir(self.mask_save_dir)
        self._make_save_dir(self.save_dir)
        return save_dir_prefix