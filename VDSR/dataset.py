import torch.utils.data as data
import torch
import os.path as osp
import numpy as np
import PIL.Image as pil_image

from tools.utils import scandir

class DatasetFromHdf5(data.Dataset):
    def __init__(self, input_folder, label_foder):
        super(DatasetFromHdf5, self).__init__()
        self.input_folder = input_folder
        self.label_foder = label_foder
        self.input_paths = list(scandir(input_folder, full_path=True))
        self.total_len = len(self.input_paths)

    def __getitem__(self, index):
        path = self.input_paths[index]
        img_input = pil_image.open(path, mode='r') # size: (width, height)
        img_label = pil_image.open(osp.join(self.label_foder, osp.basename(path)), mode='r')
        img_input = np.array(img_input.convert("YCbCr"))[:,:,0].astype(float) / 255
        img_label = np.array(img_label.convert("YCbCr"))[:,:,0].astype(float) / 255
        # img_input = np.transpose(img_input, (2,0,1))
        # img_label = np.transpose(img_label, (2,0,1))
        img_input = np.expand_dims(img_input, axis=0)
        img_label = np.expand_dims(img_label, axis=0)
        return torch.from_numpy(img_input).float(), torch.from_numpy(img_label).float()
        
    def __len__(self):
        return self.total_len

