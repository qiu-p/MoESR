import numpy as np
import argparse

from tools.gen_dataset import gen, get_default_opt
from tools.utils import get_logger

def main():
    parser = argparse.ArgumentParser(description="gen dataset")
    parser.add_argument("--input_folder", type=str, default='', help="input_folder")
    parser.add_argument("--size", type=int, default=0, help="crop size")
    opt = parser.parse_args()

    gen_opt = get_default_opt()
    if opt.input_folder != '':
        gen_opt['input_folder'] = opt.input_folder
        gen_opt['save_foder_input'] = opt.input_folder + '_input'
        gen_opt['save_foder_label'] = opt.input_folder + '_label'
    if opt.size != 0:
        gen_opt['size_input'] = opt.size
        gen_opt['size_label'] = opt.size
        gen_opt['stride'] = opt.size
        gen_opt['thresh_size'] = opt.size # 阈值size 比 thresh_size 小的补丁的会被舍弃
    gen(gen_opt)

    print('===== END =====')


if __name__ == '__main__':
    main()
