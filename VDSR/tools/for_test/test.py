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

def a():
    mod = sys.modules[__name__]
    mod2 = importlib.import_module(f'..folder01', package=mod.__name__)
    myAAA = getattr(mod2, 'AAA')(2)
    myAAA.get()