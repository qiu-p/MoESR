import cv2
from ultralytics import YOLO
import os
import numpy as np

# 这里的cordinate的规则[x[0],y[0],x[1],y[1]]分别代表图片左上和右下坐标
# "dog" : [(img1(低分辨率原图), mask)] 
'''
test_path = './test/6.png'
save_path = './seg_result6/'
model_path = "./yolov8x-oiv7.pt"
'''
class Seg_RecPic:
    '''
    表示待分割图片的类
    
    Attributes :
    input_path
    model_path
    save_path
    img_dic : A dictionary "seg_type" : [[input_path,[[cordi1], [cordi2]...]]]
    '''
    def __init__(self, lr_dir, model_path, gt_dir):
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.model_path = model_path
        self.img_dic = {"people" : [lr_dir, []],"car" : [lr_dir, []],"plane" : [lr_dir, []],"dog" : [lr_dir, []],"cat" : [lr_dir, []]}
        self.Tseg_type = {381 : "people",322 : "people",594 : "people" ,90 : "car",3 : "plane",160 : "dog",96 : "cat"} # 这里利用一个字典指定需要分割的类别 322 : "Man",594 : "Woman"
        self.type_tuple = (381, 90, 3, 160, 96, 322, 594) # 分割类别对应的元组

    @staticmethod
    def iscover(cordi1,cordi2):
        """判断两张分割图片是否pic_1包含于pic_2
        @param cordi1,cordi2: list(tensor) 两张图片的坐标 
        @return const: int 1表示有包含
        """
        if cordi1[0] >= cordi2[0]  and cordi1[1] >= cordi2[1]\
        and cordi1[2] <= cordi2[2] and cordi1[3] <= cordi2[3]:
            return 1
        else:
            return 0
    
    def seg_pic(self):
        '''
        if os.path.isdir(save_path):
            print('save_file already exists.')
            os._exit(0)
        else:
            os.makedirs(save_path)
        '''
        model = YOLO(self.model_path)
        results = model(self.lr_dir)
        for result in results:
            boxes = result.boxes  # Boxes 对象，用于边界框输出
            for seq1,val1 in enumerate(boxes.cls):
                if val1 in self.type_tuple:
                    cordi1 = list(map(int,list(boxes.xyxy[seq1])[:]))
                    Issubval = 0 # 这里是用于判断是否有同类别的子图
                    if seq1 != boxes.cls.shape[0]-1:
                        for seq2,val2 in enumerate(boxes.cls[seq1+1:],seq1+1):
                            if val2 in self.type_tuple:
                                cordi2 = list(map(int,list(boxes.xyxy[seq2])[:]))
                                if self.iscover(cordi1, cordi2) and (val1 - val2) < 1e-4:
                                    Issubval = 1
                            else:
                                continue
                    if Issubval == 0:
                        self.img_dic[self.Tseg_type[int(val1)]][1].append(cordi1)
                else:
                     continue