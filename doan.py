from os.path import splitext

import cv2
import imutils
import numpy as np
from keras.models import model_from_json
from local_utils import detect_lp

imagepath = r'D:\LapTrinhPyThon\SourceCode\NumberPlateRecognition-main\code\Example\image_test.jpg'
def pre_process_img(self, img, resize=False):

        # đưa ảnh về thang màu RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow('anh mau rgb', img)
 

        # chuẩn hóa các pixel về dải 0-1
        img = img / 255
       # cv2.imshow('anh img', img)

        return img
def get_plate(self, img, wpod_net):
        Dmax = 608
        Dmin = 256
        try:
            img_pre_process = self.pre_process_img(img)
            # print(img_pre_process.shape[:2])

            check = float(
                max(img_pre_process.shape[:2])) / min(img_pre_process.shape[:2])

            side = int(check * Dmin)
            # print(side)
            bound_dim = min(side, Dmax)

            # tìm ra vùng ảnh chứ biển số
            _, place_img, _, cor = detect_lp(
                wpod_net, img_pre_process, bound_dim, lp_threshold=0.5)
        except:
            place_img = None
            cor = None
        return place_img, cor