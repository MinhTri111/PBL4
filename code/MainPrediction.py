from collections import defaultdict
from os.path import splitext
from typing import OrderedDict
import cv2
import os
import imutils
import numpy as np
from keras.models import model_from_json
from pyasn1.type.univ import Integer
from local_utils import detect_lp
import PIL
from PIL import Image
import pyrebase
from firebase_admin import db
import firebase_admin
from firebase_admin import credentials


class Model:
    # Hàm key lúc sort
    def takeFirst(self, elem):
        return elem[0]

    # Load WPOD để detect vùng biển
    def load_model(self, path):
        try:
            path = splitext(path)[0]
            with open('%s.json' % path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json, custom_objects={})
            model.load_weights('%s.h5' % path)
            print("Loading model successfully...")
            return model
        except Exception as e:
            print(e)

    # tiền xử lý ảnh
    def pre_process_img(self, img, resize=False):

        # đưa ảnh về thang màu RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     #   cv2.imshow('anh mau rgb', img)
 

        # chuẩn hóa các pixel về dải 0-1
        img = img / 255
       # cv2.imshow('anh img', img)

        return img

    # lầy ra vùng ảnh chứa biển số
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

    # xử lý ảnh
    def process_img(self, place_img):

        # kiểm tra có ảnh vùng biển
        if (len(place_img)):
            # chuyển đổi đưa ảnh về hệ 8bit
            plate_img = cv2.convertScaleAbs(place_img[0], alpha=(255.0))
            # chuyển ảnh sang thang xám và làm mở
            img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

            # ================
            # Phân ngưỡng ảnh
            # binary = cv2.threshold(img_blur, 170, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # binary = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)\
            binary = cv2.adaptiveThreshold(
                img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imshow("Anh bien so sau threshold", binary)
            # cv2.imwrite("anhbienso2.jpg", binary)
            # dãn nở nhằm khử nhiễu và đưa ra ảnh đã được xử lý
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_thre = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        return img_thre

    # tìm ra đường viền bao
    def find_contours(self, img_thre):
        point = []
        # cv2.imshow("anh trong find_contours", img_thre)
        # tìm ra các đường viền trong ảnh
        contours = cv2.findContours(
            img_thre.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)
        # contours = sorted(contours, key = cv2.contourArea, reverse = True)[:7]

        # duyệt qua các đường bao
        for c in contours:
            # tính chu vi, tìm ra hình vuông xấp xỉ bao quanh và đưa ra tọa độ, dài, rộng
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            # kiểm ra các ô vuông, chỉ ra các ô chứa ký tự
            if 17 <= w <= 55 and 63 <= h <= 86:
                point.append((x, y, w, h))
        return point

    # sắp xếp lại các ký tự đúng theo thứ tự biển số
    def sort_point(self, img_thre, point):
        point1 = []
        point2 = []
        rate = img_thre.shape[1] / img_thre.shape[0]

        # biển chữ nhật
        if rate > 3:
            point.sort(key=self.takeFirst)

        # biển vuông
        else:
            mean = np.mean(np.array(point), axis=0)[1]
            for x in point:
                if x[1] < mean:
                    point1.append(x)
                else:
                    point2.append(x)
            point1.sort(key=self.takeFirst)
            point2.sort(key=self.takeFirst)
            point = point1 + point2
        return point

    # tách các ký tự ra làm từng ảnh con và đưa về dạng 30*60 rồi chuyển về dạng thích hơp làm input cho model
    def find_char(self, point, img_thre, place_img):
        character = []
        lis_im = []
        point = self.sort_point(img_thre, point)
        for i in point:
            # vẽ ô vuông quanh các ký tự
            x, y, w, h = i
            im1=cv2.rectangle(place_img[0], (x, y), (x + w, y + h), (0, 255, 0), 2)



            # bóc các ký tự về ảnh con
            pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            char = cv2.warpPerspective(img_thre, matrix, (w, h))
            lis_im.append(char)
           # cv2.imshow('anh',char)
           # cv2.waitKey(2000)

            # đưa về dạng thích hợp để làm input cho model
            char = cv2.resize(char, (30, 60))
            char = char.reshape(-1, 30 * 60)
          
            character.append(char)
        character = np.array(character, dtype=np.float32)
        return character,im1

    def recognize(self, model_svm, character):
        string = ''

        # dự đoán các ký tự
        for i in range(character.shape[0]):
            result = model_svm.predict(character[i])

            result = int(result[1][0][0])

            # chuyển lại mã ascii về ký tự
            result = chr(result)

            string += result
        return string

    # format lại string vừa nhận diện
    def format(self, string):
        string_new = ''
        if len(string) == 8:
            for i in range(len(string)):
                if i == 2:
                    string_new += '-'
                elif i == 4:
                    string_new += ' '
                string_new += string[i]
        elif len(string) == 9:
            for i in range(len(string)):
                if i == 2:
                    string_new += '-'
                elif i == 4:
                    string_new += ' '
                elif i == 7:
                    string_new += '.'
                string_new += string[i]
        else:
            string_new = string
        return string_new

    # viết biển số lên ảnh

    def draw_box(self, img, cor, string):
        pts = []

        # tọa x của 4 góc biển số
        x = cor[0][0]

        # tọa y của 4 góc biển số
        y = cor[0][1]

        for i in range(4):
            pts.append([int(x[i]), int(y[i])])
        pts = np.array(pts)
        peri = cv2.arcLength(pts, True)
        approx = cv2.approxPolyDP(pts, 0.018 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        # vẽ ô vuông và biển số lên ảnh
        cv2.putText(img, string, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, 255, 2)
        veo=cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return veo


def predict(path):
    # Load model
    model_svm = cv2.ml.SVM_load(r'D:\SoureDoAn\SDoAn\code\model_svmNew.xml')
    model = Model()
    wpod_net = model.load_model(r"D:\SoureDoAn\SDoAn\code\wpod-net.json")
    # print(type(wpod_net))

    img_path = path
    img = cv2.imread(img_path)
    #set lai do camera dien thoai qua to
    (h,w,d)=img.shape
    a= h//4
    img = img[a:w,0:h]
   # img = path
    place_img, cor = model.get_plate(img, wpod_net)
    assert place_img is not None
    clone_img = place_img[0].copy()
    if place_img is not None and cor is not None:
        img_thre = model.process_img(place_img)

        point = model.find_contours(img_thre)

        character,im1 = model.find_char(point, img_thre, place_img)

        string = model.recognize(model_svm, character)

        string = model.format(string)

        im=model.draw_box(img, cor, string)
       
       # cv2.imshow('anhkhoanhkitu',cv2.resize(im1,dsize=(800,600)))
       # cv2.imshow('anhkhoanh',cv2.resize(im,dsize=(1000,1000)))
        

        return string, clone_img * 255
    else:
        return None, None

config = {
  "apiKey": "AIzaSyCc9R34Svpox90nF6p74eKHCoVQl5I_7eQ",
  "authDomain": "mychatapp-91e9c.firebaseapp.com",
  "databaseURL": "https://mychatapp-91e9c-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "mychatapp-91e9c",
  "storageBucket": "mychatapp-91e9c.appspot.com",
  "messagingSenderId": "536107617441",
  "appId": "1:536107617441:web:bd38f4aaa547c031d8b4e0",
  "measurementId": "G-F4KFWF257K"
}

 
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
datafire = firebase.database()
cred = credentials.Certificate(r"D:\SoureDoAn\SDoAn\code\firebase-sdk.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':'https://mychatapp-91e9c-default-rtdb.asia-southeast1.firebasedatabase.app/'
})




while(True):
    ref = db.reference("/")
    all = datafire.get()
    list1=[]
    list2=[]
    valstr=""
    keystr=""
    for values in all.each():
        
        k = dict(values.val())
        list1 = list(k.values())
        print(list1)
        list2 = list(k.keys())
        print(list2[0]+"day la key")
        if(list1[0]==""):
            storage.child(str(list2[0])+".jpg").download(filename="testfirebase.jpg",path=os.path.basename(str(list2[0])+".jpg"))
            path = r"D:\SoureDoAn\SDoAn\testfirebase.jpg" 
            t,img=predict(path)
            ref.child(values.key()).update({list2[0]:t})
            print(t)
   
    #cv2.imshow('anh',img)
    if(cv2.waitKey(1)==ord('q')):
        break


