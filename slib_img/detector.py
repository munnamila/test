# coding: utf-8
# author: Fumihiro UEKI
# date: 2019.07.26
# description: Dlibによる顔検出

import numpy as np
import cv2
import dlib
import os
from align import Align

class Detector(Align):

    def __init__(self, dim=128):
        '''
        コンストラクタ
        '''

        self.home_path = os.path.expanduser('~')
        self.dim = dim
        self.detector = dlib.get_frontal_face_detector()

        if os.path.exists(self.home_path + "/.fulib") is False:
            os.mkdir("%s/.fulib" % (self.home_path))

        # shape_predictor_68_face_landmarks.datがない場合はダウンロードする．
        if os.path.isfile("%s/.fulib/shape_predictor_68_face_landmarks.dat" % (self.home_path)) is False:
            print("Downloading http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P %s/.fulib" % (self.home_path))
            os.system("bunzip2 %s/.fulib/shape_predictor_68_face_landmarks.dat.bz2" % (self.home_path))

        predictor_path = "%s/.fulib/shape_predictor_68_face_landmarks.dat" % (self.home_path)
        self.predictor = dlib.shape_predictor(predictor_path)
        
        super(Detector, self).__init__(self.dim)

    def face_detector(self, src):
        '''
        顔の検出を行う関数
        src : 入力画像
        return : 切り出した顔画像、検出したランドマーク、顔領域のボックス
        顔の検出を行う関数
        '''
        img = src.copy()

        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 顔のボックスを取得
        face_rect = self.detector(rgbimg)

        # 顔領域を切り出した画像を格納するリスト
        faces = []
        # 顔領域のボックスを格納するリスト
        boxes = []
        # 検出したランドマークを格納するリスト
        landmarks = []
        
        # 検出した顔の数、処理を繰り返す
        for i, rect in enumerate(face_rect):
            # ボックスの座標
            # xはボックスの左上のx座標
            x = rect.left()
            # yはボックスの左上のy座標
            y = rect.top()
            # ボックスの幅
            w = rect.right()
            # ボックスの高さ
            h = rect.bottom()
            
            boxes.append([x, y, w, h]) 

            # 座標が0よりも小さい場合は0とする
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            # ランドマークの検出
            landmark = self.find_landmark(rgbimg, rect)
            landmarks.append(landmark)

            # 入力画像から顔領域を切り出す
            crop = img[y : h, x : w]
            faces.append(crop)

        return faces, landmarks, boxes

    def find_landmark(self, rgb_img, box):
        '''
        ランドマークの検出
        rgb_img : 入力画像(RGBb画像)
        box : 顔領域のボックス
        '''

        result = self.predictor(rgb_img, box).parts()
        landmark = np.array([[point.x, point.y] for point in result], dtype=np.int32)
        return landmark

    def show_landmark(self, img, landmark):
        '''
        ランドマークの可視化
        img : 入力画像(BGR画像)
        landmark : ランドマーク
        '''
        for x, y in landmark:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        return img

if __name__ == "__main__":

    dim = 128

    obj = Detector(dim = dim)

    cap = cv2.VideoCapture(0)
    
    while 1:

        ret, frame = cap.read()

        if ret is False:
            break

        faces, landmarks, boxes = obj.face_detector(frame)

        for face, landmark, box in zip(faces, landmarks, boxes):
            x, y, w, h = box
            #align = obj.align(frame, landmark, obj.INNER_EYES_AND_BOTTOM_LIP)
            #align = obj.align(frame, landmark, obj.NOSE_AND_LIP)
            align = obj.align(frame, landmark, obj.OUTER_EYES_AND_TOP_LIP)
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255))
            obj.show_landmark(frame, landmark)
            face = cv2.resize(face, (dim, dim))
            frame[0 : dim, 0 : dim] = align
            frame[0 : dim, dim : dim + dim] = face 

        cv2.imshow("window", frame)

        key = cv2.waitKey(30)
        
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
