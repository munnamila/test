#author: sou_meirin
import cv2
import matplotlib.pyplot as plt
import glob
import torch
from torchvision.utils import save_image
class Img():
    

    def __init__(self, path):
        '''import picture'''

        import cv2

        self.path = path
        self.img = cv2.imread(self.path)

    def size(self):
        #I.size()
        import numpy as np

        print(self.img.shape)

    def gray(self):
        '''グレー化'''

        import cv2

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def flip(self):
        '''右左回転'''

        import cv2

        self.img = cv2.flip(self.img, 1)

    def choose_face_dlib(self,dim = 128):
    
        from detector import Detector
        import cv2
        import os
        import glob
        
        
        obj = Detector(dim = dim)
        
        frame = self.img
        
        faces, landmarks, boxes = obj.face_detector(frame)
        
        for face, landmark, box in zip(faces, landmarks, boxes):
            x, y, w, h = box
            #align = obj.align(frame, landmark, obj.INNER_EYES_AND_BOTTOM_LIP)
            align = obj.align(frame, landmark, obj.OUTER_EYES_AND_TOP_LIP)
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255))
            obj.show_landmark(frame, landmark)
            face = cv2.resize(face, (dim, dim))
            frame[0 : dim, 0 : dim] = align
            frame[0 : dim, dim : dim + dim] = face
        self.img = frame[0 : dim, 0 : dim]

    def choose_face_cv(self):

        from detector import Detector
        import cv2
        import os
        import glob       
        

        face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(r'/Users/songminglun/research/slib/slib_img/haarcascade_frontalface_default.xml')


        faces = face_cascade.detectMultiScale(self.img, scaleFactor = 1.1, minNeighbors = 5, minSize = (5, 5))

        print("Face : {0}".format(len(faces)))



        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]

        # for(x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2) 



        img_scr = self.img[y : y + h, x : x + w]

        img_scr = cv2.resize(img_scr,(64,64))

        self.img = img_scr

    def show(self):
        #I.show() #show the picture by plt

        import matplotlib.pyplot as plt
        import numpy as np
        import cv2

        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)

        if self.img.ndim is 3:      
            plt.imshow(self.img)
        elif self.img.ndim is 2:
            plt.imshow(self.img,cmap = 'gray')
        plt.show()

    def save(self, path = None):
        '''処理の画像を書き込み'''
        '''JPGとPNGに適用.upgrateする必要がある'''

        import cv2
        import os

        count = 1
        if path is None:
            while True:
                self.saved_path = self.path[:-4] + '_' + str(count) +'.jpg'
                if os.path.isfile(self.saved_path):
                    count += 1
                else:
                    break
        else:
            self.saved_path = path

        cv2.imwrite(self.saved_path , self.img)

    def cut_pic(self,l_or_r = 'right'):
        #cut the picture and save the part of right or left

        size = self.img.shape

        if l_or_r == 'right':

            self.img = self.img[0:size[0],int(size[1]/2):size[1]]

        if l_or_r == 'left':

            self.img = self.img[0:size[0],0:int(size[1]/2)]
    
    def resize(self,height,weight):

        self.img = cv2.resize(self.img, (height, weight))

    def show_hist(self):

        hist = cv2.calcHist([self.img],[0],None,[256],[0,255])

        plt.plot(hist,'r')

        plt.show()

    def hist(self):
        
        self.img = cv2.equalizeHist(self.img)



        

    def autoencoder(self):
        #done
        #before use encode, please do I.gray()
        #I.encode() return encoded

        import numpy as np
        import torch
        import torch.utils.data as Data
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F

        import autoencoder_model

        class CustumDataset(torch.utils.data.Dataset):
            '''
            自前のデータをDatasetクラスで管理する。
            '''
    
            def __init__(self, feature_list):
            
                
                self.feature = feature_list
            
            def __getitem__(self, index):
            
                
                x = self.feature[index]
                x = np.float32(x)
                x = torch.from_numpy(x)/255
                
                return x
            
            def __len__(self):
                '''
                データの個数を返却。
                '''
                return len(self.feature)
    

        img = [self.img]

        img = CustumDataset(img)

        train_loader = Data.DataLoader(img, batch_size=1, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = autoencoder_model.AutoEncoder()
        model.load_state_dict(torch.load("/Users/songminglun/research/slib/slib_img/model.torch",map_location='cpu'))
        model.to(device)

        loss_func = nn.MSELoss()
            # code = []

        with torch.no_grad(): # 勾配の計算をしないように設定
            
            for x in train_loader:
            
                x = x.view(-1,1,128,128).to(device) # CPU or GPUで計算をするように設定
                
                encoded, decoded = model(x) # フォワードの計算
                
                loss = loss_func(decoded, x)
                
                #print('loss:',loss)
            
                # code.append(encoded)

        return encoded, decoded

    def to_numpy(self):
        import numpy as np
        return np.array(self.img)


def show_img(img):
    #si.show_img(img) img:tensor
    import matplotlib.pyplot as plt
    import numpy as np
    
    w, h = img.shape[-2], img.shape[-1]

    plt.imshow(np.array(img).reshape(w,h),cmap = 'gray')
    plt.show()

    
def cvpaste(img, imgback, x, y, angle, scale):  
    import cv2
    import numpy as np
    # x and y are the distance from the center of the background image 

    r = img.shape[0]
    c = img.shape[1]
    rb = imgback.shape[0]
    cb = imgback.shape[1]
    hrb=round(rb/2)
    hcb=round(cb/2)
    hr=round(r/2)
    hc=round(c/2)

    # Copy the forward image and move to the center of the background image
    imgrot = np.zeros((rb,cb,3),np.uint8)
    imgrot[hrb-hr:hrb+hr,hcb-hc:hcb+hc,:] = img[:hr*2,:hc*2,:]

    # Rotation and scaling
    M = cv2.getRotationMatrix2D((hcb,hrb),angle,scale)
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Translation
    M = np.float32([[1,0,x],[0,1,y]])
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))

    # Makeing mask
    imggray = cv2.cvtColor(imgrot,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the forward image in the background image
    img1_bg = cv2.bitwise_and(imgback,imgback,mask = mask_inv)

    # Take only region of the forward image.
    img2_fg = cv2.bitwise_and(imgrot,imgrot,mask = mask)

    # Paste the forward image on the background image
    imgpaste = cv2.add(img1_bg,img2_fg)

    return imgpaste
    
def meger_imgs(path, save_path):
    files = sorted(glob.glob(path + '/*'))

    lists = []
    for i in files:
        img = cv2.imread(i, 0)
        h = img.shape[0]
        w = img.shape[1]
        img = img/255
        lists.append(img)
    a = torch.tensor(lists)


    def to_img(x, high, weigh):
        #x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, high, weigh)
        return x

    pic = to_img(a, h, w)

    save_image(pic, save_path)










if __name__ == "__main__":
    pass
