
# 必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import pandas as pd

class AutoEncoder(nn.Module):
    '''
    ネットワークの構造と，順方向の計算を定義したクラス。
    '''

    def __init__(self):
        '''
        コンストラクタ
        ここで、ネットワークの各層を定義する。
        '''
        super(AutoEncoder,self).__init__()
        
        #encoder
        
        self.con1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)#32,128,128
        self.con2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)#64,64,64
        self.con3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)#64,32,32
        self.con4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)#64,16,16
        self.con5 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)#128,8,8
        self.con6 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)#128,4,4
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        
        #decoder
        
        self.fc4 = nn.Linear(64,128)
        self.fc5 = nn.Linear(128,256)
        self.fc6 = nn.Linear(256,512)
        
        self.upsample = nn.Upsample(scale_factor=2,mode="nearest")
        
        self.con7 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)#128,4,4
        self.con8 = nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)#64,16,16
        self.con9 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)#64,32,32
        self.con10 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)#64,64,64
        self.con11 = nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1)#32,128,128
        self.con12 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)#1,128,128
        


    def forward(self, input):

        #encoder
        h1 = F.relu(self.con1(input)) #32,128,128
        h2 = self.maxpool(h1) #32,64,64
        h3 = F.relu(self.con2(h2)) # 64,64,64
        h4 = self.maxpool(h3) #64,32,32
        h5 = F.relu(self.con3(h4)) #64,32,32
        h6 = self.maxpool(h5) #64,16,16
        h7 = F.relu(self.con4(h6)) #64,16,16
        h8 = self.maxpool(h7) #64,8,8
        h9 = F.relu(self.con5(h8)) #128,8,8
        h10 = self.maxpool(h9) #128,4,4
        h11 = F.relu(self.con6(h10)) #128,4,4
        h12 = self.maxpool(h11) #128,2,2

        
        
        h13 = F.relu(self.fc1(h12.view(-1,128*2*2)))#256
        h14 = F.relu(self.fc2(h13))#128
        encoded = F.relu(self.fc3(h14))#64
        
        #decoder
        h15 = F.relu(self.fc4(encoded))#128
        h16 = F.relu(self.fc5(h15))#256
        h17 = F.relu(self.fc6(h16))#512
        
        h18 = h17.view(h17.size(0),128,2,2)#128,2,2
        h19 = self.upsample(h18) #128,4,4
        h20 = F.relu(self.con7(h19)) #128,4,4
        h21 = self.upsample(h20) #128,8,8
        h22 = F.relu(self.con8(h21)) #64,8,8
        h23 = self.upsample(h22) #64,16,16
        h24 = F.relu(self.con9(h23)) #64,16,16
        h25 = self.upsample(h24) #64,32,32
        h26 = F.relu(self.con10(h25)) #64,32,32
        h27 = self.upsample(h26) #64,64,64
        h28 = F.relu(self.con11(h27)) #32,64,64
        h29 = self.upsample(h28) #32,128,128
        decoded = torch.sigmoid(self.con12(h29)) #1,128,128

        
        
        
        return encoded, decoded
