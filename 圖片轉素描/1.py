# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:52:19 2019

@author: user
"""
#圖片有雜訊
from PIL import Image
import numpy as np

I = Image.open('pic.jpg')
W,H = I.size
data = np.asarray(I)
data2 = data.copy()
data = data.astype('float64')
noise = np.random.normal(0,20,(H,W,3))

data3 = data+noise;
data3[data3>255] = 255
data3[data3<0] = 0
data2 = data3.astype('uint8')
I2 = Image.fromarray(data2,"RGB")
I2.show()