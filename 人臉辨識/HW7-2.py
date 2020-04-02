# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:03:52 2019

@author: Po-An Chen
"""

from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt

I = Image.open('picture.jpg')
I = I.resize((266,190),Image.ANTIALIAS)
I2 = I.convert("L")
I2 = np.array(I2)

#(x,y) = I.size
#print(x,y)
#x_s = 266
#y_s = y*x_s//x
#print(x_s,y_s)

#data = np.asarray(I)
#data2 = data.copy()
#data = data.astype('float64')
#gray = (data[:,:,0]+data[:,:,1]+data[:,:,2])/3
#data2[:,:,0] = gray
#data2[:,:,1] = gray
#data2[:,:,2] = gray
#
#I2 = Image.fromarray(data2,"RGB")
#I2.show()

npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]#2429
trnn = trainnonface.shape[0]#4548
tepn = testface.shape[0]#472
tenn = testnonface.shape[0]#23578

fn = 0
ftable = []#記錄各種特徵的位置編號(總共四類)
#第一類長方形特徵，算有幾個合法的特徵
for y in range(19):#圖片高度
    for x in range(19):#圖片寬度
        for h in range(2,20):#長方形特徵高度
            for w in range(2,20):#長方形特徵寬度
                if(y+h<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([0,y,x,h,w])#將在圖片內的長方形特徵的位址編號紀錄
print(fn)
#第二類長方形特徵，算有幾個合法的特徵           
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19):
                    fn = fn + 1
                    ftable.append([1,y,x,h,w])
print(fn)
#第三類長方形特徵，算有幾個合法的特徵
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19):
                    fn = fn + 1
                    ftable.append([2,y,x,h,w])
print(fn)
#第四類長方形特徵，算有幾個合法的特徵
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([3,y,x,h,w])
print(fn)#總共有36648個可行的長方形特徵

def fe(sample,ftable,c):#第c個特徵，此函數主要是要output積分值
    ftype = ftable[c][0]#哪一類長方形特徵
    #座標編號
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))
    if(ftype==0):#第0類長方形特徵
        idx1 = T[y:y+h,x:x+w].flatten()#白色
        idx2 = T[y:y+h,x+w:x+w*2].flatten()#黑色
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)#6000張白色-6000張黑色(一起算特徵)
    elif(ftype==1):#第1類長方形特徵
        idx1 = T[y:y+h,x:x+w].flatten()#白色
        idx2 = T[y+h:y+h*2,x:x+w].flatten()#黑色
        output = np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx1],axis=1)
    elif(ftype==2):#第2類長方形特徵
        idx1 = T[y:y+h,x:x+w].flatten()#白色
        idx2 = T[y:y+h,x+w:x+w*2].flatten()#黑色
        idx3 = T[y:y+h,x+w*2:x+w*3].flatten()#白色
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)+np.sum(sample[:,idx3],axis=1)
    else:#第3類長方形特徵
        idx1 = T[y:y+h,x:x+w].flatten()#白色
        idx2 = T[y:y+h,x+w:x+w*2].flatten()#黑色
        idx3 = T[y+h:y+h*2,x:x+w].flatten()#黑色
        idx4 = T[y+h:y+h*2,x+w:x+w*2].flatten()#白色
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx3],axis=1)+np.sum(sample[:,idx4],axis=1)
    return output

trpf = np.zeros((trpn,fn)) #2429X36648個積分值
trnf = np.zeros((trnn,fn)) #4548X36648個積分值
for c in range(fn):#第c個特徵
    trpf[:,c] = fe(trainface,ftable,c)
    trnf[:,c] = fe(trainnonface,ftable,c)
    
def WC(pw,nw,pf,nf):#弱分類器 #pw是positive權重, nw是negative權重, pf是positive feature , nf是negative feature
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf#minmax的1/10處當第一刀(當作一個標準)，大於theta應該為positive 小於theta應該為negative
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])#計算分錯的，因此如果錯越多error越多
    polarity = 1 #0代表以上為正，以下為負 。1代表以上為負，以下為正
    if(error>0.5):#如果錯誤太多，不如換一個方向
        polarity = 0
        error = 1 - error
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10):#開始測試如果切其他地方的效果
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            polarity = 0
            error = 1 - error
        if(error<min_error):#要找到能夠error最小的那一刀
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity

pw = np.ones((trpn,1))/trpn/2
nw = np.ones((trnn,1))/trnn/2

SC = []#存stronge classify
for t in range(100):#存幾大特徵
    #正規畫
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0#設第零個是最好的特徵
    for i in range(1,fn):
        me,mt,mp = WC(pw,nw,trpf[:,i],trnf[:,i])
        if(me<best_error):
            best_error = me
            best_feature = i
            best_theta = mt
            best_polarity = mp
    beta = best_error/(1-best_error)
    if(best_polarity==1):
        pw[trpf[:,best_feature]>=best_theta]*=beta#分對的然後更新權重
        nw[trnf[:,best_feature]<best_theta]*=beta#分對的然後更新權重
    else:
        pw[trpf[:,best_feature]<best_theta]*=beta#分錯的然後更新權重
        nw[trnf[:,best_feature]>=best_theta]*=beta#分錯的然後更新權重
    alpha = np.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha])
    print(t)
    print([best_feature])
    

pic_num = (266-19+1)*(190-19+1)#原圖中有幾張19X19
Po_pic = np.zeros((pic_num,361))#m*361
#將原圖變為  m*361
x = 0
xytable = []
for i in range(190-18):
    for j in range(266-18):
        Po_pic[x,:] = I2[i:i+19,j:j+19].reshape(1,-1)
        x = x+1
        xytable.append([i,j])#記住座標

point = np.zeros((42656,fn))
for c in range(fn):#第c個特徵#算原圖的積分值
    point[:,c] = fe(Po_pic,ftable,c)
    

    
final_point = np.zeros((42656,1))#算分
alpha_sum = 0
#利用訓練好的100個特徵算哪張圖最有可能是人臉
for i in range(100):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if(polarity==1):
        final_point[point[:,feature]>=theta] += alpha
    else:
        final_point[point[:,feature]<theta] += alpha
final_point/=alpha_sum   
#如果分數大於某個門檻值就把它框起來
count = 0
for i in range(42656):
    if(final_point[i]>0.53):
        count+=1
        a,b = xytable[i]
        shape = [(b,a), (b+19,a+19)]
        rect = ImageDraw.Draw(I)
        rect.rectangle(shape,outline = "red")  
print(count)
I.show
#x = []
#y = []
#for i in range(1000):
#    threshold = i/1000
#    x.append(np.sum(trns>=threshold)/trnn)
#    y.append(np.sum(trps>=threshold)/trpn)
#
#    plt.plot(x,y)