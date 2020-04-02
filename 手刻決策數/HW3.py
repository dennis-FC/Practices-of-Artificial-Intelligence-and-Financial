# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:09:56 2019

@author: user
"""

import math
import numpy as np
import pandas as pd
def entropy(p1,n1):#亂度
    if(p1==0 and n1==0):
        return 1
    elif(p1==0):
        return 0
    elif(n1==0):
        return 0
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    return -pp*math.log2(pp)-pn*math.log2(pn)

def IG(p1,n1,p2,n2):
    num1 = p1+n1
    num2 = p2+n2
    num = num1+num2
    return entropy(p1+p2,n1+n2)-(num1/num*entropy(p1,n1)+num2/num*entropy(p2,n2))


#read PlayTennis.txt file
from sklearn import datasets
iris = datasets.load_iris()
np.random.seed(38)
idx = np.random.permutation(150)
feature = iris.data[idx,:]
target = iris.target[idx]
total = 0
table = [[0,0,0],
         [0,0,0],
         [0,0,0]]
for batch in range(0,150,30):#150個分成五群每次拿四群建模一群測試
    print("k=",int(batch/30)+1)
    
    X_train = np.delete(feature, range(batch, batch+30), axis=0)#120個特徵
    y_train = np.delete(target, range(batch,batch+30), axis=0)#120個正解
    #以上去建模
    #以下去測試
    X_test = feature[batch:batch+30,:]
    y_test = target[batch:batch+30]
    
    vote = [[] for i in range(30)] #投票
    final = []
    
    
    for ex in range(3):#target=(1,2),(0,2),(0,1)個建一棵樹
        print("tree",ex+1)
        X_train_filtered = []
        y_train_filtered = []
        
        for i in range(0,120):
            if y_train[i]!=ex:#去除0,1,2
                X_train_filtered.append(X_train[i])
                y_train_filtered.append(y_train[i])
        X_train_filtered = np.array(X_train_filtered)
        y_train_filtered = np.array(y_train_filtered)
        

        ans = np.unique(y_train_filtered)#剩下[1,2]...
        print("ans=",ans)
        
        #開始建決策樹
        node = dict()
        node['data'] = range(len(y_train_filtered))
        Tree = [];
        Tree.append(node)
        t = 0
        while(t<len(Tree)):#建樹
            idx = Tree[t]['data']
            if len(set(y_train_filtered[idx]))==1:#
                Tree[t]['leaf'] = 1
                Tree[t]['decision'] = np.unique(y_train_filtered[idx]).sum()#如果只剩一個[1]or[2]or[3] sum的數就是他決定的值
            else:
                bestIG = 0
                for i in range(X_train_filtered.shape[1]):
                    pool = list(set(X_train_filtered[idx,i]))
                    pool.sort()
                    for j in range(len(pool)-1):
                        thres = (pool[j]+pool[j+1])/2
                        G1 = []
                        G2 = []
                        for k in idx:
                            if(X_train_filtered[k,i]<=thres):
                                G1.append(k)
                            else:
                                G2.append(k)
                        thisIG = IG(sum(y_train_filtered[G1]==ans[0]),sum(y_train_filtered[G1]==ans[1]),sum(y_train_filtered[G2]==ans[0]),sum(y_train_filtered[G2]==ans[1]))
                        if(thisIG>bestIG):
                            bestIG = thisIG
                            bestG1 = G1
                            bestG2 = G2
                            bestthres = thres
                            bestf = i
                if(bestIG>0):
                    Tree[t]['leaf']=0
                    Tree[t]['selectf']=bestf
                    Tree[t]['threshold']=bestthres
                    Tree[t]['child']=[len(Tree),len(Tree)+1]
                    node = dict()
                    node['data'] = bestG1
                    Tree.append(node)
                    node = dict()
                    node['data'] = bestG2
                    Tree.append(node)
                else:
                    Tree[t]['leaf']=1
                    if(sum(y_train_filtered[idx]==ans[0])>sum(y_train_filtered[idx]==ans[1])):
                        Tree[t]['decision']=ans[0]
                    else:
                        Tree[t]['decision']=ans[1]
            t+=1
            
        for i in range(len(y_test)):#測樹
            test_feature=X_test[i,:]
            now = 0
            while(Tree[now]['leaf']==0):
                if(test_feature[Tree[now]['selectf']]<=Tree[now]['threshold']):
                    now = Tree[now]['child'][0]
                else:
                    now = Tree[now]['child'][1]
            vote[ex].append(Tree[now]['decision'])
            #print(y_test[i],Tree[now]['decision'])
    for i in range(30):
        if vote[0][i]==vote[1][i]==vote[2][i]:#表決如果三種各一票的話，則歸類為2
            final.append(vote[0][i])
        elif vote[0][i]==vote[1][i]:
            final.append(vote[0][i])
        elif vote[0][i]==vote[2][i]:
            final.append(vote[0][i])
        elif vote[2][i]==vote[1][i]:
            final.append(vote[1][i])
        else: final.append(2)
        print(y_test[i],final[i])
        if(y_test[i]==final[i]):#如果一樣就加一
            total = total+1
            
        a = y_test[i]
        b = final[i]
        table[a][b] = table[a][b]+1#計算table的值，confusion matrix 看預測跟正確差多少
            
        
        
print("accuracy",total/150) #算平均有幾個一樣
#print(table)   

print(pd.DataFrame(table))





            
    #print(vote[0],vote[1],vote[2])
'''
for i in range(len(target)):#測樹
    test_feature=feature[i,:]
    now = 0
    while(Tree[now]['leaf']==0):
        if(test_feature[Tree[now]['selectf']]<=Tree[now]['threshold']):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]
    print(target[i],Tree[now]['decision'])
 '''