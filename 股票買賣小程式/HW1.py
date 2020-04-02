# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

T = int(input())
for t in range(T):
    N = int(input())
    buy = []
    sell = []
    stockprice = 0
    for n in range(N):
        str = input()
        strlist = str.split()
        share = int(strlist[1])
        price = int(strlist[-1])
        if(strlist[0]=='buy'):
            while len(sell)>0:
                order = sell[0]
                if(order[0]>price):
                    break
                dealno = min(share,order[1])
                stockprice = order[0]
                order[1] -= dealno
                share -= dealno
                if(order[1]==0):
                    del sell[0]
                if(share==0):
                    break
            if(share>0):
                i = 0
                while((i<len(buy))and(price<buy[i][0])):
                    i+=1
                if((i<len(buy))and(price==buy[i][0])):
                    buy[i][1]+=share
                else:
                    buy.insert(i,[price,share])
# TODO
        else:
            while len(buy)>0:
                order = buy[0]
                if(order[0]<price):
                    break
                dealno = min(share,order[1])
                stockprice = price
                order[1] -= dealno
                share -= dealno
                if(order[1]==0):
                    del buy[0]
                if(share==0):
                    break
            if(share>0):
                i=0
                while((i<len(sell))and(price>sell[i][0])):
                    i+=1
                if((i<len(sell))and(price==sell[i][0])):
                    sell[i][1]+=share
                else:
                    sell.insert(i,[price,share])
        
        if(len(buy)>0):
            bid=buy[0][0]
        else:
            bid="-"
            
        if(len(sell)>0):
            ask=sell[0][0]
        else:
            ask="-"
        if(stockprice==0):
            stockprice="-"
        
            
        print(ask,bid,stockprice)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                        
                
                    
                    
                    
                    
                    
                    
                    
                    
                
                
                
                
                
                
                
                
                