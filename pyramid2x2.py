from __future__ import print_function,unicode_literals
from __future__ import absolute_import,division
import struct
import gzip
import numpy as np
import time
import copy

lr=0.5   #learning rate

#sigmoid function/activation
def sigmoid(x):
    return 1/(1+np.exp(-x))
#sigmoid prime
def sigmoid_derivative(x):
    return x*(1-x)

bias =1

class Neural_NetWork(object):
    def __init__(self,h):
        #parameters
        self.input_size=784
        self.hidden_size=h
        self.output_size=10
        self.old_error=99999    #sum of error
        self.new_error=0
        self.o_error=999

        #The weight matrixes
        self.Weight_1=np.random.uniform(-2,2,(1,self.input_size))
        self.Weight_2=np.random.uniform(-2,2,(self.hidden_size,self.output_size))


    def feed_forward(self,X):
        self.zz=(X*self.Weight_1).reshape((196,4))
        self.z=np.sum(self.zz,axis=1)+bias  #sum of Weight and output
        self.z2=sigmoid(self.z)                      #hidden layer activation
        self.z3=np.dot(self.z2,self.Weight_2)+bias
        o=sigmoid(self.z3)                      #output layer activation
        return o

    def back_propagation(self,X,y,o):
        self.o_error=np.sum((y-o)**2)/2         #get sum of error/ accuracy check

        #get Err
        self.d_Et_Ot=-(y - o)
        self.d_o_net=sigmoid_derivative(o).reshape((1,self.output_size))
        self.d_net_w=self.z2.repeat(self.output_size).reshape(self.hidden_size,self.output_size)*(self.Weight_2**0)

        #get dError/dWeight for output layer
        xx= self.d_Et_Ot * self.d_o_net
        self.d_error_w= xx*self.d_net_w
        self.Weight_2-=lr*self.d_error_w

        #get dError/dWeight for hidden layer
        self.d_Eo_No=self.d_Et_Ot*self.d_o_net
        self.d_No_Oh=self.Weight_2

        self.d_Eo_Oh=self.d_Eo_No*self.d_No_Oh
        self.d_Et_Oh=np.sum(self.d_Eo_Oh,axis=1)

        self.d_Oh_Nh=sigmoid_derivative(self.z2)
        yy=self.d_Et_Oh*self.d_Oh_Nh
        self.d_Et_w=X*(yy.repeat(4)).reshape((1,4*self.hidden_size))
        self.Weight_1-=lr*self.d_Et_w


    def train(self,X,y):            #forward and back once/train once
        o=self.feed_forward(X)
        self.back_propagation(X,y,o)

train_lst=np.load('train_lst.npy')
test_lst=np.load('test_lst.npy')
train_label=np.load('train_label.npy')
test_label=np.load('test_label.npy')




# net=Neural_NetWork(196)
# lstp=[]
# for e in range(1):
#     print("e:",e,"  ","hidden size: ",net.hidden_size)
#     for i in range(len(train_lst)):
#         X=train_lst[i]
#         y=train_label[i]
#         o=net.feed_forward(X)
#         net.train(X,y)

x1=[i for i in range(28)]
x2=[i+28 for i in range(28)]
x3=[]
x4=[]
for j in range(14):
    x3.append(x1[2*j])
    x3.append(x1[2*j+1])
    x3.append(x2[2*j])
    x3.append(x2[2*j+1])
print(x3)
for i in range(14):
    for j in range(len(x3)):
        x4.append(x3[j]+i*56)

fg=open("test_res_2x2_1.txt",'a+')
#first
for times in range(10):
    start=time.time()
    print("test: ",times,file=fg)
    print("test: ",times)
    #start of training
    net=Neural_NetWork(196)
    lstp=[]
    for e in range(100):
        print("e:",e,"  ","hidden size: ",net.hidden_size,file=fg)
        print("e:",e,"  ","hidden size: ",net.hidden_size)
        for i in range(len(train_lst)):
            xx=train_lst[i]
            X=[0 for i2 in range(784)]
            for ii in range(784):
                X[ii]=xx[x4[ii]]
            y=train_label[i]
            o=net.feed_forward(X)
            net.train(X,y)
            net.new_error+=net.o_error

        lstp.append(net.new_error)
        print(net.new_error,file=fg)
        print(net.new_error)
        if net.old_error-net.new_error<5 and e>5 or net.new_error<100:  #after 10 epoches and change in sum of error between epoch very small
            break
        net.old_error=net.new_error
        net.new_error=0
    end=time.time()
    #draw confusion matrix

    confusion_matrix=np.array([0]*100).reshape(10,10)
    success=0
    for i in range(len(test_label)):
        xx=test_lst[i]
        X=[0 for i2 in range(784)]
        for ii in range(784):
            X[ii]=xx[x4[ii]]
        o=net.feed_forward(X)
        x=0
        y=0
        for j in range(10):
            if test_label[i][j]==1:
                x=j
                break

        for j in range(len(o)):
            if max(o)==o[j]:
                y=j
                break
        confusion_matrix[x][y]+=1
        if x==y:
            success+=1

    print(file=fg)
    print("confusion matrix",file=fg)
    print(confusion_matrix,file=fg)
    print(file=fg)
    print("time taken: ",end-start,file=fg)

    print("success: ",success,'/',len(test_label),file=fg)
    print("success rate: ",float(success/len(test_label)),file=fg)
    print(file=fg)
    print(file=fg)
    print()
    print("confusion matrix")
    print(confusion_matrix)
    print()
    print("time taken: ",end-start)
    print("success: ",success,'/',len(test_label))
    print("success rate: ",float(success/len(test_label)))
    print()
    print()