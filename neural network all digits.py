import struct
import gzip
import numpy as np
import time


lr=0.5   #learning rate
print("hahaha")
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
        self.Weight_1=np.random.uniform(-2,2,(self.input_size,self.hidden_size))
        self.Weight_2=np.random.uniform(-2,2,(self.hidden_size,self.output_size))


    def feed_forward(self,X):

        self.z=np.dot(X,self.Weight_1)+bias  #sum of Weight and output
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
        self.d_Et_w=X.repeat(self.hidden_size).reshape(784,self.hidden_size)*yy.reshape((1,self.hidden_size))
        self.Weight_1-=lr*self.d_Et_w


    def train(self,X,y):            #forward and back once/train once
        o=self.feed_forward(X)
        self.back_propagation(X,y,o)

#obtain train images
f = gzip.open('train-images-idx3-ubyte.gz','r')
train_lst=np.array([])

for i in range(2):
    buf = f.read(8)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

buf = f.read(28*28*60001)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
train_lst=np.append(train_lst,labels)
train_lst=train_lst.reshape(int(len(train_lst)/(28*28)),28*28)/255


#obtain train labels
f = gzip.open('train-labels-idx1-ubyte.gz','r')
train_labeler=np.array([])

for i in range(1):
    buf = f.read(8)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
for i in range(60000):
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    train_labeler=np.append(train_labeler,labels)
train_label=[]
for j in range(len(train_labeler)):
    x=[0]*10
    x[int(train_labeler[j])]=1
    train_label.append(x)
np.save('train_label',np.array(train_label))
#obtain test images
f = gzip.open('t10k-images-idx3-ubyte.gz','r')
test_lst=np.array([])

for i in range(2):
    buf = f.read(8)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
buf = f.read(28*28*10000)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
test_lst=np.append(test_lst,labels)
test_lst=test_lst.reshape(int(len(test_lst)/(28*28)),28*28)/255


#obtain test label
f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
test_labeler=np.array([])

for i in range(1):
    buf = f.read(8)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
for i in range(10000):
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    test_labeler=np.append(test_labeler,labels)
test_label=[]
for j in range(len(test_labeler)):
    x=[0]*10
    x[int(test_labeler[j])]=1
    test_label.append(x)
np.save('text_label',np.array(test_label))


fg=open("test results all con.txt",'a+')
for times in range(10):
    start = time.time()
    print("test: ",times,file=fg)
    print("test: ",times)
    #start of training
    net=Neural_NetWork(49)
    lstp=[]
    for e in range(100):
        print("e:",e,"  ","hidden size: ",net.hidden_size,file=fg)
        print("e:",e,"  ","hidden size: ",net.hidden_size)
        for i in range(len(train_lst)):
            X=train_lst[i]
            y=train_label[i]
            o=net.feed_forward(X)
            net.train(X,y)
            net.new_error+=net.o_error
        lstp.append(net.new_error)
        print(net.new_error,file=fg)
        print(net.new_error)
        if net.old_error-net.new_error<5 and e>10 or net.new_error<1000:  #after 10 epoches and change in sum of error between epoch very small
            break
        net.old_error=net.new_error
        net.new_error=0
    end = time.time()
    #draw confusion matrix
    confusion_matrix=np.array([0]*100).reshape(10,10)
    success=0
    for i in range(len(test_label)):

        o=net.feed_forward(test_lst[i])
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
    print("time taken: ", end - start, file=fg)
    print("success: ",success,'/',len(test_label),file=fg)
    print("success rate: ",float(success/len(test_label)),file=fg)
    print(file=fg)
    print(file=fg)
    print()
    print("confusion matrix")
    print(confusion_matrix)
    print()
    print("time taken: ", end - start)
    print("success: ",success,'/',len(test_label))
    print("success rate: ",float(success/len(test_label)))
    print()
    print()

for times in range(10):
    start = time.time()
    print("test: ",times,file=fg)
    print("test: ",times)
    #start of training
    net=Neural_NetWork(196)
    lstp=[]
    for e in range(100):
        print("e:",e,"  ","hidden size: ",net.hidden_size,file=fg)
        print("e:",e,"  ","hidden size: ",net.hidden_size)
        for i in range(len(train_lst)):
            X=train_lst[i]
            y=train_label[i]
            o=net.feed_forward(X)
            net.train(X,y)
            net.new_error+=net.o_error
        lstp.append(net.new_error)
        print(net.new_error,file=fg)
        print(net.new_error)
        if net.old_error-net.new_error<5 and e>10 or net.new_error<1000:  #after 10 epoches and change in sum of error between epoch very small
            break
        net.old_error=net.new_error
        net.new_error=0
    end = time.time()
    #draw confusion matrix
    confusion_matrix=np.array([0]*100).reshape(10,10)
    success=0
    for i in range(len(test_label)):

        o=net.feed_forward(test_lst[i])
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
    print("time taken: ", end - start, file=fg)
    print("success: ",success,'/',len(test_label),file=fg)
    print("success rate: ",float(success/len(test_label)),file=fg)
    print(file=fg)
    print(file=fg)
    print()
    print("confusion matrix")
    print(confusion_matrix)
    print()
    print("time taken: ", end - start)
    print("success: ",success,'/',len(test_label))
    print("success rate: ",float(success/len(test_label)))
    print()
    print()

# for times in range(10):
#     print("test: ",times,file=fg)
#     print("test: ",times)
#     #start of training
#     net=Neural_NetWork(196)
#     lstp=[]
#     for e in range(100):
#         print("e:",e,"  ","hidden size: ",net.hidden_size,file=fg)
#         print("e:",e,"  ","hidden size: ",net.hidden_size)
#         for i in range(len(train_lst)):
#             X=train_lst[i]
#             y=train_label[i]
#             o=net.feed_forward(X)
#             net.train(X,y)
#             net.new_error+=net.o_error
#         lstp.append(net.new_error)
#         print(net.new_error,file=fg)
#         print(net.new_error)
#         if net.old_error-net.new_error<5 and e>10 or net.new_error<1000:  #after 10 epoches and change in sum of error between epoch very small
#             break
#         net.old_error=net.new_error
#         net.new_error=0
#
#     #draw confusion matrix
#     confusion_matrix=np.array([0]*100).reshape(10,10)
#     success=0
#     for i in range(len(test_label)):
#
#         o=net.feed_forward(test_lst[i])
#         x=0
#         y=0
#         for j in range(10):
#             if test_label[i][j]==1:
#                 x=j
#                 break
#
#         for j in range(len(o)):
#             if max(o)==o[j]:
#                 y=j
#                 break
#         confusion_matrix[x][y]+=1
#         if x==y:
#             success+=1
#
#     print(file=fg)
#     print("confusion matrix",file=fg)
#     print(confusion_matrix,file=fg)
#     print(file=fg)
#     print("success: ",success,'/',len(test_label),file=fg)
#     print("success rate: ",float(success/len(test_label)),file=fg)
#     print(file=fg)
#     print(file=fg)
#     print()
#     print("confusion matrix")
#     print(confusion_matrix)
#     print()
#     print("success: ",success,'/',len(test_label))
#     print("success rate: ",float(success/len(test_label)))
#     print()
#     print()