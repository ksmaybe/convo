import numpy as np
import copy
def sig(x):
    return 1/(1+np.exp(-x))
lst=[i for i in range(28*28)]
lst1=copy.deepcopy(lst)
cc1=[]
cc2=[]
cc3=[]
cc4=[]
x=0
for i in range(7):
    for j in range(4):
        cc1.append(x)
        x+=1
    for j in range(4):
        cc2.append(x)
        x+=1
    for j in range(4):
        cc3.append(x)
        x+=1
    for j in range(4):
        cc4.append(x)
        x+=1

kk=0
for i in range(7):
    for j in range(len(cc1)):
        lst1[kk]=lst[112*i+cc1[j]]
        kk+=1
    for j in range(len(cc2)):
        lst1[kk]=lst[112*i+cc2[j]]
        kk+=1
    for j in range(len(cc2)):
        lst1[kk]=lst[112*i+cc3[j]]
        kk+=1
    for j in range(len(cc2)):
        lst1[kk]=lst[112*i+cc4[j]]
        kk+=1
print(np.array(lst1).reshape((28,28)))
#
# X=[0 for i in range(784)]
# for i in range(784):
#     X[i]=lst[lst1[i]]
# print(X)