def find_duplicate(lst):
    x=len(lst)
    res=[]
    for i in range(x):
        print(lst[abs(lst[i])])
        if lst[abs(lst[i])] >= 0:
            lst[abs(lst[i])] = -lst[abs(lst[i])]
        else:
            res.append(abs(lst[i]))
    return res

def remove_all(lst,value):
    count=0
    i=0
    while i+count<len(lst):

        if lst[i]==value:
            count+=1
        lst[i]=lst[i+count]
        i+=1

        print(lst)
    print(count)
    for i in range(count):
        lst.pop()
lst=[2,3,4,5,6,2]
remove_all(lst,2)
print(lst)
