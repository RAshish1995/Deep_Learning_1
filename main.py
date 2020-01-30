#!/usr/bin/env python
# coding: utf-8

# In[89]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
import pandas as pd
#Neural Net
inp = sys.argv[2]

print('Name : Ashish Raghuvanshi')
print('SR No. 15887')
print('Dept : CSA')

data_1 = pd.read_csv(inp,sep="\t",header = None)





#data_1 = pd.read_csv(inp,sep="\t",header = None)
data = data_1
#print(len(data))
out = [0 for i in range(len(data))]
for i in range(0,(len(data_1))):
    if(data[0][i]%3==0 and data[0][i]%5 == 0):
        out[i]='fizzbuzz'
    elif(data[0][i]%3 ==0):
        out[i]='fizz'
    elif(data[0][i]%5 == 0):
        #print(i)
        out[i]= 'buzz'
    else:
        out[i]= data[0][i]
#print(out)

file1 = open("Software1.txt","w")   
# \n is placed to indicate EOL (End of Line) 
file1.write(str(out)) 
#file1.writelines(L) 
file1.close()




#data_1 = pd.read_csv('../input1.txt',sep="\t",header = None)
data = data_1[0].values
#print(data.dtype)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 200)
        #self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 4)
        
        # Define sigmoid activation and softmax output 
        #self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        


def dec_to_bin(X):
    #print('in fun')
    x_ = []

    for i in range(len(X)):
        a = []
        N = bin(X[i])
        length = len(N)-2
        #print(N,len(N))
        for j in range(10-length):
            a.append(0)
        for i in range(length):
            a.append(int(N[i+2]))
        
        x_.append(a)
        #print(a)
    #print(x_)
    return x_


net = Net()

'''
out = [0 for i in range(len(data))]
for i in range(0,(len(data_1))):
    if(data[0][i]%3==0 and data[0][i]%5 == 0):
        out[i]='fizzbuzz'
    elif(data[0][i]%3 ==0):
        out[i]='fizz'
    elif(data[0][i]%5 == 0):
        #print(i)
        out[i]= 'buzz'
    else:
        out[i]= data[0][i]
net.load_state_dict(torch.load('./model/model.pt'))
net.eval()

'''

#print(data[0])
test = dec_to_bin(data)

test = Variable(torch.Tensor(test))
test_y = net(test)

indices = []
for i in range(len(test_y)):
    _, index = torch.max(test_y[i], 0)
    indices.append(index)

final = np.array(indices)

last = []
#predictions
#[[str(i), "fizz", "buzz", "fizzbuzz"][] for i in final]
for i in range(len(final)):
    if (final[i] == 0 ):
        last.append(data[i])
    if (final[i] == 1 ):
        last.append('fizz')
    if (final[i] == 2 ):
        last.append('buzz')
    if (final[i] == 3 ):
        last.append('fizzbuzz')
        
        
def test_accuracy():
    count = 0
    count_n = 0
    for i in range(len(data)):
        #print(out[i],last[i])
        if( out[i] == last[i]):
            count+=1
            #print('here')
        else:
            count_n+=1
    #print(count,count_n)
    num = count/(count+count_n)
    return num
test_accuracy = test_accuracy()
print('test accuracy is ' ,test_accuracy)

def accuracy_(a):
    count  = 0
    count_n = 0
    for i in range(len(data)):
        #print(last[i],out[i])
        if(last[i] == a and out[i] == a):
            count+=1
        elif(out[i] == a ):
            count_n+=1
    #print(count,count_n)
    num = count/(count+count_n)
    return num


def dec_to_bin(X):
    #print('in fun')
    x_ = []

    for i in range(len(X)):
        a = []
        N = bin(X[i])
        length = len(N)-2
        #print(N,len(N))
        for j in range(10-length):
            a.append(0)
        for i in range(length):
            a.append(int(N[i+2]))
        
        x_.append(a)
        #print(a)
    #print(x_)
    return x_

file2 = open("Software2.txt","w")   
# \n is placed to indicate EOL (End of Line) 
file2.write(str(last)) 
#file1.writelines(L) 
file2.close()


acc_fizz =  accuracy_('fizz')

print('test accuracy for fizz' ,acc_fizz)

acc_buzz=  accuracy_('buzz')

print('test accuracy for buzz' ,acc_buzz)

acc_fizzbuzz =  accuracy_('fizzbuzz')

print('test accuracy for fizzbuzz' ,acc_fizzbuzz)


# In[ ]:




