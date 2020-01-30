#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import sys

inp = sys.argv[2]

print('Name : Ashish Raghuvanshi')
print('SR No. 15887')
print('Dept : CSA')

data_1 = pd.read_csv(inp,sep="\t",header = None)
data = data_1
#print(len(data))
out = [0 for i in range(100)]

for i in range(1,101):
    if(i%3==0 and i%5 == 0):
        out[i-1]='fizzbuzz'
    elif(i%3 ==0):
        out[i-1]='fizz'
    elif(i%5 == 0):
        #print(i)
        out[i-1]= 'buzz'
    else:
        out[i-1]= i
#print(out)

file1 = open("software1.txt","w")   
# \n is placed to indicate EOL (End of Line) 
file1.write(str(out)) 
#file1.writelines(L) 
file1.close()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

#Neural Net
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
        

net = Net()
#print(net)

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


#binary encoding

num = [i+101 for i in range(900)]

#x_mat = [[0 for i in range(10)] for j in range (900)]
#'10'.zfill(10)
#print(len(x_mat))
x_mat = dec_to_bin(num)
#print(len(x_mat))

#print(num)
label = []#0 for i in range(900)]
for i in range(101, 1001):
    #print(i)
    if(((i)%15 == 0 )):
        label.append(3)
    elif((i)%3 == 0 ):
        label.append(1)
    elif((i)%5 == 0 ):
        
        label.append(2)
    else:
        label.append(0)
#print(len(label))

#print(x_mat[0])
X = Variable(torch.Tensor(x_mat))
#print(X)
Y = Variable(torch.LongTensor(label))
#print(Y.dtype)
#Y = torch.from_numpy(Y).long()

criterion = torch.nn.CrossEntropyLoss()
#labels = torch.as_tensor([0,1,2,3])
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

batch_size = 32


# Start training it
for epoch in range(1000):
    for start in range(0, len(x_mat), batch_size):
        #print(end)
        end = start + batch_size
        batchX = X[start:end]
        batchY = Y[start:end]

        y_pred = net(batchX)
        #print(batchY.shape)
        #print(y_pred.shape)
        loss = criterion(y_pred, batchY)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print(loss)
    #a = model(trX)
    # Find loss on training data
    #loss = loss_fn(a, trY).data[0]
    #print('Epoch:', epoch, 'Loss:', loss)

print('Training accuracy is ',1-loss)

torch.save(net.state_dict(), 'model_new.pt')


# In[ ]:





# In[ ]:




