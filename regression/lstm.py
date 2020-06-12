import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim
# coding=gbk
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

def creat_dataset(dataset,look_back):
    data_x = []
    data_y = []
    for i in range(len(dataset)-look_back):
        data_x.append(dataset[i:i+look_back])
        data_y.append(dataset[i+look_back])
    return np.asarray(data_x), np.asarray(data_y) #转为ndarray数据

final_file_path = './final_file1.txt'
fo_final_file_path = open(final_file_path, "r")
num = 0
line_all_box = []
counts = []
for line_all in fo_final_file_path.readlines():
    line_all = line_all.split(",")
    line_all_box.append([line_all[0], int(line_all[1])])
    counts.append(int(line_all[1]))

datas =counts

max_value = np.max(datas)

min_value = np.min(datas)
scalar = max_value - min_value
datas = list(map(lambda x: x / scalar, datas))


dataX, dataY = creat_dataset(datas,2)
#print(dataX,dataY)
train_size = int(len(dataX)*0.5)

x_train = dataX[:train_size] #训练数据
y_train = dataY[:train_size] #训练数据目标值

x_train = x_train.reshape(-1, 1, 2) #将训练数据调整成pytorch中lstm算法的输入维度
y_train = y_train.reshape(-1, 1, 1)  #将目标值调整成pytorch中lstm算法的输出维度

#将ndarray数据转换为张量，因为pytorch用的数据类型是张量

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()  # 面向对象中的继承
        self.lstm = nn.LSTM(2, 6, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.out = nn.Linear(6, 1)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

    def forward(self, x):
        x1, _ = self.lstm(x)

        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
        return out1


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(),lr = 0.02)
loss_func = nn.MSELoss()

for i in range(1000):
    var_x = Variable(x_train).type(torch.FloatTensor)
    var_y = Variable(y_train).type(torch.FloatTensor)
    out = rnn(var_x)
    loss = loss_func(out, var_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 100 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))



dataX1 = dataX.reshape(-1,1,2)
dataX2 = torch.from_numpy(dataX1)
var_dataX = Variable(dataX2).type(torch.FloatTensor)
pred = rnn(var_dataX)
pred_test = pred.view(-1).data.numpy()  #转换成一维的ndarray数据，这是预测值dataY为真实值

plt.plot(pred.view(-1).data.numpy(), 'r', label='prediction')
plt.plot(dataY, 'b', label='real')
plt.legend(loc='best')
plt.show()
