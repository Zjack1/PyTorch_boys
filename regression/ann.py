import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim
# coding=gbk
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.net=nn.Sequential(
        nn.Linear(in_features=1,out_features=100),nn.ReLU(),
        nn.Linear(100,100), nn.ReLU(),
        nn.Linear(100, 100), nn.ReLU(),
        nn.Linear(100, 100), nn.ReLU(),
        nn.Linear(100,1)
    )

  def forward(self, input:torch.FloatTensor):
    return self.net(input)


class Fcnn(nn.Module):
    def __init__(self):
        super(Fcnn,self).__init__()
        self.classifier=nn.Sequential(
            nn.Linear(in_features=1, out_features=100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self,states):
        states=self.classifier(states)
        return states

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

def creat_dataset(dataset):
    data_x = []
    data_y = []
    len_data =len(dataset)
    for i in range(len(dataset)):
        data_x.append(dataset[i])
        data_y.append(((i)/150))
    return  np.asarray(data_y) ,np.asarray(data_x)

x, y = creat_dataset(datas)
X = torch.unsqueeze(torch.linspace(-2,2,150),dim=1)
Y = np.expand_dims(datas,axis=1)
dataset=TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=150,shuffle=True)
net=Net()
optim=torch.optim.SGD(Net.parameters(net),lr=0.5)
Loss=nn.MSELoss()


for epoch in range(10000):
  loss=None
  for batch_x,batch_y in dataloader:
    y_predict=net(batch_x)
    loss=Loss(y_predict,batch_y)
    optim.zero_grad()
    loss.backward()
    optim.step()
  if (epoch+1)%100==0:
    print("step: {0} , loss: {1}".format(epoch+1,loss.item()))

predict=net(torch.tensor(X,dtype=torch.float))

import matplotlib.pyplot as plt
plt.plot(x,y,label="fact")
plt.plot(x,predict.detach().numpy(),label="predict")
plt.title("f(x) function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
