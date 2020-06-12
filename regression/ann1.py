import torch
from torch.autograd import Variable
import  torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



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
datas = list(map(lambda x: x / scalar*2, datas))

x=torch.unsqueeze(torch.linspace(-1,1,150),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
y = np.expand_dims(datas,axis=1)  # 加括号



x,y=Variable(torch.tensor(x,dtype=torch.float)),Variable(torch.tensor(y,dtype=torch.float))

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_ouput):
        super(Net, self).__init__()
        self.hidden1=torch.nn.Linear(n_feature,n_hidden,n_ouput)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden, n_ouput)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden, n_ouput)
        self.predict=torch.nn.Linear(n_hidden,n_ouput)

    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x=self.predict(x)
        return x

net=Net(n_feature=1,n_hidden=100,n_ouput=1)
print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.1)
loss_func=torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(10000):
    prediction=net(x)
    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 100 == 0:
        print("step: {0} , loss: {1}".format(t , loss.item()))
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

# 使用训练好的模型进行预测
predict=net(x)
plt.figure()
plt.plot(x,y,label="fact")
plt.plot(x,predict.detach().numpy(),label="predict")
plt.title("sin function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()
