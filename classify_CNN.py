import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from VGG16 import SimpleNet,VGG16


BARCH_SIZE = 128
LR = 0.001
EPOCH = 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    normalize
])
# 从文件夹中读取训练数据
train_dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
#print(train_dataset.class_to_idx)
#print(train_dataset.imgs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BARCH_SIZE, shuffle=True)

# 从文件夹中读取validation数据
validation_dataset = torchvision.datasets.ImageFolder(root='./data/test', transform=transform)

#print(validation_dataset.class_to_idx)

test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)

alexNet = VGG16(2).to(device)
#alexNet = alexNet(pretrained=True)
#alexNet = torch.load("./alexNet.pth")
criterion = nn.CrossEntropyLoss()
opti = torch.optim.Adam(alexNet.parameters(), lr=LR)

if __name__ == '__main__':
    Accuracy_list = []
    Loss_list = []

    for epoch in range(EPOCH):
        sum_loss = 0.0
        correct1 = 0

        total1 = 0
        for i, (images, labels) in enumerate(train_loader):
            num_images = images.size(0)

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            out = alexNet(images).to(device)

            _, predicted = torch.max(out.data, 1)

            total1 += labels.size(0)

            correct1 += (predicted == labels).sum().item()

            loss = criterion(out, labels)
            #print(loss)
            opti.zero_grad()
            loss.backward()
            opti.step()

            # 每训练10个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 10 == 0:
                print("EPOCH:", epoch, " Iteration :", i," Ave loss:", sum_loss / 10, " lr:",LR)
                sum_loss = 0.0
        Accuracy_list.append(100.0 * correct1 / total1)
        print('train accurary={}'.format(100.0 * correct1 / total1))
        Loss_list.append(loss.item())
        correct = 0
        test_loss = 0.0
        test_total = 0
        test_total = 0
        alexNet.eval()
        for data in test_loader:
            images, labels = data
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            outputs = alexNet(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))

    torch.save(alexNet, './alexNet.pth')#保存整个网络结构信息和模型参数信息
    torch.save(alexNet.state_dict(), './params.pth')#保存模型训练好的参数

    x1 = range(0, EPOCH)
    x2 = range(0, EPOCH)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.savefig("accuracy_epoch" + (str)(EPOCH) + ".png")
    plt.show()
