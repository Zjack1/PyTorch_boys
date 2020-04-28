import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from VGG16 import SimpleNet,VGG16


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class VGG16(nn.Module):

    def __init__(self,num_classes =2):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 8, 3)  # 64 * 224 * 224
        self.conv1_2 = nn.Conv2d(8, 8, 3, padding=(1, 1))  # 64 * 224* 224
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(8, 16, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))  # 128 * 112 * 112
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(16, 32, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))  # 256 * 56 * 56
        self.conv3_3 = nn.Conv2d(32, 32, 3, padding=(1, 1))  # 256 * 56 * 56
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(32, 32, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))  # 512 * 28 * 28
        self.conv4_3 = nn.Conv2d(32, 32, 3, padding=(1, 1))  # 512 * 28 * 28
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(32, 32, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))  # 512 * 14 * 14
        self.conv5_3 = nn.Conv2d(32, 32, 3, padding=(1, 1))  # 512 * 14 * 14
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(32 * 7 * 7, out_features=num_classes)
        #self.fc2 = nn.Linear(4096, 4096)
        #self.fc3 = nn.Linear(4096, out_features=num_classes)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        #out = F.relu(out)
        #out = self.fc2(out)
        #out = F.relu(out)
        #out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out



BARCH_SIZE = 128
LR = 0.001
EPOCH = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
alexNet = torch.load("./alexNet.pth")
criterion = nn.CrossEntropyLoss()
opti = torch.optim.Adam(alexNet.parameters(), lr=LR)
lr_init = opti.param_groups[0]['lr']
if __name__ == '__main__':
    Accuracy_list = []
    Loss_list = []

    for epoch in range(EPOCH):
        sum_loss = 0.0
        correct1 = 0
        adjust_learning_rate(opti, epoch, lr_init)
        lr = opti.param_groups[0]['lr']
        total1 = 0
        for i, (images, labels) in enumerate(train_loader):
            num_images = images.size(0)

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            out = alexNet(images)

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
                print("EPOCH:", epoch, " Iteration :", i," Ave loss:", sum_loss / 10, " lr:",lr)
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

    torch.save(alexNet, './alexNet.pth')

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
