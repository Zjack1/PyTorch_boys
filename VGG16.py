import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):

    def __init__(self,num_classes =2):
        super(VGG16, self).__init__()
        self.relu = nn.ReLU()
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
        out = self.relu(out)
        out = self.conv1_2(out)  # 222
        out = self.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = self.relu(out)
        out = self.conv2_2(out)  # 110
        out = self.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = self.relu(out)
        out = self.conv3_2(out)  # 54
        out = self.relu(out)
        out = self.conv3_3(out)  # 54
        out = self.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = self.relu(out)
        out = self.conv4_2(out)  # 26
        out = self.relu(out)
        out = self.conv4_3(out)  # 26
        out = self.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = self.relu(out)
        out = self.conv5_2(out)  # 12
        out = self.relu(out)
        out = self.conv5_3(out)  # 12
        out = self.relu(out)
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




class SimpleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(in_features=80 * 80 * 24, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool(output)

        output = self.conv3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)

        output = output.view(-1, 80 * 80 * 24)

        output = self.fc(output)

        return output


class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32 * 6 * 6, out_features=4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 32 * 6 * 6)
        x = self.classifier(x)
        return x

