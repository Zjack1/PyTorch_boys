import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VGG16 import SimpleNet,VGG16

import os

from PIL import Image, ImageDraw, ImageFont

#记住图像尺度统一为224×224时，要用transforms.Resize([224, 224]),不能写成transforms.Resize(224)，
# transforms.Resize(224)表示把图像的短边统一为224，另外一边做同样倍速缩放，不一定为224
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])
}
data = {'test': datasets.ImageFolder(root="./data/test", transform=image_transforms['test'])}
idx_to_class = {v: k for k, v in data['test'].class_to_idx.items()}
test_data_size = len(data['test'])
test_data = DataLoader(data['test'], batch_size=99, shuffle=True)


def computeTestSetAccuracy(model, loss_function):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    test_loss = 0.0

    with torch.no_grad():
        model.eval()

        for j, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            test_acc += acc.item() * inputs.size(0)

            print("Test Batch Number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(
                j, loss.item(), acc.item()
            ))
    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size

    print("Test accuracy : " + str(avg_test_acc))




def predict(model, test_image_name):
    transform = image_transforms['test']#

    test_image = Image.open(test_image_name)
    draw = ImageDraw.Draw(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()

        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        print("Prediction : ", idx_to_class[topclass.cpu().numpy()[0][0]], ", Score: ", topk.cpu().numpy()[0][0])
        text = idx_to_class[topclass.cpu().numpy()[0][0]] + " " + str(topk.cpu().numpy()[0][0])
        font = ImageFont.truetype('arial.ttf', 36)
        draw.text((0, 0), text, (255, 0, 0), font=font)
        test_image.show()



#model = torch.load('./alexNet.pth', map_location='cpu')#使用cpu去推理
model =  VGG16(2)
model.load_state_dict(torch.load('./params.pth', map_location='cpu'))#使用cpu去推理
loss_func = nn.NLLLoss()
predict(model, './data/test/drink/1_6_702.jpg')
computeTestSetAccuracy(model, loss_func)
