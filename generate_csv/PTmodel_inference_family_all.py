import pandas as pd 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import pandas as pd
import time

from torchvision import transforms 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_path = "C:/Users/User/Desktop/side_project_data/imagefile_family/"
ROOT = 'C:/vscode/age_estimation_forlab/'
NUM_CLASSES = 8
ADD_CLASS = 76
GRAYSCALE = False
ALL = []
mod = []
groundTruth_ages = []
pred = []
err = []
df = pd.read_csv("C:/vscode/age_estimation_forlab/606analysis_ptmodel_regression.csv")
csv_filename = df["csv"].values
Models = df["model"].values
#df1 = pd.read_csv("C:/vscode/age_estimation_forlab/PT_sorted_age.csv")
df1 = pd.read_csv("C:/vscode/age_estimation_forlab/cacd-coral-TL-family_correct.csv")
fname = df1["filename"].values
ages = df1['age'].values

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model
for i,name in enumerate(csv_filename) :
    MAE = []
    RMSE = []
    #best model names
    md = os.path.join(ROOT,f'TF_model_1k_epochs/best_model{Models[i]}.pt')
    #different CSVs
    # path = os.path.join(ROOT,f"TF_csv_new/{name}")
    # print(path)
    print("model:",Models[i])
    mae = 0
    mse = 0
    ct = 0
    #df1 = pd.read_csv(f"{path}")
    #filenames in each CSV
    # fname = df1["filename"].values
    # ages = df1['age'].values
    for j in range(len(ages)):
        #groundTruth_ages.append(ages[i])
        picture_name = fname[j]
        IMAGE_PATH = os.path.join(image_path, picture_name)
        image = Image.open(IMAGE_PATH)

        custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                            transforms.CenterCrop((120, 120)),
                                            transforms.ToTensor()])
        image = custom_transform(image)
        DEVICE = torch.device('cuda:0')
        image = image.to(DEVICE)
        #######################
        ### Initialize Model
        #######################   
        model = resnet34(NUM_CLASSES, GRAYSCALE)
        #model = nn.DataParallel(model)
        model.load_state_dict(torch.load(md, map_location=DEVICE),False)
        model.eval()
        #send model weights to the GPU
        model.cuda()
        image = image.unsqueeze(0)
        if ages[j] > 0: #above70 use 70-52='18' 71-52=19
            ct+=1
            with torch.set_grad_enabled(False):
                logits, probas = model(image) #probas : 模型預測該照片為各個class的機率
                predict_levels = probas > 0.5
                #print(predict_levels)
                predicted_label = torch.sum(predict_levels, dim=1)
                mae += torch.sum(torch.abs(predicted_label + ADD_CLASS - ages[j]))
                mse += torch.sum((predicted_label + ADD_CLASS - ages[j])**2)
                #print(type(predicted_label.item() + ADD_CLASS - ages[i]))
                groundTruth_ages.append(ages[j])
                pred.append(predicted_label.item() + ADD_CLASS)
                err.append(predicted_label.item() + ADD_CLASS - ages[j])
                mod.append(Models[i])
                ALL.append(fname[j])
    if mae != 0:
        Mae = mae.float() / len(groundTruth_ages)
        RMse = torch.sqrt(mse.float() / len(groundTruth_ages))

    for n in range(ct): # n: files' names(jpg) 
        #mod.append(Models[i])
        #ALL.append(n)
        MAE.append(round(Mae.item(),3))
        RMSE.append(round(RMse.item(),3))
        #print(n)





d = {'filename':ALL,"model":mod,"error":err,"predicted":pred,"groundTruth":groundTruth_ages,"MAE":MAE,"RMSE":RMSE}
fl = pd.DataFrame(data=d)
fl.to_csv('C:/vscode/generate_csv/606/Regression_fm_all.csv')

# kwargs = dict(alpha=0.5, bins=40)
# plt.hist(err,**kwargs,label="patient")
# #plt.hist(err_fm_ls,**kwargs,label="family")
# plt.xlabel("Error")
# plt.ylabel("Frequency")
# plt.title("Age Estimaton Error")
# plt.legend()
# plt.savefig("patient_testset_under70_histo.png")
# plt.show()

# plt.scatter(groundTruth_ages, pred , s = 5, c="b" , marker = 'o', label ='patient')
# #plt.scatter(groundTruth_ages_pt, predicted_age_pt , s = 5, c="r" , marker = 'o', label ='patient')
# plt.plot(range(100),range(100),label = 'predicted = real',color = 'black')
# plt.xlabel('real age ')
# plt.ylabel('predicted age')
# plt.legend()
# plt.savefig("patient_testset_under70.png")
# plt.show()