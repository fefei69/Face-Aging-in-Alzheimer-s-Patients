# coding: utf-8

#############################################
# Consistent Cumulative Logits with ResNet-34
'''
python coral.py --dataset afad \
--image_path example-images/afad/18_years__948-0.jpg \
--state_dict_path ../afad/afad-coral__seed1/best_model.pt
python coral.py --dataset afad -i C:/vscode/coral-cnn-master/single-image-prediction__w-pretrained-models/example-images/afad  -s ../afad/afad-coral__seed1/best_model.pt

C:/vscode/coral-cnn-master/single-image-prediction__w-pretrained-models/example-images/afad
'''
#############################################

from matplotlib import colors
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import cv2
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path',
                    type=str,
                    default= '../TEST-centered/53_Patrick_Swayze_0008.jpg',
                    required=False)

parser.add_argument('-s', '--state_dict_path',
                    type=str,
                    default='C:/vscode/age_estimation_forlab/TF_model_1k_epochs/best_model225.pt',  #40414347#afad-model1-CACD2000_centered 'TF_model/best_model6.pt'  cacd-coral__seed0/best_model.pt
                    required=False)

parser.add_argument('-d', '--dataset',
                    help="Options: 'afad', 'morph2', or 'cacd'.",
                    type=str,
                    default='TF',
                    required=False)


args = parser.parse_args()
#IMAGE_PATH = args.image_path
STATE_DICT_PATH = args.state_dict_path
path_ = '255'
GRAYSCALE = False
OUT_PATH = f'C:/vscode/age_estimation_forlab/Results_FM_PT/{path_}.png'
OUT_PATH_h = f'C:/vscode/age_estimation_forlab/Results_FM_PT/histo{path_}.png'
if args.dataset == 'afad':
    print("dataset: afad")
    NUM_CLASSES = 26
    ADD_CLASS = 15

elif args.dataset == 'morph2':
    NUM_CLASSES = 55
    ADD_CLASS = 16

elif args.dataset == 'cacd':
    NUM_CLASSES = 49
    ADD_CLASS = 14

elif args.dataset == 'TF':
    NUM_CLASSES = 41#8#41
    ADD_CLASS = 56#76#56

else:
    raise ValueError("args.dataset must be 'afad',"
                     " 'morph2', or 'cacd'. Got %s " % (args.dataset))




##########################
# MODEL
##########################


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

############################
### Load image
############################
pic_conut = 0
root_path = '../'
predicted_age_pt = []
predicted_age_fm = []
groundTruth_ages_pt =[]
groundTruth_ages_fm = []
err_pt_ls = []
err_fm_ls = []

#image_path_pt = "D:/side_project_data/imagefile_patient/"
image_path_fm = "C:/Users/User/Desktop/side_project_data/imagefile_family/"
#image_path = 'C:/vscode/coral-cnn-master/model-code/AFAD-Full/' 
#image_path = 'C:/vscode/coral-cnn-master/CACD2000-centered/'

#TH_pt = 'coral_data_patient_0105.csv'
#TH_fm = "cacd-coral-TL-family_correct.csv"#'coral-cacd-TL-family_correct.csv'
#TH_pt = 'C:/vscode/age_estimation_forlab/PT_sorted_age.csv'
fm = 'C:/vscode/age_estimation_forlab/cacd-coral-TL-family_correct.csv'
#df_pt = pd.read_csv(TH_pt)
df_fm = pd.read_csv(fm)
#df = pd.read_csv(TH)


ages_fm = df_fm['age'].values
image_names_fm = df_fm['filename'].values 
#image_names = df['path'].values

del df_fm
start_time = time.time()

mae_p = 0
mse_p = 0
mae_f = 0
mse_f = 0
err_pt = 0
err_fm = 0

for i in range(len(ages_fm)):
    
    groundTruth_ages_fm.append(ages_fm[i])
    picture_name = image_names_fm[i]
    pic_conut+=1
    print("name :",picture_name,"\n NO.",pic_conut)
    IMAGE_PATH = os.path.join(image_path_fm, picture_name)
    print(IMAGE_PATH)
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
    model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=DEVICE),False)
    model.eval()
    #send model weights to the GPU
    model.cuda()
    image = image.unsqueeze(0)
    with torch.set_grad_enabled(False):
        logits, probas = model(image)
        predict_levels = probas > 0.5
        predicted_label = torch.sum(predict_levels, dim=1)
        mae_f += torch.sum(torch.abs(predicted_label + ADD_CLASS - ages_fm[i]))
        mse_f += torch.sum((predicted_label + ADD_CLASS - ages_fm[i])**2)
        err_fm += torch.sum(predicted_label + ADD_CLASS - ages_fm[i])
        err_fm_ls.append(predicted_label.item() + ADD_CLASS - ages_fm[i])
        print("mae",mae_f)
        print('Predicted age in years:', predicted_label.item() + ADD_CLASS)
        print("real age :",ages_fm[i])
        
        predicted_age_fm.append(predicted_label.item() + ADD_CLASS)
        class_probabilities  = probas.tolist()
        class_probabilities = np.array(class_probabilities)
        #class_probabilities = class_probabilities[0]
print("total inference time:",(time.time()-start_time)/60,"min")
mae_fm = mae_f.float() / len(ages_fm)
mse_fm = mse_f.float() / len(ages_fm)


print("fm :mae :",mae_fm)
#print("fm :mse",mse_fm)
print("fm :rmse :",torch.sqrt(mse_fm))
#print("fm :R2 score :",1-mse_fm/ np.var(ages_fm))
#print("fm :std of ages :",round(np.std(ages_fm),3))
print("fm :error:",err_fm/len(ages_fm))

#print("fm : ground truth avg ",len(groundTruth_ages_fm),round(np.average(groundTruth_ages_fm),3))
#print("pt : ground truth avg ",len(groundTruth_ages_pt),round(np.average(groundTruth_ages_pt),3))
print("fm : predicted avg ",round(np.average(predicted_age_fm),3))
print("fm : predicted std ",round(np.std(predicted_age_fm),3))
print("fm : avg of error",round(np.average(err_fm_ls),3))
print("fm : std of error",round(np.std(err_fm_ls),3))
#print("error of family",err_fm_ls)

#print("error of patient",err_pt_ls)
kwargs = dict(alpha=0.5, bins=40)
plt.hist(err_fm_ls,**kwargs,label="family")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Age Estimaton Error")
plt.legend()
plt.savefig(OUT_PATH_h)
plt.show()
#plt.hist(err_fm_ls,bins=20)
# plt.xlabel("Error")
# plt.ylabel("Frequency")
# plt.title("Age Estimaton Error of Family")
# plt.show()
#linear regression
# par = np.polyfit(groundTruth_ages_fm, predicted_age_fm,2)
# poly = np.poly1d(par)
# par1 = np.polyfit(groundTruth_ages_pt, predicted_age_pt,2)
# poly1 = np.poly1d(par1)
# xp = np.linspace(56, 79, 100)
# plt.plot(xp,poly(xp),color = "g")
# plt.plot(xp,poly1(xp),color = "r")
plt.scatter(groundTruth_ages_fm, predicted_age_fm , s = 5, c="b" , marker = 'o', label ='family')
plt.scatter(groundTruth_ages_pt, predicted_age_pt , s = 5, c="r" , marker = 'o', label ='patient')


#plt.scatter(groundTruth_ages, predicted_age_fm , s = 5, c="r" , marker = 'o', label ='x-predicted y-real')
#plt.scatter(predicted_age, groundTruth_ages , s = 5, marker = 'o', label ='x-predicted y-real')
# plt.pyplot.scatter(predicted_age, groundTruth_ages, s=20, c='b', marker='o', cmap=None, norm=None,
#                           vmin=None, vmax=None, alpha=None, linewidths=None,
#                           faceted=True, verts=None, hold=None, **kwargs)
plt.plot(range(100),range(100),label = 'predicted = real',color = 'black')
plt.xlabel('real age ')
plt.ylabel('predicted age')
plt.legend()
plt.savefig(OUT_PATH)
plt.show()