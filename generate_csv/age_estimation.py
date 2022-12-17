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
                    default='C:/vscode/age_estimation_forlab/TF_model/best_model101.pt',  #afad-model1-CACD2000_centered 'TF_model/best_model6.pt'  cacd-coral__seed0/best_model.pt
                    required=False)

parser.add_argument('-d', '--dataset',
                    help="Options: 'afad', 'morph2', or 'cacd'.",
                    type=str,
                    default='TF',
                    required=False)

args = parser.parse_args()
IMAGE_PATH = args.image_path
STATE_DICT_PATH = args.state_dict_path
GRAYSCALE = False
#OUT_PATH = f'C:/vscode/age_estimation_forlab/Results_train_valid_test/{STATE_DICT_PATH[9:22]}.png'
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
    NUM_CLASSES = 42
    ADD_CLASS = 52

else:
    raise ValueError("args.dataset must be 'afad',"
                     " 'morph2', or 'cacd'. Got %s " % (args.dataset))


############################
### Load image
############################


# image = Image.open(IMAGE_PATH)
# custom_transform = transforms.Compose([transforms.Resize((128, 128)),
#                                        transforms.CenterCrop((120, 120)),
#                                        transforms.ToTensor()])
# image = custom_transform(image)
# DEVICE = torch.device('cpu')
# image = image.to(DEVICE)


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

#orig_path = os.path.join(root_path, 'Test-centered/')
#TEST_CSV_PATH = '../datasets/cacd_test.csv'
#TEST_CSV_PATH = 'test_csv.csv'
#TEST_CSV_PATH = 'C:/vscode/age_estimation_forlab/coral_data_family.csv'
#TEST_CSV_PATH = 'C:/vscode/age_estimation_forlab/coral_data_pt.csv'
train_list = []
valid_list = []
test_list = []
groundTruth_ages_train =[]
groundTruth_ages_valid =[]
groundTruth_ages_test =[]
err_train = []
err_valid = []
err_test = []
nm = []
for j in range(3):
    if j == 0:
        image_path = "D:/side_project_data/imagefile_family/" #training
        #csv_path = 'coral-cacd-TL-family-trainingset.csv'
        csv_path = 'C:/vscode/age_estimation_forlab/TF_csv_new/coral-cacd-TL-family-trainingset07.csv'
        #csv_path = 'C:/vscode/age_estimation_forlab/TF_csv/new_family_training02-1.csv'
    elif j == 1:
        image_path = "D:/side_project_data/imagefile_family/"  #validation
        #csv_path = 'coral-cacd-TL-family-validationset.csv'
        csv_path = 'C:/vscode/age_estimation_forlab/TF_csv_new/coral-cacd-TL-family-validationset07.csv'
        #csv_path = 'C:/vscode/age_estimation_forlab/TF_csv/new_family_validation02-1.csv'
    else:
        image_path = "D:/side_project_data/imagefile_family/"  #testing
        #csv_path = 'coral-cacd-TL-family-testset.csv'
        csv_path = 'C:/vscode/age_estimation_forlab/TF_csv_new/coral-cacd-TL-family-testset07.csv'
        #csv_path = 'C:/vscode/age_estimation_forlab/TF_csv/new_family_testing02-1.csv'
    
    
    df = pd.read_csv(csv_path)
    
    #df = pd.read_csv(TH)
    ages = df['ageinyears'].values
    image_names = df['filename'].values 
    del df
    if j == 0:
        length_tr = len(ages)
    elif j == 1:
        length_vl = len(ages)
    else:
        length_ts = len(ages)

    start_time = time.time()
    mae = 0
    mse = 0
    err_pt = 0
    err_fm = 0
    #image = Image.open(IMAGE_PATH)
    #for picture_name in os.listdir(orig_path):
    for i in range(len(ages)):
        if j == 0:
            groundTruth_ages_train.append(ages[i])
        elif j == 1:
            groundTruth_ages_valid.append(ages[i])
        else:
            groundTruth_ages_test.append(ages[i])
            nm.append(image_names[i])

        picture_name = image_names[i]
        pic_conut+=1
        print("name :",picture_name,"\n NO.",pic_conut)
        IMAGE_PATH = os.path.join(image_path, picture_name)
        print(IMAGE_PATH)
        # stream = open(path, "rb")
        # bytes = bytearray(stream.read())
        # numpyarray = np.asarray(bytes, dtype=np.uint8)
        # image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
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
            #print(predicted_label)
            #print('Class probabilities:', probas)
            #print(len(probas))
            #print('Predicted class label:', predicted_label.item())
            mae += torch.sum(torch.abs(predicted_label + ADD_CLASS - ages[i]))
            mse += torch.sum((predicted_label + ADD_CLASS - ages[i])**2)
            err_pt += torch.sum(predicted_label + ADD_CLASS - ages[i])
            #print(type(predicted_label.item() + ADD_CLASS - ages[i]))
            err_add = predicted_label.item() + ADD_CLASS - ages[i]
            #print(type(err_add))
            
            print("mae",mae)
            print('Predicted age in years:', predicted_label.item() + ADD_CLASS)
            print("real age :",ages[i])
            #predicted_age_pt.append(predicted_label.item() + ADD_CLASS)

            if j == 0:
                train_list.append(predicted_label.item() + ADD_CLASS)
                mae_train = mae
                mse_train = mse
                err_train.append(err_add)
            elif j == 1:
                valid_list.append(predicted_label.item() + ADD_CLASS)
                mae_valid = mae
                mse_valid = mse
                err_valid.append(err_add)
            else :
                test_list.append(predicted_label.item() + ADD_CLASS)
                mae_test = mae
                mse_test = mse
                err_test.append(err_add)

            class_probabilities  = probas.tolist()
            class_probabilities = np.array(class_probabilities)
            #class_probabilities = np.reshape(class_probabilities, [len(class_probabilities),54])
            #print(class_probabilities[0])
            class_probabilities = class_probabilities[0]


# print("predicted : ",predicted_age)
# print("real age  :",groundTruth_ages)
print("total inference time:",(time.time()-start_time)/60,"min")
print("train:")
# print(mae_train.float() / length_tr)
# print(mse_train.float() / length_tr)
# print(round(np.average(err_train),3))
# print(round(np.std(err_train),3),"\n")

# print("valid:")
# print(mae_valid.float() / length_vl)
# print(mse_valid.float() / length_vl)
# print(round(np.average(err_valid),3))
# print(round(np.std(err_valid),3),"\n")

#print("test:")
#print(mae_test.float() / length_vl)
#print(mse_test.float() / length_vl)
# print(round(np.average(err_test),3))
# print(round(np.std(err_test),3),"\n")
# print(err_test)
d = {'filename':nm , 'error':err_test}
fl = pd.DataFrame(data=d)
fl.to_csv('test03.csv')
# plt.scatter(groundTruth_ages_train, train_list , s = 5, c="b" , marker = 'o', label ='train set')
# plt.scatter(groundTruth_ages_valid, valid_list , s = 5, c="r" , marker = 'o', label ='valid set')
# plt.scatter(groundTruth_ages_test, test_list , s = 5, c="g" , marker = 'o', label ='test set')
# #plt.scatter(groundTruth_ages_pt, predicted_age_pt , s = 5, c="r" , marker = 'o', label ='patient')
# plt.plot(range(100),range(100),label = 'predicted = real',color = 'black')
# plt.xlabel('real age ')
# plt.ylabel('predicted age')
# plt.legend()
# plt.savefig(OUT_PATH)
# plt.show()