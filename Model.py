from torch.utils.tensorboard import SummaryWriter
import SaveBestModel
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import Load

def get_layer(type_l='linear', inputs=28, outputs=10, drop=0.5, in_c=1, out_c=10,ker_size=[5,2], srides=[1, 2]):

    if(type_l== 'linear'):
        return [nn.Linear(inputs, outputs)]

    if(type_l == 'linear_relu'):
        return [nn.Linear(inputs, outputs),
                nn.ReLU(inplace=True)]

    if (type_l == 'linear_dropout'):
        return [nn.Dropout(drop),
                nn.Linear(inputs, outputs)]

    if (type_l == 'linear_dropout_relu'):
        return [nn.Dropout(drop),
                nn.Linear(inputs,outputs),
                nn.ReLU(inplace=True)]

    if (type_l== 'conv'):
        return [
            nn.Conv2d(in_c, out_c,
                      kernel_size=ker_size[0],
                      stride=srides[0],
                      padding=int((ker_size[0] - 1) / 2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=ker_size[1], stride=srides[1])
        ]
    if (type_l== 'conv_bn_relu'):
        return [nn.Conv2d(in_c, out_c,
                          kernel_size=ker_size[0],
                          stride=1,
                          padding=int((ker_size[0]-1)/2), bias=True),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)]

    if (type_l == 'conv_bn_bloc'):
        bloc=[]
        bloc.extend(get_layer(type_l='conv_bn_relu', inputs=inputs, outputs=outputs, drop=drop,
                         in_c=in_c, out_c=out_c,ker_size=ker_size, srides=srides))
        bloc.extend(get_layer(type_l='conv_bn_relu', inputs=inputs, outputs=outputs, drop=drop,
                          in_c=out_c, out_c=out_c, ker_size=ker_size, srides=srides))
        bloc.append(nn.MaxPool2d(kernel_size=ker_size[1], stride=srides[1]))

        return bloc


def out_conv_size(Conv_layers,input_size,input_c=1):
    x = torch.zeros((1,input_c,input_size,input_size))
    for l in Conv_layers:
        x = l(x)
    x = x.view(x.size()[0], -1)
    return x.size()[1]

class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        y = self.classifier(x)
        return y

def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

class FullyConnected(nn.Module):

    def __init__(self, input_size, num_classes,drop=[]):
        super(FullyConnected, self).__init__()
        self.classifier =  nn.Sequential(
            *linear_relu(input_size, 256),
            *linear_relu(256, 256),
            nn.Linear(256, num_classes)
        )
        if(drop!=[]):
            self.classifier = nn.Sequential(
                nn.Dropout(drop[0]),
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(drop[1]),
                nn.Linear(256,num_classes)
            )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

class FullyConnectedRegularized(nn.Module):

    def __init__(self, input_size, num_classes, l2_reg):
        super(FullyConnectedRegularized, self).__init__()
        self.l2_reg = l2_reg
        self.lin1 = nn.Linear(input_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_classes)


    def penalty(self):
        return self.l2_reg * (self.lin1.weight.norm(2) + self.lin2.weight.norm(2) + self.lin3.weight.norm(2))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        y = self.lin3(x)
        return y

class CNN(nn.Module):
    def __init__(self, inc=1, input_size=28, out=10,ker_size=[5, 2],
                 srides=[1,2], nb_filtres=[16,32,64]
                 , Dr=[0.5,0.5,0.5], FC_L_size=[128,64]):

        super(CNN, self).__init__()

        self.Conv_layers=[]
        self.conv_classifier=get_layer(type_l='conv',in_c=inc,out_c=nb_filtres[0],ker_size=ker_size,srides=srides)
        self.Conv_layers.extend(self.conv_classifier)
        for i in range(1,len(nb_filtres)):
            self.conv_classifier = get_layer(type_l='conv', in_c=nb_filtres[i-1], out_c=nb_filtres[i],ker_size=ker_size,srides=srides)
            self.Conv_layers.extend(self.conv_classifier)
        self.Conv_classifer=nn.Sequential(*self.Conv_layers)

        fc_input=out_conv_size(Conv_layers=self.Conv_layers,input_size=input_size,input_c=inc)
        #print(" size of FC input vector {} ".format(fc_input))

        self.FC_layers = []
        self.FC = get_layer(type_l='linear_dropout_relu', inputs=fc_input, outputs=FC_L_size[0], drop=Dr[0])
        self.FC_layers.extend(self.FC)
        for i in range(1, len(FC_L_size)):
            self.FC = get_layer(type_l='linear_dropout_relu',inputs=FC_L_size[i - 1],outputs=FC_L_size[i], drop=Dr[i])
            self.FC_layers.extend(self.FC)
        self.FC =nn.Linear(FC_L_size[-1], out)
        self.FC_layers.append(self.FC)
        self.FC_classifier= nn.Sequential(*self.FC_layers)


    def forward(self, x):

        #for l in self.Conv_layers:
        #    x = l(x)
        x=self.Conv_classifer(x)

        x=x.view(x.size()[0],-1)

        #for c in self.FC_layers:
        #    x=c(x)

        x=self.FC_classifier(x)

        return x

class CNNnoFC(nn.Module):
    def __init__(self,inc=1, input_size=28, out=10,ker_size=[3, 2],
                 srides=[1,2], nb_filtres=[64,128,256]):
        super(CNNnoFC, self).__init__()

        mapsize = input_size
        self.Conv_layers = []
        self.conv_classifier = get_layer(type_l='conv_bn_bloc', in_c=inc, out_c=nb_filtres[0], ker_size=ker_size, srides=srides)
        self.Conv_layers.extend(self.conv_classifier)
        mapsize = divmod(mapsize, ker_size[1])[0]

        for i in range(1,len(nb_filtres)-1):
            self.conv_classifier = get_layer(type_l='conv_bn_bloc', in_c=nb_filtres[i - 1],out_c=nb_filtres[i],
                                             ker_size=ker_size, srides=srides)
            self.Conv_layers.extend(self.conv_classifier)
            mapsize = divmod(mapsize, ker_size[1])[0]

        self.conv_classifier=get_layer(type_l='conv_bn_relu',in_c=nb_filtres[-2],out_c=nb_filtres[-1],ker_size=ker_size,srides=srides)
        self.Conv_layers.extend(self.conv_classifier)
        self.conv_classifier = get_layer(type_l='conv_bn_relu',in_c=nb_filtres[-1],out_c=nb_filtres[-1],ker_size=ker_size,srides=srides)
        self.Conv_layers.extend(self.conv_classifier)
        self.Conv_layers.append(nn.AvgPool2d(kernel_size=mapsize))
        self.Conv_classifer = nn.Sequential(*self.Conv_layers)

        self.Dense_layer = get_layer(type_l='linear_dropout', inputs=nb_filtres[-1], outputs=out, drop=0.5)
        self.Dense_layer =nn.Sequential(*self.Dense_layer)


    def forward(self, x):

        x = self.Conv_classifer(x)
        x = x.view(x.size()[0], -1)
        x = self.Dense_layer(x)

        return x
