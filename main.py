import argparse
import Train
from  Model import *

parser = argparse.ArgumentParser()


model_tag_list=[
    'Linear',
    'FullyConnected',
    'Linear_Normalized',
    'FullyConnected_Normalized',
    'FullyConnectedRegularized',
    'FullyConnectedDroped',
    'CNN',
    'CNNwithoutDr',
    'CNNnoFCAugment'
]

parser.add_argument('-model', default='Linear',
                    choices=model_tag_list,
                            dest='model',
                            help='deep neural network model')

parser.add_argument('-num_ep',default=100, type=int,
                            dest='num_ep',
                            help='number of epochs')

params = parser.parse_args()

model_class_list=[
    LinearNet,FullyConnected,LinearNet,FullyConnected,
    FullyConnectedRegularized,
    FullyConnected,
    CNN,
    CNN,
    CNNnoFC,
    CNNnoFC
]
model_pram={
    'Linear' : {'input_size':1 * 28 * 28, 'num_classes': 10},
    'FullyConnected' : {'input_size':1 * 28 * 28, 'num_classes': 10},
    'Linear_Normalized' : {'input_size':1 * 28 * 28, 'num_classes': 10},
    'FullyConnected_Normalized' : {'input_size':1 * 28 * 28, 'num_classes': 10} ,
    'FullyConnectedRegularized' : {'input_size':1 * 28 * 28, 'num_classes': 10, 'l2_reg' : 10**-3},
    'FullyConnectedDroped' : {'input_size':1 * 28 * 28, 'num_classes': 10, 'drop':(0.2,0.5)},
    'CNN' : {'inc':1, 'out':10},
    'CNNwithoutDr' : {'inc':1, 'out':10 ,'Dr':[0.,0.]} ,
    'CNNnoFCAugment' : {'inc':1, 'out':10}

}

model_name=params.model
tag="metrics/"+model_name
print('############TAG##########')
print(tag)
param=model_pram[model_name]
print('############Param##########')
print(param)
i=model_tag_list.index(model_name)
model=model_class_list[i](**param)
print('############Architecture##########')
print(model)

f_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
Train.train_nep_save(params.num_ep, tag, f_loss, optimizer, model)
