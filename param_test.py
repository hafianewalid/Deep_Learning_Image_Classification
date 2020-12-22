import random

import parser

import Train
from Model import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def standarparam(modelname):
    if (modelname == 'FullyConnected'):
        return {'model':['FullyConnected'], 'tag':['FCtest'], 'epochs':[10], 'normalized':[True], 'drop':[[0,0]]}
    if (modelname == 'CNN'):
        return {'model':['CNN'], 'tag':['CNNtest'], 'epochs':[10], 'normalized':[True],'ker_size':[[3,2]],'srides':[[1,2]],'nb_filtres':[[64,128,256]],'Dr':[[0.5,0.5,0.5]],'FC_L_size':[[128,64]]}
    if (modelname == 'CNNnoFC'):
        return {'model':['CNNnoFC'], 'tag':['CNNnoFCtest'], 'epochs':[10], 'normalized':[True], 'drop':[[0,0]]}

def eval_model(param):
    print(param)
    if(param['model'][0]=='FullyConnected'):
        model = FullyConnected(1*28*28,10,drop=param['drop'][0])
    if(param['model'][0]=='CNN'):
        model = CNN(inc=1,out=10,ker_size=param['ker_size'][0],srides=param['srides'][0], nb_filtres=param['nb_filtres'][0],Dr=param['Dr'][0], FC_L_size=param['FC_L_size'][0])
    if (param['model'][0]=='CNNnoFC'):
        model = CNNnoFC(inc=1,out=10,ker_size=param['ker_size'][0],srides=param['srides'][0],nb_filtres=param['nb_filtres'][0])
    f_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return Train.train_nep_save(param['epochs'][0], param['tag'][0], f_loss, optimizer, model, normalized=param['normalized'][0])

def testMultiParam(params):
  list_param=[]
  list_acc=[]
  list_loss=[]
  v=[]

  for i in params:
      if(len(params[i])>1):
          list_param.append(i)

  if(len(list_param)==1):
      for p in params[list_param[0]]:
          ptest=standarparam(params['model'][0])
          ptest[list_param[0]]=[p]
          for u in params:
              if (len(params[u])==1):
                  ptest[u] = params[u]
          loss,acc=eval_model(ptest)
          list_loss.append(loss)
          list_acc.append(acc)
          v.append(p)

  if (len(list_param)==2):
      for p1 in params[list_param[0]]:
          #l1,l2=[],[]
          for p2 in params[list_param[1]]:
              ptest = standarparam(params['model'][0])
              ptest[list_param[0]]=[p1]
              ptest[list_param[1]]=[p2]
              for u in params:
                  if(len(params[u])==1):
                    ptest[u]=params[u]
              loss,acc=eval_model(ptest)
              list_loss.append(loss)
              list_acc.append(acc)
              v.append([p1,p2])

  return list_param,list_acc,list_loss,v

def zip_ind(Z,e):
    ind=-1
    cnt=0
    for i in Z :
        if i[0]==e[0] and i[1]==e[1]:
            ind=cnt
        cnt+=1
    return ind

def f(X,Y,XY,Z):
    Zz=[]
    for e in zip(X,Y):
        ind=zip_ind(XY,e)
        v=0 if ind<0 else Z[ind]
        Zz.append(v)
    return Zz


def Plotparam3D(param1,param2,v1,v2,acc,loss,tag):

    fig = plt.figure(0)
    plt.style.context("seaborn")
    ax = fig.add_subplot(111, projection='3d')
    X = np.array([str(x) for x in v1])
    Y = np.array([str(x) for x in v2])
    Z = np.array(acc)
    print(X,Y,Z)

    #ax.plot_trisurf(range(len(X)),range(len(Y)), Z, cmap=cm.coolwarm, vmin=min(Z),vmax=max(Z))
    ax.scatter(X,Y,Z, marker='^',c=Z,linewidths=4)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel('Accuracy')
    plt.savefig(tag+"acc.png")
    #plt.show()
    plt.figure(0)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    Z = np.array(loss)
    #ax.plot_trisurf(v1, v2, Z, cmap=cm.coolwarm, vmin=min(Z), vmax=max(Z))
    ax.scatter(X,Y, Z, marker='^',c=Z,linewidths=4)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel('Loss')
    plt.savefig(tag + "loss.png")
    #plt.show()
    plt.figure(1)

def Plotparam2D(param1, v, acc, loss, tag):
    plt.figure(0)
    plt.style.use("ggplot")
    X = np.array([str(x) for x in v ])
    Y = np.array(acc)
    #plt.plot(X, Y)
    plt.scatter(X,Y, marker='^',c=Y,linewidths=4)
    plt.xlabel(param1)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.savefig(tag+"acc.png")
    plt.figure(0)
    #plt.show()
    plt.figure(1)
    plt.style.use("ggplot")
    Y = np.array(loss)
    #plt.plot(X,Y)
    plt.scatter(X, Y, marker='^', c=Y,linewidths=4)
    plt.xlabel(param1)
    plt.ylabel('Loss')
    plt.xticks(rotation=90)
    plt.savefig(tag + "loss.png")
    plt.figure(1)
    #plt.show()

def test_plot(list_param,list_acc,list_loss,v,exp_name):
  if(len(list_param)==2):
         V=np.array(v)
         Plotparam3D(list_param[0],list_param[1],V[:,0],V[:,1],list_acc,list_loss,exp_name)
  else:
         Plotparam2D(list_param[0],v,list_acc,list_loss,exp_name)

def Experience(params,exp_name):
    list_param,list_acc,list_loss,v=testMultiParam(params)
    test_plot(list_param,list_acc,list_loss,v,exp_name)



drop_values=[[0,0],[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.4,0.4],[0.5,0.5],[0.6,0.6],[0.7,0.7],[0.8,0.8],[0.9,0.9],[1,1]]
params={'model':['FullyConnected'],'epochs':[10],'drop':drop_values}
Experience(params,"drop value")



params={'model':['CNN'],'epochs':[10],'ker_size':[[3,2],[5,2],[7,2],[9,2],[11,2],[13,2],[15,2]]}
Experience(params,"CNN_conv ")



Convarchi=[
[10,20,30],[30,60,120],[40,80,160],[50,100,200]
]
FCarchi=[
    [10,10],[30,10],[60,30],[80,40],[128,64],[220,100]
    ,[128,128,64],[128,64,64],[128,128,128,64],[128,64,64,64]
]



params={'model':['CNN'],'epochs':[10],'Dr':[[0.5,0.5,0.5,0.5,0.5]],
        'nb_filtres':Convarchi,'FC_L_size':FCarchi}
Experience(params,"CNN_FC ARChi")



params={'model':['CNNnoFC'],'epochs':[10],'nb_filtres':Convarchi,'ker_size':[[3,2],[5,2],[7,2],[9,2],[11,2]]}
Experience(params,"CNNnoFC_conv")



'''
randlist = lambda p1 : [random.randint(0,100)/100 for i in range(p1)]
Plotparam2D('param1',randlist(10),randlist(10),randlist(10), "tag")


randlist2D = lambda p1 : [randlist(p1) for i in range(p1)]
Plotparam3D('param1','param2',randlist(10),randlist(10),randlist(10),randlist(10),"tag3D")
'''
