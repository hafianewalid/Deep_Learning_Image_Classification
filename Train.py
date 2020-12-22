import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import os.path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torch.utils.tensorboard import SummaryWriter
import SaveBestModel
import Load



def train(model, loader, f_loss, optimizer, device ,penalty):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        #print('inputs',inputs.shape)
        #print('targets',targets)
        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        #print('out',outputs)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()

        if penalty :
            model.penalty().backward()

        optimizer.step()

        # We accumulate the exact number of processed samples
        N += inputs.shape[0]

        # We accumulate the loss considering
        # The multipliation by inputs.shape[0] is due to the fact
        # that our loss criterion is averaging over its samples
        tot_loss += inputs.shape[0] * loss.item()

        # For the accuracy, we compute the labels for each input image
        # Be carefull, the model is outputing scores and not the probabilities
        # But given the softmax is not altering the rank of its input scores
        # we can compute the label by argmaxing directly the scores
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()

    return tot_loss / N, correct / N



def test(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for i, (inputs, targets) in enumerate(loader):
            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
        return tot_loss / N, correct / N




def train_nep_save(epochs,tag,f_loss,optimizer,model,normalized=False,penalty=False,Augment=False):

    train_loader, valid_loader, test_loader = Load.loadFashionMNIST(normaliz=normalized,augment_Tronsform=Augment)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    path = SaveBestModel.save_path()
    model_checkpoint = SaveBestModel.ModelCheckpoint(path + "/best_model.pt", model)

    tensorboard_writer = SummaryWriter(log_dir=path)

    for t in range(epochs):
        print("Epoch {}".format(t))
        train_loss,train_acc=train(model, train_loader, f_loss, optimizer, device,penalty)

        val_loss, val_acc = test(model, valid_loader, f_loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))

        model_checkpoint.update(val_loss,val_acc)

        '''
        
        tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
        tensorboard_writer.add_scalar('metrics/train_acc', train_acc, t)
        tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
        tensorboard_writer.add_scalar('metrics/val_acc', val_acc, t)
        
        '''

        tensorboard_writer.add_scalars(tag+'loss', {'metrics/train_loss': train_loss,'metrics/val_loss': val_loss}, t)
        tensorboard_writer.add_scalars(tag+'acc', {'metrics/train_acc': train_acc, 'metrics/val_acc': val_acc}, t)

    tensorboard_writer.close()
    return model_checkpoint.get_best_loss()

