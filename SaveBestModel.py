import os

import torch


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.acc = None
        self.filepath = filepath
        self.model = model

    def update(self, loss,acc):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss
            self.acc=acc

    def get_best_loss(self):
        return self.min_loss,self.acc

def save_path():
    ###################################################
    # Example usage :
    # 1- create the directory "./logs" if it does not exist
    top_logdir = "./logs"
    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)

    # 2- We test the function by calling several times our function
    logdir = generate_unique_logpath(top_logdir, "linear")
    print("Logging to {}".format(logdir))
    # -> Prints out     Logging to   ./logs/linear_0
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    return logdir
