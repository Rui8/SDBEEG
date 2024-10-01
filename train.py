import numpy as np
import random
import torch
import os
import torch.nn as nn
from model import DCN, soft_beta_loss
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
def objective1(trial,X_train,y_train,step,sub,config):
    su=sub-1
    x_sub = np.arange(su)
    random.seed(1)
    random.shuffle(x_sub)
    flag = False
    for ep in range(su):
        if not flag:
            X_train1 = np.expand_dims(X_train[x_sub[ep]],axis=0)
            y_train1 = np.expand_dims(y_train[x_sub[ep]],axis=0)
            flag = True
        else:
            X_train1 = np.concatenate((X_train1,np.expand_dims(X_train[x_sub[ep]],axis=0)))#打乱顺序后的训练集和测试集
            y_train1 = np.concatenate((y_train1,np.expand_dims(y_train[x_sub[ep]],axis=0)))
    del X_train,y_train
    acc = objective(trial, step,X_train1,y_train1, config)#迭代ee次后三折交叉验证最好的结果


    return acc

def get_optimizer(trial,net,config):
    weight_decay = trial.suggest_loguniform('weight_decay', config.weight_decay_p.min, config.weight_decay_p.max)
    lr = trial.suggest_loguniform('lr', config.lr_p.min, config.lr_p.max)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay,
                                 amsgrad=True)
    return optimizer

class MyDataset1(Dataset):
    def __init__(self,train_data,train_label):
        self.data = train_data
        self.label = train_label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def objective(trial, step1, X_train1, y_train1, config):
    re = 0
    train_data = X_train1
    train_label = y_train1
    batch_size = 64
    train_data = train_data.transpose(1,0,2,3,4)
    train_label = train_label.transpose(1,0)

    dataset = MyDataset1(train_data,train_label)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = DCN('elu').cuda()
    ema_model=DCN('elu').cuda()
    for param in ema_model.parameters():
        param.detach()
    global_step = 0
    optimizer1 = get_optimizer(trial, model, config)
    criterion = nn.CrossEntropyLoss().cuda()
    for ep in range(config.epochh):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            model.train()
            ema_model.train()
            fflage = False
            for s in range(x_batch.shape[1]):

                x_batch1, y_batch1 = Variable(x_batch[:,s,:,:,:]).cuda(), Variable(y_batch[:,s]).cuda()
                x_batch2 = x_batch1.type(torch.cuda.FloatTensor)
                optimizer1.zero_grad()
                y_pred = model(x_batch2).cuda()
                with torch.no_grad():
                    outputs_orig = ema_model(x_batch2).cuda()
                if not fflage:
                    y = y_pred
                    ys = outputs_orig
                    y_batch2 = y_batch1
                    fflage = True
                else:
                    y = torch.cat((y,y_pred),0)
                    ys = torch.cat((ys,outputs_orig),0)
                    y_batch2 = torch.cat((y_batch2,y_batch1),0)
            outputs_orig = ys
            outputs = y
            labels_var = torch.Tensor.long(y_batch2.squeeze())
            beta = 3.0
            alpha = 0.1
            beta_loss = soft_beta_loss(outputs, labels_var, beta, outputs_orig, num_classes=2)
            loss = (1-alpha)*criterion(outputs, labels_var) + alpha*beta_loss
            loss.backward()
            optimizer1.step()
            global_step += 1
            alpha_now = min(1 - 1 / (global_step + 1), 0.999)
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(alpha_now).add_(1 - alpha_now, param.data)
    model_save_path = os.path.join(config.results_save_path, 'Deepmodel', 'models')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    saveName = os.path.join(model_save_path, 'Deepmodel'+str(step1)+".pth")
    torch.save(model.state_dict(), saveName)
    model.load_state_dict(torch.load(saveName))
    return re