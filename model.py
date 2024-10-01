import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
def choose_act_func(act_name):
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    else:
        raise TypeError('activation_function type not defined.')

class DCN(nn.Module):
    def __init__(self, act_func):
        super(DCN, self).__init__()
        self.block1 = nn.Sequential()
        self.block1.add_module('conv1', nn.Conv2d(1, 25, (1, 4), bias=False))
        chann = 62
        self.block1.add_module('conv2', nn.Conv2d(25, 25, (chann, 1), bias=False))
        self.block1.add_module('norm1', nn.BatchNorm2d(25))
        self.block1.add_module('act1', choose_act_func(act_func))
        # [B, 25, 2, 750] -> [B, 25, 1, 373]
        self.block1.add_module('pool1', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block1.add_module('drop1', nn.Dropout(p=0.25))

        self.block2 = nn.Sequential()
        self.block2.add_module('conv3', nn.Conv2d(25, 50, (1, 4), bias=False))
        self.block2.add_module('norm2', nn.BatchNorm2d(50))
        self.block2.add_module('act2', choose_act_func(act_func))
        self.block2.add_module('pool2', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block2.add_module('drop2', nn.Dropout(p=0.25))

        self.block3 = nn.Sequential()
        self.block3.add_module('conv4', nn.Conv2d(50, 100, (1, 4), bias=False))
        self.block3.add_module('norm3', nn.BatchNorm2d(100))
        self.block3.add_module('act3', choose_act_func(act_func))
        self.block3.add_module('pool3', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block3.add_module('drop3', nn.Dropout(p=0.25))

        self.block4 = nn.Sequential()
        self.block4.add_module('conv5', nn.Conv2d(100, 200, (1, 4), bias=False))
        self.block4.add_module('norm4', nn.BatchNorm2d(200))
        self.block4.add_module('act4', choose_act_func(act_func))
        self.block4.add_module('pool4', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block4.add_module('drop4', nn.Dropout(p=0.25))

        self.classify = nn.Sequential(
            nn.Linear(400,2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        res = x.view(x.size(0), -1)  # [B, 200, 1, 43] -> [B, 200 * 43]
        out = self.classify(res)
        return out

def soft_beta_loss(outputs, labels, beta, outputs_orig, num_classes=10):
    softmaxes = F.softmax(outputs, dim=1)
    n, num_classes = softmaxes.shape
    tensor_labels = Variable(torch.zeros(n, num_classes).cuda().scatter_(1, labels.long().view(-1, 1).data, 1))
    softmaxes_orig = F.softmax(outputs_orig, dim=1)
    maximum, _ = (softmaxes_orig*tensor_labels).max(dim=1)
    maxes, indices = maximum.sort()

    sorted_softmax, sorted_labels = softmaxes[indices], tensor_labels[indices]
    random_beta = np.random.beta(beta, 1, n)
    random_beta.sort()
    random_beta = torch.from_numpy(random_beta).cuda()

    uniform = (1 - random_beta) / (num_classes - 1)
    random_beta -= uniform
    random_beta = random_beta.view(-1, 1).repeat(1, num_classes).float()
    beta_label = sorted_labels*random_beta
    beta_label += uniform.view(-1, 1).repeat(1, num_classes).float()

    loss = -beta_label * torch.log(sorted_softmax)
    loss = loss.sum() / n
    return loss





