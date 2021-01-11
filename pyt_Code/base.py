import numpy as np
import torch
import torch.nn as nn

class MLP():
    def __init__(self,shape):
        self.shape = shape
        self.weight_init = torch.nn.init.xavier_uniform()
        # self.bias_init = 
        n = len(shape)
        weights = nn.init.xavier_uniform(nn.Linear(self.shape[0], self.shape[1],bias = True))
        self.model = nn.ModuleList([weights])

    
        for i in range(n-2):
            weights= nn.init.xavier_uniform(nn.Linear(self.shape[i], self.shape[i]+1),bias = True)

            self.model.extend( weights)
        weights = nn.Linear(self.shape[-2], self.shape[-1],bias = True)
        self.model.append(weights)
    def forward(self,x):
        x = self.model(x)