import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,shape):
        super(MLP, self).__init__()
        self.shape = shape
        # self.weight_init = torch.nn.init.xavier_uniform()
        # self.bias_init = 
        n = len(shape)
        layer = nn.Linear(self.shape[0], self.shape[1],bias = True)
        nn.init.xavier_uniform(layer.weight)
        self.model = [layer]#nn.ModuleList([layer])

        self.model.extend( [nn.LeakyReLU()])
    
        for i in range(1,n-2):

            layer = nn.Linear(self.shape[i], self.shape[i+1],bias = True)
            weights= nn.init.xavier_uniform(layer.weight)

            self.model.extend( [layer])
            self.model.extend( [nn.LeakyReLU()])
            
        layer = nn.Linear(self.shape[-2], self.shape[-1],bias = True)
        nn.init.xavier_uniform(layer.weight)
        # self.model = nn.ModuleList([layer])

        # weights = nn.Linear(self.shape[-2], self.shape[-1],bias = True)
        self.model.extend([layer])
        # print(self.model)
        self.model = nn.ModuleList(self.model)
        # print(self.model)
    def forward(self,x):
        for f in self.model:
            x = f(x)
        return x


if __name__ == "__main__":
    shape = [10,20,30]
    batch_size = 32
    mlp = MLP(shape)
    print(mlp.model)

    x = torch.rand([batch_size,shape[0]])
    print('input shape: ',x.shape)
    out = mlp(x)
    print('output shape: ',out.shape)
    