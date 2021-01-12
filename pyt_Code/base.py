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
# Source: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
# Modification: added a response variable to the dataset
class BatchManager():

    def __init__(self, X, Y):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._X = X
        self._num_examples = X.shape[0]
        self._Y = Y

    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        
        # Shuffle the data on the first call
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self._X = self.X[idx]
            self._Y = self.Y[idx]
        
        # If there aren't enough points left in this epoch to fill the minibatch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start

            # Load the remaining data
            X_rest_part = self.X[start:self._num_examples]
            Y_rest_part = self.Y[start:self._num_examples]
            
            # Reshuffle the dataset
            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self._X = self.X[idx0]
            self._Y = self.Y[idx0]
            
            # Get the remaining samples for the batch from the next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end =  self._index_in_epoch
            X_new_part = self._X[start:end]
            Y_new_part = self._Y[start:end]
            return np.concatenate((X_rest_part, X_new_part), axis=0), np.concatenate((Y_rest_part, Y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._Y[start:end]


if __name__ == "__main__":
    shape = [10,20,30]
    batch_size = 32
    mlp = MLP(shape)
    print(mlp.model)

    x = torch.rand([batch_size,shape[0]])
    print('input shape: ',x.shape)
    out = mlp(x)
    print('output shape: ',out.shape)
    