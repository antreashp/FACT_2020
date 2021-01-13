
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys
import torch

from base import MLP, BatchManager


def train_reg(x, y,
          encoder_shape = [100, 100, 100, 2], decoder_shape = [2, 100, 100, 100], learner_shape = [2, 200, 200, 200, 1], recon_weight = 5,
          learning_rate = 0.001, batch_size = 4, min_epochs = 100, stopping_epochs = 50, tol = 0.001, freq_eval = 1):
   
   
    # Setup directory
    os.system("rm -rf Model")
    cwd = os.getcwd()
    os.makedirs("Model")
    os.chdir("Model")
    
    sys.stdout = open("train.txt", "w")

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)
    
    # Evaluate baseline models
    model_lm = LinearRegression().fit(x_train, y_train)
    print("\nLM MSE: ", mean_squared_error(model_lm.predict(x_test), y_test), "\n")

    model_rf = RandomForestRegressor(n_estimators = 10).fit(x_train, y_train)
    print("\nRF MSE: ", mean_squared_error(model_rf.predict(x_test), y_test), "\n")
   
    # Get sizes for future reference
    n = x_train.shape[0]
    n_input = x_train.shape[1]
    encoder_shape.insert(0, n_input)
    decoder_shape.append(n_input)
    
    # Batch Manager
    bm = BatchManager(x_train, y_train)
    '''todo
    # # Graph inputs
    # X = tf.placeholder("float", [None, n_input], name = "X_in")
    # R = tf.placeholder("float", [None, 2], name = "R_in")
    # Y = tf.placeholder("float", [None, 1], name = "Y_in")
    '''
    encoder = MLP(encoder_shape)
    # rep = encoder.model(X)

    decoder = MLP(decoder_shape)
    learner = MLP(learner_shape)
    params = encoder.parameters + decoder.parameters + learner.parameters
    optimizer = torch.optim.Adam(params,lr=learning_rate)
    criterion = nn.MSELoss()


