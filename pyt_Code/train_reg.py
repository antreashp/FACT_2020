
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
    

    # sys.stdout = open("train.txt", "w")
    print('blah')
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)
    print('blah')
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
    # model_loss = mse(Y,pred)
    # recon_loss = mse(X,recon)
    # loss_op = model_loss + recon_weight *recon_loss 
    print(type(encoder.parameters))
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(learner.parameters())
    optimizer = torch.optim.Adam(params,lr=learning_rate)
    criterion = torch.nn.MSELoss()

    best_epoch = 0 
    best_loss = np.inf
    epoch = 0
    total_batch = int(n / batch_size)
    while True:

        # Stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            break

        # Run a training epoch
        for i in range(total_batch):
            x_batch, y_batch = bm.next_batch(batch_size = batch_size)
            # summary, _ = sess.run([summary_op, train_op], feed_dict = {X: x_batch, Y: y_batch})
            # train_writer.add_summary(summary, epoch * total_batch + i)
            x_batch = torch.from_numpy(x_batch).float()
            y_batch = torch.from_numpy(y_batch)
            rep = encoder(x_batch)
            print('rep',rep)
            recon = decoder(rep)
            print('recon',recon)
            pred = learner(rep)
            print('pred',pred)
            # pred_from_rep = learner(R)

            model_loss = criterion(pred,y_batch)
            print('model_loss',model_loss)
            recon_loss = criterion(x_batch,recon)
            print('recon_loss',recon_loss)
            loss_op = model_loss + recon_weight * recon_loss
            print('loss_op',loss_op)
            loss_op.backward()
            optimizer.step()
    

            exit()

if __name__ == "__main__":
    batch_size = 4
    x = np.random.random((batch_size*20,100))
    y = np.random.random((batch_size*20,))
    train_reg(x,y,batch_size=batch_size)