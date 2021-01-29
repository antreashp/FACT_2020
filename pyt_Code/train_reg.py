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
from tqdm import tqdm
from base import MLP, BatchManager



def train(encoder,decoder,learner,optimizer,criterion,bm,bm_val,bm_test,batch_size,stopping_epochs,min_epochs,n,recon_weight,freq_eval,tol):
    best_epoch = 0 
    best_loss = np.inf
    epoch = 0
    total_batch = int(n / batch_size)
    while True:

        # Stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            break

        # Run a training epoch
        for i in tqdm(range(total_batch)):
            x_batch, y_batch = bm.next_batch(batch_size = batch_size)
            optimizer.zero_grad()
            x_batch = torch.from_numpy(x_batch).float()
            y_batch = torch.from_numpy(y_batch).float()
            rep = encoder(x_batch)

            recon = decoder(rep)
            pred = learner(rep)


            model_loss = criterion(pred,y_batch)

            recon_loss = criterion(x_batch,recon)

            loss_op = model_loss.float() + (float(recon_weight)* recon_loss.float())

            loss_op.backward()
            optimizer.step()
            if epoch % freq_eval == 0:
                val_loss = val(encoder,decoder,learner,optimizer,criterion,bm_val,batch_size,stopping_epochs,min_epochs,n,recon_weight)
                if val_loss < best_loss-tol:
                    print(epoch, ' ',val_loss.item())
                    best_loss = val_loss
                    best_epoch = epoch
                    torch.save(encoder.state_dict(), "./model_encoder_reg.pt")

            epoch += 1
        test_loss = test(encoder,decoder,learner,optimizer,criterion,bm_test,batch_size,stopping_epochs,min_epochs,n,recon_weight)
        print('test_loss',test_loss)
        return encoder,decoder,learner

def val(encoder,decoder,learner,optimizer,criterion,bm_val,batch_size,stopping_epochs,min_epochs,n,recon_weight):
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
            x_batch, y_batch = bm_val.next_batch(batch_size = batch_size)
            x_batch = torch.from_numpy(x_batch).float()
            y_batch = torch.from_numpy(y_batch).float()
            rep = encoder(x_batch)
            recon = decoder(rep)
            pred = learner(rep)

            model_loss = criterion(pred,y_batch)
            recon_loss = criterion(x_batch,recon)
            loss_op = model_loss.float() + (float(recon_weight)* recon_loss.float())
            return loss_op

def test(encoder,decoder,learner,optimizer,criterion,bm_test,batch_size,stopping_epochs,min_epochs,n,recon_weight):
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
            x_batch, y_batch = bm_test.next_batch(batch_size = batch_size)
            x_batch = torch.from_numpy(x_batch).float()
            y_batch = torch.from_numpy(y_batch).float()
            rep = encoder(x_batch)
            recon = decoder(rep)
            pred = learner(rep)
 

            model_loss = criterion(pred,y_batch)
            recon_loss = criterion(x_batch,recon)
            loss_op = model_loss.float() + (float(recon_weight)* recon_loss.float())

            loss_op.backward()
            optimizer.step()

            return loss_op

def train_reg(x, y,
          encoder_shape = [100, 100, 100, 2], decoder_shape = [2, 100, 100, 100], learner_shape = [2, 200, 200, 200, 1], recon_weight = 5,
          learning_rate = 0.001, batch_size = 4, min_epochs = 100, stopping_epochs = 50, tol = 0.001, freq_eval = 1):
   
    # Setup directory
    os.system("rm -rf Model")
    cwd = os.getcwd()
    os.makedirs("Model")
    os.chdir("Model")
    

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5)

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
    bm_val = BatchManager(x_val,y_val)
    bm_test = BatchManager(x_test,y_test)

    encoder = MLP(encoder_shape)


    decoder = MLP(decoder_shape)
    learner = MLP(learner_shape)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(learner.parameters())
    optimizer = torch.optim.Adam(params,lr=learning_rate)
    criterion = torch.nn.MSELoss()

    encoder,decoder,learner = train(encoder,decoder,learner,optimizer,criterion,bm,bm_val,bm_test,batch_size,stopping_epochs,min_epochs,n,recon_weight,freq_eval,tol)
    x = np.vstack((x_train, x_val, x_test))
    points = encoder(torch.from_numpy(x).float())
    print(points)
    points = points.detach().numpy()
    plt.scatter(points[:, 0], points[:, 1], s = 10)

    # Plot the function over that space
    min_0 = np.min(points[:, 0])
    max_0 = np.max(points[:, 0])
    min_1 = np.min(points[:, 1])
    max_1 = np.max(points[:, 1])

    feature_0 = np.linspace(min_0, max_0, 50)
    feature_1 = np.linspace(min_1, max_1, 50)
    r = np.zeros((1, 2))
    map = np.empty((50, 50))
    for i in range(50):
        r[0, 1] = feature_1[i]
        for j in range(50):
            r[0, 0] = feature_0[j]
            map[i, j] = learner(torch.from_numpy(r).float())  #sess.run(pred_from_rep, {R: r})

    plt.contour(feature_0, feature_1, map)
    plt.colorbar()

    plt.savefig("learned_function.pdf")
    plt.show()
    plt.close()
    
    pickle.dump(map, open("map.pkl", "wb"))
    pickle.dump(feature_0, open("f0.pkl", "wb"))
    pickle.dump(feature_1, open("f1.pkl", "wb"))

    pickle.dump(points, open("points.pkl", "wb"))

    # Go back to directory
    os.chdir(cwd)
if __name__ == "__main__":
    batch_size = 4
    x = np.random.random((batch_size*20,100))
    y = np.random.random((batch_size*20,))
    train_reg(x,y,batch_size=batch_size)