# <<<<<<< HEAD
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import shutil

from base import  MLP, BatchManager
from tqdm import tqdm
def train_ae(x,
          encoder_shape = [100, 100, 100, 2], decoder_shape = [2, 100, 100, 100],
          learning_rate = 0.001, batch_size = 4, min_epochs = 100, stopping_epochs = 50, tol = 0.001, freq_eval = 1,
          device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup directory
    if os.name == "nt":  # For Windows compatibility
        shutil.rmtree('Model')
    else:
        os.system("rm -rf Model")

    cwd = os.getcwd()
    os.makedirs("Model")
    os.chdir("Model")

    sys.stdout = open("train.txt", "w")

    # Create the summary writer
    writer = SummaryWriter()

    # Split the dataset
    x_train, x_val = train_test_split(x, test_size=0.25)

    # Get sizes for future reference
    n = x_train.shape[0]
    n_input = x_train.shape[1]
    encoder_shape.insert(0, n_input)
    decoder_shape.append(n_input)

    # Batch Manager
    y_train = np.zeros((n, 1))  # Dumby variable to let use the BatchManager
    bm = BatchManager(x_train, y_train)
    x_val = torch.from_numpy(x_val).float()
    # Build the models
    encoder = MLP(encoder_shape)
    decoder = MLP(decoder_shape)

    # Define the loss and optimizer
    loss_op = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=learning_rate)

    # Train and evaluate the model
    best_epoch = 0
    best_loss = np.inf
    epoch = 0
    total_batch = int(n / batch_size)

    hyper_parameters = {'encoder_shape':encoder_shape,'decoder_shape':decoder_shape}
    while True:

        # Stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            break

        # Run a training epoch
        for i in tqdm(range(total_batch)):
            optimizer.zero_grad()
            x_batch, y_batch = bm.next_batch(batch_size=batch_size)
            x_batch = torch.from_numpy(x_batch).float()
            y_batch = torch.from_numpy(y_batch).float()
            rep = encoder(x_batch)
            recon = decoder(rep)
            recon_loss = loss_op(x_batch, recon)
            writer.add_scalar("ReconMSE", recon_loss, epoch)
            writer.add_scalar("Loss/train", recon_loss, epoch)
            recon_loss.backward()
            optimizer.step()

        # Run model metrics
        if epoch % freq_eval == 0:

            rep = encoder(x_val)
            recon = decoder(rep)
            val_loss = loss_op(x_val, recon)

            if val_loss < best_loss - tol:
                print(epoch, " ", val_loss)
                best_loss = val_loss
                best_epoch = epoch
                torch.save({'shape':encoder_shape ,'state_dict':encoder.state_dict(), 'epochs':epoch}, "./model_encoder.pt")
                torch.save({'shape':decoder_shape, 'state_dict':decoder.state_dict(), 'epochs':epoch}, "./model_decoder.pt")

            writer.add_scalar("Loss/val", val_loss, epoch)

        epoch += 1

    writer.close()

    # Evaluate the final model
    encoder_checkpoint = torch.load("./model_encoder.pt")

    encoder_best = MLP(encoder_checkpoint["shape"])
    encoder_best.load_state_dict(encoder_checkpoint["state_dict"].copy())

    # Find the 2d point representation
    points = np.zeros((n, 2))
    for i in range(total_batch):

        start = i * batch_size
        stop = min(n, (i + 1) * batch_size)
        x_batch = x[start:stop, :]
        x_batch = torch.from_numpy(x_batch).float()
        points_batch = encoder_best(x_batch)
        points[start:stop, :] = points_batch.detach()
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.savefig("representation.pdf")
    plt.close()
    torch.save(points, "./points.pt")

    # Go back to directory
    os.chdir(cwd)
# =======
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# import sys
# import torch
# from tqdm import tqdm
# from base import MLP, BatchManager
#
#
#
# def train(encoder,decoder,optimizer,criterion,bm,bm_val,batch_size,stopping_epochs,min_epochs,n,freq_eval,tol):
#     best_epoch = 0
#     best_loss = np.inf
#     epoch = 0
#     total_batch = int(n / batch_size)
#     while True:
#
#         # Stopping condition
#         if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
#             break
#
#         # Run a training epoch
#         for i in tqdm(range(total_batch)):
#             x_batch, y_batch = bm.next_batch(batch_size = batch_size)
#             # summary, _ = sess.run([summary_op, train_op], feed_dict = {X: x_batch, Y: y_batch})
#             # train_writer.add_summary(summary, epoch * total_batch + i)
#             optimizer.zero_grad()
#             x_batch = torch.from_numpy(x_batch).float()
#             y_batch = torch.from_numpy(y_batch).float()
#             rep = encoder(x_batch)
#             # print('rep',rep)
#             recon = decoder(rep)
#             # print('recon',recon)
#             # pred = learner(rep)
#             # print('pred',pred)
#             # pred_from_rep = learner(R)
#
#             # model_loss = criterion(pred,y_batch)
#             # print('model_loss',model_loss)
#             recon_loss = criterion(x_batch,recon)
#             # print('recon_loss',recon_loss)
#             loss_op = recon_loss
#             # print('loss_op',loss_op)
#             loss_op.backward()
#             optimizer.step()
#             if epoch % freq_eval == 0:
#                 val_loss = val(encoder,decoder,criterion,bm_val,batch_size,stopping_epochs,min_epochs,n)
#                 if val_loss < best_loss-tol:
#                     print(epoch, ' ',val_loss.item())
#                     best_loss = val_loss
#                     best_epoch = epoch
#                     '''TODO save model'''
#
#             epoch += 1
#         # test_loss = test(encoder,decoder,learner,criterion,bm_test,batch_size,stopping_epochs,min_epochs,n,recon_weight)
#         print('val_loss',val_loss)
#         return encoder,decoder
#
# def val(encoder,decoder,criterion,bm_val,batch_size,stopping_epochs,min_epochs,n):
#     best_epoch = 0
#     best_loss = np.inf
#     epoch = 0
#     total_batch = int(n / batch_size)
#     while True:
#
#         # Stopping condition
#         if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
#             break
#
#         # Run a training epoch
#         for i in range(total_batch):
#             x_batch, y_batch = bm_val.next_batch(batch_size = batch_size)
#             # summary, _ = sess.run([summary_op, train_op], feed_dict = {X: x_batch, Y: y_batch})
#             # train_writer.add_summary(summary, epoch * total_batch + i)
#             # optimizer.zero_grad()
#             x_batch = torch.from_numpy(x_batch).float()
#             y_batch = torch.from_numpy(y_batch).float()
#             rep = encoder(x_batch)
#             # print('rep',rep)
#             recon = decoder(rep)
#             # print('recon',recon)
#             # pred = learner(rep)
#             # print('pred',pred)
#             # pred_from_rep = learner(R)
#
#             # model_loss = criterion(pred,y_batch)
#             # print('model_loss',model_loss)
#             recon_loss = criterion(x_batch,recon)
#             # print('recon_loss',recon_loss)
#             loss_op = recon_loss#model_loss.float() + (float(recon_weight)* recon_loss.float())
#             # print('loss_op',loss_op)
#             # loss_op.backward()
#             # optimizer.step()
#             return loss_op

# def train_ae(x,
#           encoder_shape = [100, 100, 100, 2], decoder_shape = [2, 100, 100, 100],
#           learning_rate = 0.001, batch_size = 4, min_epochs = 100, stopping_epochs = 50, tol = 0.001, freq_eval = 1):
#     # encoder_shape = list(encoder_shape)
#     # Setup directory
#     os.system("rm -rf Model")
#     cwd = os.getcwd()
#     os.makedirs("Model")
#     os.chdir("Model")
#
#
#     # sys.stdout = open("train.txt", "w")
#     # print('blah')
#     # Split the dataset
#     x_train, x_val = train_test_split(x, test_size = 0.25)
#     print(encoder_shape)
#     # Get sizes for future reference
#     n = x_train.shape[0]
#     n_input = x_train.shape[1]
#     encoder_shape.insert(0, n_input)
#     decoder_shape.append(n_input)
#     # print('blah')
#     # Evaluate baseline models
#     # model_lm = LinearRegression().fit(x_train, y_train)
#     # print("\nLM MSE: ", mean_squared_error(model_lm.predict(x_test), y_test), "\n")
#
#     # model_rf = RandomForestRegressor(n_estimators = 10).fit(x_train, y_train)
#     # print("\nRF MSE: ", mean_squared_error(model_rf.predict(x_test), y_test), "\n")
#
#     # Get sizes for future reference
#     # n = x_train.shape[0]
#     # n_input = x_train.shape[1]
#     # encoder_shape.insert(0, n_input)
#     # decoder_shape.append(n_input)
#     y_train = np.zeros((n,1))
#     y_val = np.zeros((n,1))
#     # Batch Manager
#     bm = BatchManager(x_train, y_train)
#     bm_val = BatchManager(x_val,y_val)
#     # bm_test = BatchManager(x_test,y_test)
#     '''todo
#     # # Graph inputs
#     # X = tf.placeholder("float", [None, n_input], name = "X_in")
#     # R = tf.placeholder("float", [None, 2], name = "R_in")
#     # Y = tf.placeholder("float", [None, 1], name = "Y_in")
#     '''
#     encoder = MLP(encoder_shape)
#     # rep = encoder.model(X)
#
#     decoder = MLP(decoder_shape)
#     # learner = MLP(learner_shape)
#     # model_loss = mse(Y,pred)
#     # recon_loss = mse(X,recon)
#     # loss_op = model_loss + recon_weight *recon_loss
#     # print(type(encoder.parameters))
#     params = list(encoder.parameters()) + list(decoder.parameters())
#     optimizer = torch.optim.Adam(params,lr=learning_rate)
#     criterion = torch.nn.MSELoss()
#
#     encoder,decoder= train(encoder,decoder,optimizer,criterion,bm,bm_val,batch_size,stopping_epochs,min_epochs,n,freq_eval,tol)
#     points = np.zeros((n, 2))
#     total_batch = int(n / batch_size)
#
#     for i in range(total_batch):
#         start = i * batch_size
#         stop = min(n, (i + 1) * batch_size)
#         x_batch = x[start:stop, :]
#         points_batch = encoder(torch.from_numpy(x_batch).float())
#         points_batch = points_batch.detach()
#         points[start:stop, :] = points_batch
#     plt.scatter(points[:, 0], points[:, 1], s = 10)
#     plt.savefig("representation.pdf")
#     plt.show()
#     plt.close()
#     pickle.dump(points, open("points.pkl", "wb"))
#     # Go back to directory
#     os.chdir(cwd)
# if __name__ == "__main__":
#     batch_size = 4
#     x = np.random.random((batch_size*20,100))
#     y = np.random.random((batch_size*20,))
#     train_ae(x,batch_size=batch_size)
# >>>>>>> 2e66210d2e1b1b8b3bbfc03188b2e9b3a7b5f8b6
