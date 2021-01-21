import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from base import  MLP, BatchManager

def train_ae(x,
          encoder_shape = [100, 100, 100, 2], decoder_shape = [2, 100, 100, 100],
          learning_rate = 0.001, batch_size = 4, min_epochs = 100, stopping_epochs = 50, tol = 0.001, freq_eval = 1,
          device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup directory
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

    # Build the models
    encoder = MLP(encoder_shape)
    decoder = MLP(decoder_shape)

    # Define the loss and optimizer
    loss_op = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([encoder.params, decoder.params], lr=learning_rate)

    # Train and evaluate the model
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
            optimizer.zero_grad()
            x_batch, y_batch = bm.next_batch(batch_size=batch_size)
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
                torch.save(encoder_shape, encoder.state_dict(), epoch, "./model_encoder.pt")
                torch.save(decoder_shape, decoder.state_dict(), epoch, "./model_decoder.pt")

            writer.add_scalar("Loss/val", val_loss, epoch)

        epoch += 1

    writer.close()

    # Evaluate the final model
    encoder_checkpoint = torch.load("./model_encoder.pt")
    encoder_best = MLP(encoder_checkpoint["hyper_parameters"]["shape"])
    encoder_best.load_state_dict(encoder_checkpoint["state_dict"].copy())

    # Find the 2d point representation
    points = np.zeros((n, 2))
    for i in range(total_batch):
        start = i * batch_size
        stop = min(n, (i + 1) * batch_size)
        x_batch = x[start:stop, :]
        points_batch = encoder_best(x_batch)
        points[start:stop, :] = points_batch
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.savefig("representation.pdf")
    plt.close()
    torch.save(points, "./points.pt")

    # Go back to directory
    os.chdir(cwd)
