import torch
import torch.nn as nn
import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from scvis import scvis
from load_scvis import load_model
from vae_utils import compute_transition_probability, log_likelihood_student, tsne_repel, Unbuffered
from make_checkpoint import make_checkpoint
from base import  BatchManager

def train_scvis(x, train_params, optim_params=None, model_params=None, pretrained_path=None):
    if pretrained_path is None:
        dof = None
        if model_params is None:
            raise TypeError("Please specify the required model hyperparameters when not using a pretrained model.")

        if type(model_params) == list:
            vae = scvis(model_params[0], model_params[1], model_params[2], model_params[3], model_params[4],
                        model_params[5], model_params[6])
        elif type(model_params) == dict:
            vae = scvis(model_params["encoder_shape"], model_params["decoder_shape"], model_params["activate_op"],
                        model_params["eps"], model_params["max_sigma_square"], model_params["prob"],
                        model_params["initial"])
        else:
            raise TypeError("The model hyperparameters should be a `dict`, or a `list`.")
    else:
        vae, model_params = load_model(pretrained_path + "/scvis.pt", get_hparams=True)  # No need to specify the
                                                                                         # hyperparameters as they are
                                                                                         # stored with the model
        dof = torch.load(pretrained_path + "dof.pt")

    # Split the dataset
    x_train_np, x_val_np = train_test_split(x, test_size=0.25)

    # Get sizes for future reference
    n = x_train_np.shape[0]
    n_input = x_train_np.shape[1]
    z_dim = vae.hidden_dim

    # Get training parameters
    if type(train_params) == list:
        batch_size = train_params[0]
        stopping_epochs = train_params[1]
        min_epochs = train_params[2]
        freq_eval = train_params[3]
        perplexity = train_params[4]
        clip_norm = train_params[5]
        clip_value = train_params[6]
        tol = train_params[7]
        store_path = train_params[8]
        device = train_params[9]
    elif type(train_params) == dict:
        batch_size = train_params["batch_size"]
        stopping_epochs = train_params["stopping_epochs"]
        min_epochs = train_params["min_epochs"]
        freq_eval = train_params["freq_eval"]
        perplexity = train_params["perplexity"]
        clip_norm = train_params["clip_norm"]
        clip_value = train_params["clip_value"]
        tol = train_params["tol"]
        store_path = train_params["store_path"]
        device = train_params["device"]
    else:
        raise TypeError("The train hyperparameters should be a `dict`, or a `list`.")

    # Setting up the proper device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae.to(device)

    # Define trainable parameters
    dof = torch.ones(n_input, requires_grad=True, device=device) if dof is None else dof

    # Setup directory
    if os.name == "nt":  # For Windows compatibility
        try:
            shutil.rmtree(store_path)
        except FileNotFoundError:
            pass
    else:
        try:
            os.system(f"rm -rf {store_path}")
        except FileNotFoundError:
            pass

    cwd = os.getcwd()
    os.makedirs(store_path)
    os.chdir(store_path)

    sys.stdout = open("train.txt", "w")
    sys.stdout = Unbuffered(sys.stdout)

    print(f"Running on {device}.")

    # Create the summary writer
    writer = SummaryWriter()

    # Batch Manager
    y_train_np = np.zeros((n, 1))  # Dumby variable to let use the BatchManager
    bm = BatchManager(x_train_np, y_train_np)
    x_val = torch.from_numpy(x_val_np).float().to(device)

    # Define the loss and optimizer
    if optim_params is None:
        optimizer = torch.optim.Adam(list(vae.parameters()) + [dof], lr=0.01, eps=1e-20)
    elif type(model_params) == list:
        optimizer = torch.optim.Adam(list(vae.parameters()) + [dof], lr=optim_params[0], betas=optim_params[1],
                                     eps=optim_params[2], weight_decay=optim_params[3])
    elif type(model_params) == dict:
        optimizer = torch.optim.Adam(list(vae.parameters()) + [dof], lr=optim_params["lr"], betas=optim_params["betas"],
                                     eps=optim_params["eps"], weight_decay=optim_params["l2_norm"])
    else:
        raise TypeError("The optimizer hyperparameters should be a `dict`, a `list`, or `None`.")

    # Train and evaluate the model
    best_epoch = 0
    best_loss = np.inf
    epoch = 0
    total_batch = int(n / batch_size)
    iter = 0

    while True:

        # Stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            break

        # Run a training epoch
        vae.train()
        for i in tqdm(range(total_batch)):
            iter += 1

            optimizer.zero_grad()
            x_batch_np, y_batch_np = bm.next_batch(batch_size=batch_size)
            x_batch = torch.from_numpy(x_batch_np).float().to(device)
            y_batch = torch.from_numpy(y_batch_np).float().to(device)

            encoder_mu, encoder_sigma_square = vae.encode(x_batch)
            encoder_sigma = torch.sqrt(encoder_sigma_square)

            noise = torch.normal(torch.zeros_like(encoder_mu), torch.ones_like(encoder_sigma)).to(device)
            z_batch = encoder_sigma * noise + encoder_mu

            decoder_mu, decoder_sigma_square = vae.decode(z_batch)

            dof_clipped = torch.clamp(dof, 0.1, 10)
            p = compute_transition_probability(x_batch_np, perplexity=perplexity)
            weight = torch.from_numpy(np.clip(np.sum(p, 0), 0.01, 2.0)).to(device)  # Note that this is not a parameter

            log_likelihood = torch.mean(log_likelihood_student(x_batch, decoder_mu, decoder_sigma_square, dof_clipped)
                                        * weight)

            kl_divergence = torch.mean(0.5 * torch.sum(encoder_mu**2 + encoder_sigma_square
                                                       - torch.log(encoder_sigma_square) - 1, dim=1))
            kl_divergence *= 1.0 if n_input/iter > 1.0 else n_input/iter

            elbo = log_likelihood - kl_divergence

            kl_pq = tsne_repel(z_dim, z_batch, n, torch.from_numpy(p).to(device))
            kl_pq *= n_input if iter > n_input else n_input

            obj = kl_pq - elbo  # The regularization is added via the `weight_decay` parameter in the optimizer
            obj.backward()

            normal_clip = [dof]
            small_clip = []

            for name, param in vae.named_parameters():
                if "encoder.sigma_layer" in name:
                    small_clip.append(param)
                else:
                    normal_clip.append(param)

            torch.nn.utils.clip_grad_norm_(normal_clip + small_clip, max_norm=clip_norm)  # Global norm clip
            torch.nn.utils.clip_grad_value_(normal_clip, clip_value=clip_value)           # Value clip
            torch.nn.utils.clip_grad_value_(small_clip, clip_value=0.1*clip_value)        # Small value clip

            optimizer.step()

        # Run model metrics
        if epoch % freq_eval == 0:
            vae.eval()
            with torch.no_grad():
                encoder_mu, encoder_sigma_square = vae.encode(x_val)
                encoder_sigma = torch.sqrt(encoder_sigma_square)

                noise = torch.normal(torch.zeros_like(encoder_mu), torch.ones_like(encoder_sigma)).to(device)
                z_batch = encoder_sigma * noise + encoder_mu

                decoder_mu, decoder_sigma_square = vae.decode(z_batch)

                dof_clipped = torch.clamp(dof, 0.1, 10)
                p = compute_transition_probability(x_val_np, perplexity=perplexity)
                weight = torch.from_numpy(np.clip(np.sum(p, 0), 0.01, 2.0)).to(device)  # Note that this is not a parameter

                log_likelihood = torch.mean(
                    log_likelihood_student(x_val, decoder_mu, decoder_sigma_square, dof_clipped)
                    * weight)

                kl_divergence = torch.mean(0.5 * torch.sum(encoder_mu ** 2 + encoder_sigma_square
                                                           - torch.log(encoder_sigma_square) - 1, dim=1))
                kl_divergence *= 1.0 if n_input/iter < 1.0 else n_input/iter

                elbo = log_likelihood - kl_divergence

                kl_pq = tsne_repel(z_dim, z_batch, n, torch.from_numpy(p).to(device))
                kl_pq *= n_input if iter > n_input else n_input

                obj = kl_pq - elbo
                val_loss = obj

                if val_loss < best_loss - tol:
                    print(epoch, " ", val_loss)
                    best_loss = val_loss
                    best_epoch = epoch
                    make_checkpoint(model_params, vae.state_dict, "scvis.pt", epoch=epoch)
                    torch.save(dof, "dof.pt")

                writer.add_scalar("Loss/val", val_loss, epoch)

        epoch += 1

    writer.close()

    # Evaluate the final model
    best_vae = load_model("scvis.pt")

    # Find the 2d point representation
    points = np.zeros((n, 2))
    for i in range(total_batch):
        start = i * batch_size
        stop = min(n, (i + 1) * batch_size)
        x_batch = x[start:stop, :]
        x_batch = torch.from_numpy(x_batch).float()
        points_batch = best_vae(x_batch)
        points[start:stop, :] = points_batch.detach()
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.savefig("representation.pdf")
    plt.close()
    torch.save(points, store_path + "/points.pt")

    # Go back to directory
    os.chdir(cwd)

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--encoder_shape', type=str, default='128,64,32',
                        help='A comma-separated string of the encoder shape.')
    parser.add_argument('--decoder_shape', type=str, default='32,32,32,64,128',
                        help='A comma-separated string of the decoder shape.')
    parser.add_argument('--activate_op', type=str, default='ELU', choices=['ELU'],
                        help='!UNUSED! The activation operation for the encoder')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='The in-model minimum sigma clamp parameter.')
    parser.add_argument('--max_sigma_square', type=float, default=1e10,
                        help='The in-model maximum sigma clamp parameter.')
    parser.add_argument('--prob', type=float, default=0.5,
                        help='The training dropout probability.')
    parser.add_argument('--initial', type=type(None), default=None, choices=[None],
                        help='!UNUSED! The intial layer of the vae.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='The batch size.')
    parser.add_argument('--stopping_epochs', type=int, default=50,
                        help='The number of epoch after which training stops if there is no improvement.')
    parser.add_argument('--min_epochs', type=int, default=100,
                        help='The minimum number of epoch to train the model on.')
    parser.add_argument('--freq_eval', type=int, default=1,
                        help='The model will evaluate at `epoch % freq_eval == 0`.')
    parser.add_argument('--perplexity', type=float, default=10.0,
                        help='The perplexity for the transition probability.')
    parser.add_argument('--clip_norm', type=float, default=10.0,
                        help='The maximum norm for norm gradient clipping.')
    parser.add_argument('--clip_value', type=float, default=3.0,
                        help='The maximum value for value gradient clipping.')
    parser.add_argument('--tol', type=float, default=0.001,
                        help='The minimum loss improvement required to accept a new best model.')
    parser.add_argument('--store_path', type=str, default='./Model',
                        help='The path to the directory to store the model in.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='The device to run the model on.')

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam'],
                        help='!UNUSED! The optimizer used for training.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The learning rate.')
    parser.add_argument('--betas', type=str, default='0.9,0.999',
                        help='A comma-separated string with the two beta optimization parameters.')
    parser.add_argument('--epsilon', type=float, default=0.001,
                        help='The epsilon optimization parameter.')
    parser.add_argument('--l2', type=float, default=0.001,
                        help='The l2 norm given for the weight decay parameter.')

    # Miscellaneous parameters
    parser.add_argument('--data_path', type=str,
                        help='The path to the data .pt file to train the model on.')
    parser.add_argument('--load_path', type=str,
                        help='The path to the directory to load a partially trained model from.')

    config = parser.parse_args()

    def int_css_to_list(css):
        return [int(x) for x in css.split(',')]
    def float_css_to_list(css):
        return [float(x) for x in css.split(',')]

    if config.activate_op == "ELU":
        m = nn.ELU()

    model_params = [int_css_to_list(config.encoder_shape), int_css_to_list(config.decoder_shape), m, config.eps,
                    config.max_sigma_square, config.prob, None]

    train_params = [config.batch_size, config.stopping_epochs, config.min_epochs, config.freq_eval,
                    config.perplexity, config.clip_norm, config.clip_value, config.tol, config.store_path,
                    config.device]

    optim_params = [config.lr, float_css_to_list(config.betas), config.epsilon, config.l2, config.optimizer]

    if config.data_path is None:
        x = np.zeros((400, int_css_to_list(config.encoder_shape)[0]))
        for i in range(400):
            if np.random.uniform() < 0.5:
                x[i, 0] = 1.0 + np.random.normal(loc=0.0, scale=0.2)

            if np.random.uniform() < 0.5:
                x[i, 1] = 1.0 + np.random.normal(loc=0.0, scale=0.2)

            x[i, 2] = np.random.normal(loc=0.0, scale=0.5)
            x[i, 3] = x[i, 0] + np.random.normal(loc=0.0, scale=0.05)
    else:
        x = torch.load(config.data_path)  # X is expected to be a numpy array with the data


    if config.load_path is None:
        load_path = None
    else:
        load_path = config.load_path

    train_scvis(x=x, train_params=train_params, optim_params=optim_params, model_params=model_params,
                pretrained_path=load_path)
