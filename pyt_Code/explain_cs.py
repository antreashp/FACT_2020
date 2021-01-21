import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

def explain(load_model, x_means, y_means, path=None,
            lambda_global=0.5, init_mode="zero",
            consecutive_steps=10, learning_rate=0.0005, clip_val=5.0, min_iters=2000, stopping_iters=2000, tol=0.0001,
            discount=0.99, verbose=False):
    num_clusters = x_means.shape[0]
    n_input = x_means.shape[1]
    n_output = y_means.shape[1]

    if path is None:
        model = load_model()  # !!! Use different load script form the original!
    else:
        model = load_model(path)  # !!! Use different load script form the original!
    writer = SummaryWriter()

    # Find the explanation
    if init_mode == "zero":
        deltas = np.zeros((num_clusters - 1, n_input))  # Row i is the explanation for "Cluster 0 to Cluster i + 1"
    elif init_mode == "mean":
        deltas = np.zeros((num_clusters - 1, n_input))
        for i in range(1, num_clusters):
            deltas[i - 1] = x_means[i, :] - x_means[0, :]

    iter = 0
    best_iter = 0
    best_loss = np.inf
    best_deltas = None
    ema = None
    while True:

        # Stopping condition
        if iter - best_iter > stopping_iters and iter > min_iters:
            break

        # Choose the initial and target cluster
        if iter % consecutive_steps == 0:
            initial, target = np.random.choice(num_clusters, 2, replace=False)

        # point and target
        p = x_means[initial]
        t = y_means[target]

        if initial == 0:
            d = deltas[target - 1]
        elif target == 0:
            d = -1.0 * deltas[initial - 1]
        else:
            d = -1.0 * deltas[initial - 1] + deltas[target - 1]

        X = torch.from_numpy(np.reshape(p, (1, n_input)))
        X = X.type(torch.float32)
        X.requires_grad = True

        T = torch.from_numpy(np.reshape(t, (1, n_output)))
        T = T.type(torch.float32)
        T.requires_grad = False

        D = torch.from_numpy(np.reshape(d, (1, n_input)))
        D = D.type(torch.float32)
        D.requires_grad = False

        rep = model.encode(X, D)  # ??? Not sure if this is the way to go?
        l_t = ((rep - T) ** 2).sum() / 2
        writer.add_scalar("loss/target", l_t, iter)

        l_g = lambda_global * torch.mean(torch.abs(D))
        writer.add_scalar("loss/global", l_g, iter)

        l = l_t + l_g
        writer.add_scalar("loss/total", l, iter)

        l.backward()

        deltas_grad = X.grad.numpy()
        deltas_grad = np.clip(np.squeeze(deltas_grad[0]), -1.0 * clip_val, clip_val)

        if iter == 0:
            ema = l
        else:
            ema = discount * ema + (1 - discount) * l

        if ema < best_loss - tol:
            best_iter = iter
            best_loss = ema
            best_deltas = deltas
            if verbose:
                print(iter, ema)

        deltas_grad = np.clip(np.squeeze(deltas_grad[0]), -1.0 * clip_val, clip_val)

        # Update the corresponding delta
        if initial == 0:
            deltas[target - 1] -= learning_rate * deltas_grad
        elif target == 0:
            deltas[initial - 1] += learning_rate * deltas_grad
        else:
            deltas[initial - 1] += learning_rate * 0.5 * deltas_grad
            deltas[target - 1] -= learning_rate * 0.5 * deltas_grad

        iter += 1

    writer.flush()

    return best_deltas


def apply(load_model, x, y, indices, c1, d_g, num_points=200):
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(2, 1, 1)
    plt.scatter(y[:, 0], y[:, 1], s=12)

    # Sample num_points in cluster c1
    indices_c1 = np.random.choice(indices[c1], num_points, replace=False)

    points_c1 = x[indices_c1]

    # Load the model
    model = load_model()
    rep_d = model.encode(points_c1)  # ???
    rep_d_g = model.encode(points_c1, d_g)  # ???
    d = np.zeros((1, x.shape[1]))

    # Plot the chosen points before perturbing them
    y_c1 = model.decode(rep_d)  # ???
    plt.scatter(y_c1[:, 0], y_c1[:, 1], marker="v", c="green", s=64)

    # Plot the chosen points after perturbing them
    y_c1 = model.decode(rep_d_g)  # ???
    plt.scatter(y_c1[:, 0], y_c1[:, 1], marker="v", c="red", s=64)

    plt.subplot(2, 1, 2)

    feature_index = np.array(range(d_g.shape[1]))
    plt.scatter(feature_index, d_g, label="Explantion - Change per Dataset Feature", marker="x")

    plt.show()

    plt.close()
