# !!!
# This contains untranslated (since there is no tensorflow code) copies of https://github.com/shahcompbio/scvis/blob/af73849679e8f84d80e4418d0badf5ea9d9a87af/lib/scvis/tsne_helper.py
# !!!

import sys
import numpy as np
import torch

MAX_VAL = np.log(sys.float_info.max) / 2.0

np.random.seed(0)


def compute_transition_probability(x, perplexity=30.0,
                                   tol=1e-4, max_iter=50, verbose=False):
    # x should be properly scaled so the distances are not either too small or too large

    if verbose:
        print('tSNE: searching for sigma ...')

    (n, d) = x.shape
    sum_x = np.sum(np.square(x), 1)

    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    p = np.zeros((n, n))

    # Parameterized by precision
    beta = np.ones((n, 1))
    entropy = np.log(perplexity) / np.log(2)

    # Binary search for sigma_i
    idx = range(n)
    for i in range(n):
        idx_i = list(idx[:i]) + list(idx[i+1:n])

        beta_min = -np.inf
        beta_max = np.inf

        # Remove d_ii
        dist_i = dist[i, idx_i]
        h_i, p_i = compute_entropy(dist_i, beta[i])
        h_diff = h_i - entropy

        iter_i = 0
        while np.abs(h_diff) > tol and iter_i < max_iter:
            if h_diff > 0:
                beta_min = beta[i].copy()
                if np.isfinite(beta_max):
                    beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta[i] *= 2.0
            else:
                beta_max = beta[i].copy()
                if np.isfinite(beta_min):
                    beta[i] = (beta[i] + beta_min) / 2.0
                else:
                    beta[i] /= 2.0

            h_i, p_i = compute_entropy(dist_i, beta[i])
            h_diff = h_i - entropy

            iter_i += 1

        p[i, idx_i] = p_i

    if verbose:
        print('Min of sigma square: {}'.format(np.min(1 / beta)))
        print('Max of sigma square: {}'.format(np.max(1 / beta)))
        print('Mean of sigma square: {}'.format(np.mean(1 / beta)))

    return p


def compute_entropy(dist=np.array([]), beta=1.0):
    p = -dist * beta
    shift = MAX_VAL - max(p)
    p = np.exp(p + shift)
    sum_p = np.sum(p)

    h = np.log(sum_p) - shift + beta * np.sum(np.multiply(dist, p)) / sum_p

    return h, p / sum_p

def log_likelihood_student(x, mu, sigma_square, df=2.0):
    sigma = torch.sqrt(sigma_square)
    dist = torch.distributions.studentT.StudentT(df=df, loc=mu, scale=sigma)
    return torch.sum(dist.log_prob(x), dim=1)

def tsne_repel(z_dim, z_batch, batch_dim, p):
    nu = z_dim - 1

    sum_y = torch.sum(z_batch**2, dim=1)
    num = -2.0 * z_batch @ z_batch.T + torch.reshape(sum_y, [-1, 1]) + sum_y
    num = num / nu

    p = p + 0.1 / batch_dim
    p = p / torch.unsqueeze(torch.sum(p, dim=1), dim=1)

    num = torch.pow(1.0 + num, -(nu + 1.0) / 2.0)
    attraction = p * torch.log(num)
    attraction = -torch.sum(attraction)

    den = torch.sum(num, dim=1) - 1
    den[den==0] = 1e-8  # Zero-valued elements will
    repellant = torch.sum(torch.log(den))

    return (repellant + attraction) / batch_dim

# !!!
# Found at https://stackoverflow.com/questions/107705/disable-output-buffering
# This code makes sure our model print immediately
# !!!
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
