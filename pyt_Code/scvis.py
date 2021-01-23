import torch
import torch.nn as nn

class scvis(nn.Module):
    def __init__(self, encoder_shape, decoder_shape, activate_op=nn.ELU(), eps=1e-6, max_sigma_square=1e10, prob=0.5, initial=None):
        if initial is None:  # Do not define this operation if
            initial = nn.Sequential(nn.Linear(encoder_shape[0], decoder_shape[1]),  # encoder_shape[0] should be the input size
                                    activate_op)

        self.encoder = scvis_encoder(encoder_shape, activate_op, eps, max_sigma_square, prob, initial)
        self.decoder = scvis_decoder(decoder_shape, activate_op, eps, max_sigma_square)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        noise = torch.normal(torch.zeros_like(mu), torch.ones_like(sigma))
        z = sigma * noise + mu
        mu, sigma_square = self.decoder(z)
        return mu, sigma_square

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

class scvis_encoder(nn.Module):
    def __init__(self, shape, activate_op=nn.ELU(), eps=1e-6, max_sigma_square=1e10, prob=0.5, initial=None):
        if initial is None:  # Do not define this operation if
            self.initial_layers = nn.Sequential(nn.Linear(shape[0], shape[1]),  # shape[0] should be the input size
                                                activate_op)
        else:
            self.initial_layers = initial

        self.mu_layer = nn.Linear(shape[1], shape[2])
        self.sigma_layer = nn.Linear(shape[1], shape[2])

        self.soft_plus = nn.Softplus()

        self.prob = prob
        self.eps = eps
        self.max_sigma_square = max_sigma_square

    def forward(self, x, prob=None):
        hidden_layer_out = self.initial_layers(x)
        if prob != 0 and self.training:
            if prob is None:
                dropout = nn.Dropout(self.prob)
            else:
                dropout = nn.Dropout(prob)
            hidden_layer_out = dropout(hidden_layer_out)

        mu = self.mu_layer(hidden_layer_out)
        sigma = self.sigma_layer(hidden_layer_out)
        return mu, torch.clamp(self.soft_plus(sigma), self.eps, self.max_sigma_square)

class scvis_decoder(nn.Module):
    def __init__(self, shape, activate_op=nn.ELU(), eps=1e-6, max_sigma_square=1e10):
        self.initial_layers = nn.Sequential(nn.Linear(shape[0], shape[1]),  # shape[0] should be the input size
                                            activate_op)

        in_size = shape[1]
        hidden_layers = []
        for out_size in shape[2:-1]:
            hidden_layers.append(nn.Linear(in_size, out_size))
            hidden_layers.append(activate_op)
            in_size = out_size
        if hidden_layers:
            self.hidden_layers = nn.Sequential(hidden_layers)
        else:
            self.hidden_layers = None

        self.mu_layer = nn.Linear(shape[-2], shape[-1])
        self.sigma_layer = nn.Linear(shape[-2], shape[-1])

        self.soft_plus = nn.Softplus()

        self.eps = eps
        self.max_sigma_square = max_sigma_square

    def forward(self, z):
        hidden_layer_out = self.initial_layers(z)
        hidden_layer_out = self.hidden_layers(hidden_layer_out)
        mu = self.mu_layer(hidden_layer_out)
        sigma = self.sigma_layer(hidden_layer_out)
        return mu, torch.clamp(self.soft_plus(sigma), self.eps, self.max_sigma_square)
