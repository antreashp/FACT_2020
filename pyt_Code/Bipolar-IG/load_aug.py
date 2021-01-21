import torch
import torch.nn as nn
from scvis import scvis_encoder

def load_model(path="models/scvis.pt"):
    checkpoint = torch.load(path)
    state_dict = checkpoint["state_dict"].copy()
    model = scvis_encoder(encoder_shape=checkpoint["hyper_parameters"]["encoder_shape"],
                          decoder_shape=checkpoint["hyper_parameters"]["decoder_shape"],
                          activate_op=checkpoint["hyper_parameters"]["activate_op"],
                          eps=checkpoint["hyper_parameters"]["eps"],
                          max_sigma_square=checkpoint["hyper_parameters"]["max_sigma_square"],
                          initial=checkpoint["hyper_parameters"]["initial"])
    model.load_state_dict(state_dict)
    return model
