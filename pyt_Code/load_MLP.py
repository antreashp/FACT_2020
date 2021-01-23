import torch
import torch.nn as nn
from base import MLP

def load_model(path="models/scvis.pt"):
    checkpoint = torch.load(path)
    state_dict = checkpoint["state_dict"].copy()
    model = MLP(shape=checkpoint["shape"])
    model.load_state_dict(state_dict)
    return model
