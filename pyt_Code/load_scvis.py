import torch
from scvis import scvis_encoder

def load_model(path="models/scvis.pt", get_hparams=False):
    checkpoint = torch.load(path)
    state_dict = checkpoint["state_dict"].copy()
    model = scvis_encoder(encoder_shape=checkpoint["hyper_parameters"]["encoder_shape"],
                          decoder_shape=checkpoint["hyper_parameters"]["decoder_shape"],
                          activate_op=checkpoint["hyper_parameters"]["activate_op"],
                          eps=checkpoint["hyper_parameters"]["eps"],
                          max_sigma_square=checkpoint["hyper_parameters"]["max_sigma_square"],
                          prob=checkpoint["hyper_parameters"]["prob"],
                          initial=checkpoint["hyper_parameters"]["initial"])
    model.load_state_dict(state_dict)
    if get_hparams:
        return model, checkpoint["hyper_parameters"]
    else:
        return model
