import torch

def make_checkpoint(params, state_dict, path, epoch=-1):
    r"""
    :param params: A dictionary with hyperparameter names as keys, and their settings as keys
    :param state_dict: The model's state dictionary
    :param path: The path, including the file name, to store the data to
    :param epoch: The current training epoch (defaults to -1)
    :return: None
    """
    torch.save({"hyper_parameters":params,
                "state_dict":state_dict,
                "epoch":epoch}, path)
