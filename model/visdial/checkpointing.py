import os
import torch
from torch import nn, optim

class CheckpointManager(object):
    def __init__(
        self, 
        mi_estimator,
        encoder,
        decoder,
        checkpoint_dirpath,
    ):
        self.mi_estimator = mi_estimator
        self.encoder = encoder
        self.decoder = decoder
        self.ckpt_dirpath = checkpoint_dirpath

        if not os.path.exists(self.ckpt_dirpath):
            os.makedirs(self.ckpt_dirpath)
    
    def step(self, epoch):
        file_name = "checkpoint_visdial.pth"
        model_path = os.path.join(self.ckpt_dirpath, file_name)
        save_dict = {
            "mi_estimator": self.mi_estimator.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        torch.save(save_dict, model_path)

