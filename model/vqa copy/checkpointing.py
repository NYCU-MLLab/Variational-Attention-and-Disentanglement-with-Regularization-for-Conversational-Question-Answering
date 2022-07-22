import os
import torch
from torch import nn, optim

class CheckpointManager(object):
    def __init__(
        self, 
        mi_estimator,
        textual_encoder,
        visual_encoder,
        vis_ques_encoder,
        fusion_nn,
        decoder,
        checkpoint_dirpath,
    ):
        self.mi_estimator = mi_estimator
        self.textual_encoder = textual_encoder
        self.visual_encoder = visual_encoder
        self.vis_ques_encoder = vis_ques_encoder
        self.fusion_nn = fusion_nn
        self.decoder = decoder
        self.ckpt_dirpath = checkpoint_dirpath

        if not os.path.exists(self.ckpt_dirpath):
            os.makedirs(self.ckpt_dirpath)
    
    def step(self, epoch):
        file_name = "checkpoint_" + str(epoch) + ".pth"
        model_path = os.path.join(self.ckpt_dirpath, file_name)
        save_dict = {
            "mi_estimator": self.mi_estimator.state_dict(),
            "textual_encoder": self.textual_encoder.state_dict(),
            "visual_encoder": self.visual_encoder.state_dict(),
            "vis_ques_encoder": self.vis_ques_encoder.state_dict(),
            "fusion_nn": self.fusion_nn.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        torch.save(save_dict, model_path)


def load_checkpoint(checkpoint_pthpath):
    components = torch.load(checkpoint_pthpath)
    load_item = [
        components["mi_estimator"],
        components["textual_encoder"], 
        components["visual_encoder"], 
        components["vis_ques_encoder"],
        components["fusion_nn"],
        components["decoder"],
    ]
    return load_item

