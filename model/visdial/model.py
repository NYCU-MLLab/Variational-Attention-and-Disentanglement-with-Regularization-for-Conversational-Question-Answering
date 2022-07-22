import os
import torch
from torch import nn

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        """
            Convenience wrapper module, wrapping Encoder and Decoder modules.

            Parameters
            ----------
            encoder: nn.Module
            decoder: nn.Module
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def load_pretrained(self, load_pth):
        if not os.path.exists(load_pth):
            raise FileNotFoundError(f"Checkpoints file do not exist at {load_pth}")
        self.encoder.load_pretrained(load_pth)
        self.decoder.load_pretrained(load_pth)

    def forward(self, batch, test=False):       

        if test:
            cont_emb, type_emb, encoder_output = self.encoder(batch, test)
            pred_score = self.decoder(encoder_output, batch)
            return cont_emb, type_emb, pred_score
        else:
            cont_emb, type_emb, encoder_output, KLD = self.encoder(batch)
            pred_score = self.decoder(encoder_output, batch) 
            return cont_emb, type_emb, KLD, pred_score

    def criterion(self, cont_emb, pred_score, batch):
        loss_cont, loss_disc = self.decoder.criterion(cont_emb, pred_score, batch)
        return loss_cont, loss_disc