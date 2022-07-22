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

    def forward(self, batch):       
        cont_emb, type_emb, encoder_output = self.encoder(batch)
        word_scores = self.decoder(encoder_output)
        return cont_emb, type_emb, word_scores

    def criterion(self, cont_emb, type_emb, word_scores, batch):
        loss_cont, loss_type, loss_vqa = self.decoder.criterion( 
            cont_emb, type_emb, 
            word_scores, 
            batch
        )
        return loss_cont, loss_type, loss_vqa