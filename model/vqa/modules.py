import os
import numpy as np

import torch
from torch import nn

from model.vqa.encoders import (
    QuestionEncoder, Disentangler,
    VisualEncoder, VisualQuestionEncoder, 
    LinearFusionNetwork
)
from model.vqa.decoders import Predictor, GenerativeDecoder

class TextualEncodeModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.question_encoder = QuestionEncoder()
        self.disentangler = Disentangler(
            hparams.sent_emb_size, 
            hparams.cont_emb_size,
            hparams.type_emb_size
        )

    def forward(self, batch):
        """
            Parameters
            ----------
            batch : For detailed dataset structure, please refer to 
                    data/visdial_daset.py
                    data/vqa_dataset.py

            Returns
            -------
            cont_emb    : torch.Size([batch_size, num_rounds, 512])
            type_emb    : torch.Size([batch_size, num_rounds, 512])
            sep_emb    : torch.Size([batch_size, num_rounds, 786])
        """
        ques_emb, sep_emb = self.question_encoder(batch)
        cont_emb, type_emb = self.disentangler(ques_emb)
        return cont_emb, type_emb, sep_emb
    

class MIEstimator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.dim_x = hparams.cont_emb_size
        self.dim_y = hparams.type_emb_size
        self.mu = nn.Sequential(
            nn.Linear(self.dim_x, 256),
            nn.ReLU(),
            nn.Linear(256, self.dim_y)
        )
        self.logvar = nn.Sequential(
            nn.Linear(self.dim_x, 256),
            nn.ReLU(),
            nn.Linear(256, self.dim_y),
            nn.Tanh()
        )

    def get_mu_logvar(self, emb_x):
        mu = self.mu(emb_x)
        logvar = self.logvar(emb_x)
        return mu, logvar
    
    def forward(self, emb_x, emb_y):
        mu, logvar = self.get_mu_logvar(emb_x)

        # Reshape dimension.
        mu = mu.view(-1, self.dim_y)
        logvar = logvar.view(-1, self.dim_y)
        emb_y = emb_y.view(-1, self.dim_y)

        # log of conditional probability of positive pairs.
        pos = -(mu - emb_y)**2 /2. /logvar.exp()

        mu_unsqe = mu.unsqueeze(1)
        emb_y_unsqe = emb_y.unsqueeze(0)
        neg = -((emb_y_unsqe - mu_unsqe)**2).mean(dim=1) /2. /logvar.exp()

        return (pos.sum(dim=-1) - neg.sum(dim=-1)).mean()

    def log_likelihood(self, emb_x, emb_y):
        # Reshape dimension.
        mu, logvar = self.get_mu_logvar(emb_x)
        mu = mu.view(-1, self.dim_y)
        logvar = logvar.view(-1, self.dim_y)
        emb_y = emb_y.view(-1, self.dim_y)
        return (-(mu - emb_y)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, emb_x, emb_y):
        return -self.log_likelihood(emb_x, emb_y)


class Regularizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, batch, model):
        cont_emb, type_emb, _ = model.encoder.textual_encoder(batch)

        with torch.no_grad():
            img_emb, img_masks = model.encoder.visual_encoder(batch)
            vis_ques_emb = model.encoder.vis_ques_encoder(img_emb, img_masks, cont_emb, type_emb)
            encoder_output = model.encoder.fusion_nn(vis_ques_emb)
            ref_dist = model.decoder(encoder_output)
            sou_dist = model.decoder.type_predictor(type_emb)

        return self._get_doe(ref_dist, sou_dist)

    def _get_doe(self, ref_dist, sou_dist):
        ref = self.softmax(ref_dist.view(-1, ref_dist.size(-1)))
        sou = self.softmax(sou_dist.view(-1, sou_dist.size(-1)))
        log_ref = self.log_softmax(ref_dist.view(-1, ref_dist.size(-1)))
        log_sou = self.log_softmax(sou_dist.view(-1, sou_dist.size(-1)))
        ref_entropy = -1 * (ref * log_ref).sum(-1)
        sou_entropy = -1 * (sou * log_sou).sum(-1)
        return (ref_entropy - 0.25 * sou_entropy).mean()
        # return (ref_entropy).mean()


class VQAEncodeModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.textual_encoder = TextualEncodeModule(hparams)
        self.visual_encoder = VisualEncoder(hparams)
        self.vis_ques_encoder = VisualQuestionEncoder(hparams)
        self.fusion_nn = LinearFusionNetwork(hparams)
    
    def forward(self, batch):
        """
            Parameters
            ----------
            batch : For detailed dataset structure, please refer to 
                    data/visdial_daset.py
                    data/vqa_dataset.py

            Returns
            -------
            cont_emb       : torch.Size([batch_size, num_rounds, 512])
            type_emb       : torch.Size([batch_size, num_rounds, 512])
            encoder_output : torch.Size([batch_size, num_rounds, 512])
        """
        img_emb, img_masks = self.visual_encoder(batch)
        cont_emb, type_emb, _ = self.textual_encoder(batch)
        vis_ques_emb = self.vis_ques_encoder(img_emb, img_masks, cont_emb, type_emb)
        encoder_output = self.fusion_nn(vis_ques_emb)
        return cont_emb, type_emb, encoder_output


class VQADecodeModule(nn.Module):
    def __init__(self, hparams, vocabulary):
        super().__init__()
        self.hparams = hparams
        self.vocabulary = vocabulary
        self.type_predictor = Predictor(hparams, hparams.type_emb_size)
        self.ques_gen = GenerativeDecoder(hparams, vocabulary, options="ques")
        self.ans_predictor = Predictor(hparams, hparams.fusion_out_size)

        # Initializing word_embed using GloVe.
        if os.path.exists(hparams.glove_npy):
            self.ques_gen.word_embed.weight.data = torch.from_numpy(np.load(hparams.glove_npy))
            # print("\t- Loaded GloVe vectors from {}".format(hparams.glove_npy))

    def forward(self, encoder_output):
        word_scores = self.ans_predictor(encoder_output) # bs, nr, num_answers
        return word_scores
        
    def criterion(self, cont_emb, type_emb, word_scores, batch):
        """Disentanglement Loss"""
        ques_output = self.ques_gen(cont_emb, batch)
        loss_cont = self.ques_gen.criterion(ques_output, batch)
        type_scores = self.type_predictor(type_emb)
        loss_type = self.type_predictor.criterion(type_scores, batch)

        """Visual Question Answering Loss"""
        loss_vqa = self.ans_predictor.criterion(word_scores, batch)
    
        return loss_cont, loss_type, loss_vqa
