import os
import numpy as np

import torch
from torch import nn

from model.vqa.encoders import (
    VisualEncoder, VisualQuestionEncoder, 
    LinearFusionNetwork
)
from model.vqa.modules import TextualEncodeModule, VQADecodeModule
from model.visdial.encoders import QAPairEncoder, HistoryEncoder
from model.visdial.decoders import DiscriminativeDecoder

class BernoulliVariationalAttentionModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.pmax = 0.8
        self.pmin = 0
        self.layernorm = nn.LayerNorm(hparams.hist_emb_size)
        self.softmax = nn.Softmax(dim=-1)
        self.prior_proj = nn.Linear(hparams.sent_emb_size, hparams.hist_emb_size)
        self.post_proj = nn.Linear(hparams.sent_emb_size, hparams.hist_emb_size)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hparams.hist_emb_size,
            num_heads=self.hparams.num_heads,
        )
    
    def _init_prob(self, prob):
        return self.pmin + (self.pmax - self.pmin) * prob

    def _get_prob(self, input_norm, hist_norm, hist_masks):
        q = input_norm.permute(1, 0, 2)
        k = hist_norm.permute(1, 0, 2)
        v = hist_norm.permute(1, 0, 2)
        _, prob = self.multihead_attn(q, k, v, key_padding_mask=hist_masks)
        return prob

    def _get_sample_ave_bernou_mu_var(self, bernou_prob):
        mu = bernou_prob
        var = (bernou_prob * (1 - bernou_prob)) / self.hparams.num_samples
        return mu, var
    
    def _get_reparam_mu_var(self, mu, var, hist_norm):
        mu = torch.matmul(mu, hist_norm)
        var = torch.matmul(var, hist_norm ** 2)
        return mu, var

    def reparameterize(self, mu, var):
        std = var ** 0.5
        eps = torch.randn_like(std)
        return eps * std + mu

    def KLD_cost(self, mu_p, std_p, mu_q, std_q):
        KLD = 0.5 * (2 * torch.log(std_p / std_q) - 1 + (std_q / std_p) ** 2 + ((mu_p - mu_q) / std_p) ** 2).sum()
        return KLD

    def add(self, emb_org, emb_trans):
        assert emb_org.size() == emb_trans.size()
        return emb_org + emb_trans

    def forward(self, hist_emb, hist_masks, prior_emb, post_emb=None):
        """
            Parameters
            ----------
            hist_emb   : torch.Size([batch_size, num_rounds, num_sentence, 512])
            hist_masks : torch.Size([batch_size, num_rounds, num_sentence])
            prior_emb  : torch.Size([batch_size, num_rounds, 786])
            post_emb   : torch.Size([batch_size, num_rounds, 786])
        """
        bs, nr, _ = prior_emb.size()
        hist_norm_emb = self.layernorm(hist_emb)   # bs, nr, ns, 512
        prior_emb = self.prior_proj(prior_emb)  
        prior_norm_emb = self.layernorm(prior_emb) # bs, nr, 512

        if post_emb is not None:
            post_emb = self.post_proj(post_emb)    
            post_norm_emb = self.layernorm(post_emb) # bs, nr, 512
            mu_pri, std_pri = [], []
            mu_pos, std_pos = [], []
            hist_rel_pos = []
        else:
            hist_rel_pri = []
        
        for r in range(nr):
            hist_norm = hist_norm_emb[:, r, :, :] # bs, ns, 512

            prior_org = prior_emb[:, r, :].unsqueeze(1)                          
            prior_norm = prior_norm_emb[:, r, :].unsqueeze(1)                       # bs, 1, 512
            prior_prob = self._get_prob(prior_norm, hist_norm, hist_masks[:, r, :]) # bs, 1, ns
           
            if post_emb is not None:
                mu, var = self._get_sample_ave_bernou_mu_var(self._init_prob(prior_prob)) # bs, 1, ns
                prior_mu, prior_var = self._get_reparam_mu_var(mu, var, hist_norm)        # bs, 1, 512

                post_org = post_emb[:, r, :].unsqueeze(1)
                post_norm = post_norm_emb[:, r, :].unsqueeze(1)                       # bs, 1, 512
                post_prob = self._get_prob(post_norm, hist_norm, hist_masks[:, r, :]) # bs, 1, ns
                mu, var = self._get_sample_ave_bernou_mu_var(self._init_prob(post_prob))
                post_mu, post_var = self._get_reparam_mu_var(mu, var, hist_norm)      # bs, 1, 512
                
                reparam_hist_rel = self.reparameterize(post_mu, post_var)
                reparam_hist_rel = self.add(post_org, reparam_hist_rel)
                hist_rel_pos.append(reparam_hist_rel)

                mu_pri.append(prior_mu)
                mu_pos.append(post_mu)
                std_pri.append(prior_var ** 0.5)
                std_pos.append(post_var ** 0.5)
            else:
                sampled_weights = torch.bernoulli(self._init_prob(prior_prob))
                sampled_weights = self.softmax(sampled_weights)
                hist_rel = torch.matmul(sampled_weights, hist_norm) # bs, 1, 512
                hist_rel = self.add(prior_org, hist_rel)
                hist_rel_pri.append(hist_rel)

        if post_emb is not None:
            hist_rel_emb = torch.cat(hist_rel_pos, dim=1) # bs, nr, 512
            mu_pri = torch.cat(mu_pri, dim=1).view(-1, self.hparams.hist_emb_size)   
            std_pri = torch.cat(std_pri, dim=1).view(-1, self.hparams.hist_emb_size)
            mu_pos = torch.cat(mu_pos, dim=1).view(-1, self.hparams.hist_emb_size)
            std_pos = torch.cat(std_pos, dim=1).view(-1, self.hparams.hist_emb_size)
            KLD = self.KLD_cost(mu_pri, std_pri, mu_pos, std_pos)
            return hist_rel_emb, KLD
        else:
            hist_rel_emb = torch.cat(hist_rel_pri, dim=1) # bs, nr, 512
            return hist_rel_emb


class VisDialEncodeModule(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.qa_pair_encoder = QAPairEncoder(hparams)
        self.history_encoder = HistoryEncoder(hparams)
        self.bva_module = BernoulliVariationalAttentionModule(hparams)
        self.cont_proj = nn.Linear(hparams.hist_emb_size + hparams.cont_emb_size, hparams.cont_emb_size)
        self.fusion_nn = LinearFusionNetwork(hparams)

        self.textual_encoder = TextualEncodeModule(hparams)
        self.visual_encoder = VisualEncoder(hparams)
        self.vis_ques_encoder = VisualQuestionEncoder(hparams)

    def load_pretrained(self, load_pth):
        checkpoint = torch.load(load_pth)
        self.textual_encoder.load_state_dict(checkpoint["textual_encoder"])
        self.visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        self.vis_ques_encoder.load_state_dict(checkpoint["vis_ques_encoder"])
        print("\t- Loaded Textual Encoder weights from {}".format(load_pth))
        print("\t- Loaded Visual  Encoder weights from {}".format(load_pth))
        print("\t- Loaded VisQues Encoder weights from {}".format(load_pth))

    def forward(self, batch, test=False):
        img_emb, img_masks = self.visual_encoder(batch)
        hist_emb, hist_masks = self.history_encoder(batch)
        cont_emb, type_emb, prior_emb = self.textual_encoder(batch)
        
        if test:
            hist_rel_emb = self.bva_module(hist_emb, hist_masks, prior_emb)
        else:
            post_emb = self.qa_pair_encoder(batch)
            hist_rel_emb, KLD = self.bva_module(hist_emb, hist_masks, prior_emb, post_emb=post_emb)
        
        hist_cont_emb = torch.cat([hist_rel_emb, cont_emb], dim=-1)
        hist_cont_emb = self.cont_proj(hist_cont_emb)
        vis_ques_emb = self.vis_ques_encoder(img_emb, img_masks, hist_cont_emb, type_emb.detach())
        encoder_output = self.fusion_nn(vis_ques_emb)

        if test:
            return cont_emb, type_emb, encoder_output
        else:
            return cont_emb, type_emb, encoder_output, KLD


class VisDialDecodeModule(nn.Module):
    def __init__(self, hparams, vocabulary):
        super().__init__()
        self.hparams = hparams
        self.vocabulary = vocabulary
        self.disc_decoder = DiscriminativeDecoder(hparams, vocabulary)
        if os.path.exists(hparams.glove_npy):
            self.disc_decoder.word_embed.weight.data = torch.from_numpy(np.load(hparams.glove_npy))
            print("\t- Loaded GloVe vectors from {}".format(hparams.glove_npy))

        vqa_decoder = VQADecodeModule(self.hparams, self.vocabulary)
        self.ques_gen = vqa_decoder.ques_gen
        # self.ques_gen = None

    def load_pretrained(self, load_pth):
        checkpoint = torch.load(load_pth)
        vqa_decoder = VQADecodeModule(self.hparams, self.vocabulary)
        vqa_decoder.load_state_dict(checkpoint["decoder"])
        self.ques_gen = vqa_decoder.ques_gen
        print("\t- Loaded Disentanglement weights from {}".format(load_pth))
    
    def forward(self, encoder_output, batch):
        pred_score = self.disc_decoder(encoder_output, batch)
        return pred_score

    def criterion(self, cont_emb, pred_score, batch):
        ques_output = self.ques_gen(cont_emb, batch)
        loss_cont = self.ques_gen.criterion(ques_output, batch)
        loss_disc = self.disc_decoder.criterion(pred_score, batch)
        return loss_cont, loss_disc
