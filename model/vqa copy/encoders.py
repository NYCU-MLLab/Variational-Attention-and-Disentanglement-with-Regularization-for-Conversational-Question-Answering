import numpy as np

import torch
from torch import nn
from transformers import BertModel

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN = max(0, xw_1 + b_1)w_2 + b_2 equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.network(x)


class QuestionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, batch):
        ques = self.init_batch(batch)
        ques_emb = self.encode(ques) 
        return ques_emb

    def init_batch(self, batch):
        return batch["ques"].to(self.device)

    def _get_SEP_emb(self, ques, bert_outputs):
        sep_emb = []
        for i in range(ques.size(0)):
            sep_pos = torch.where(ques[i] == 102)[0]
            emb = bert_outputs[i, sep_pos[-1], :]
            sep_emb.append(emb.unsqueeze(0))
        sep_emb = torch.cat(sep_emb, dim=0)
        return sep_emb

    def encode(self, ques):
        bs, nr, seq_l = ques.size()
        ques = ques.view(-1, seq_l)
        outputs = self.bert_encoder(ques)
        outputs = outputs[0]
        ques_emb = outputs[:, 0, :]
        sep_emb = self._get_SEP_emb(ques, outputs)
        
        ques_emb = ques_emb.view(bs, nr, -1)
        sep_emb = sep_emb.view(bs, nr, -1)
        return ques_emb, sep_emb


class Disentangler(nn.Module):
    def __init__(self, dim_in, dim_cont, dim_type):
        super().__init__()
        self.dim_cont = dim_cont
        self.dim_type = dim_type
        self.encoder = nn.Linear(dim_in, dim_cont + dim_type)        

    def forward(self, ques_emb):
        output_emb = self.encoder(ques_emb)
        cont_emb = output_emb[:, :, :self.dim_cont]
        type_emb = output_emb[:, :, self.dim_cont:]
        return cont_emb, type_emb


class VisualEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """Image Encoders"""
        if "dan" in hparams.img_feature_type and hparams.spatial_feat:
            img_input_size = hparams.img_feat_size + hparams.img_sp_feat_size
        else:
            img_input_size = hparams.img_feat_size
        self.img_encoder = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(img_input_size, hparams.img_hidden_size)
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
            img_emb   : torch.Size([batch_size, num_proposals, 1024])
            img_masks : torch.Size([batch_size, num_proposals])
        """
        img_feats, img_sp_feats = self.init_batch(batch)
        img_masks = self._get_padding_masks(img_feats) # bs, np

        if img_sp_feats is not None:
            img = torch.cat([img_feats, img_sp_feats], dim=-1) # bs, np, 2054
        else:
            img = img_feats                                    # bs, np, 2048
        
        img_emb = self.img_encoder(img)    # bs, np, 1024
        return img_emb, img_masks

    def init_batch(self, batch):
        img_feats = batch["img_feats"].to(self.device)
        if "dan" in self.hparams.img_feature_type and self.hparams.spatial_feat:
            img_sp_feats = batch["img_sp_feats"].to(self.device)
        else:
            img_sp_feats = None
        return img_feats, img_sp_feats

    def _get_padding_masks(self, img_feats):        
        return (0 == img_feats.abs().sum(-1))


class VisualQuestionEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        """Layer Nomalization"""
        self.layernorm = nn.LayerNorm(hparams.img_hidden_size)

        """Image Question Fusion Encoders"""
        self.W_cont = nn.Linear(hparams.cont_emb_size, hparams.img_hidden_size)
        self.W_cat = nn.Linear(
            hparams.cont_emb_size + hparams.type_emb_size, 
            hparams.img_hidden_size
        )
        self.cont_img_multihead_attn = nn.MultiheadAttention(
            embed_dim=hparams.img_hidden_size,
            num_heads=hparams.num_heads,
        )
        self.cat_img_multihead_attn = nn.MultiheadAttention(
            embed_dim=hparams.img_hidden_size,
            num_heads=hparams.num_heads,
        )
        self.pnn = self.pnn = PositionwiseFeedForward(
            d_model=hparams.img_hidden_size, 
            d_ff=hparams.img_hidden_size,
        )

    def forward(self, img_emb, img_masks, cont_emb, type_emb):
        """
            Parameters
            ----------
            img_emb   : torch.Size([batch_size, num_proposals, 1024])
            img_masks : torch.Size([batch_size, num_proposals])
            cont_emb  : torch.Size([batch_size, num_rounds, 512])
            type_emb  : torch.Size([batch_size, num_rounds, 512])

            Returns
            -------
            vis_ques_emb : torch.Size([batch_size, num_rounds, 1536])
        """

        """First Stage: Content - Image"""
        cont_q = self.W_cont(cont_emb)      
        cont_norm_emb = self.layernorm(cont_q) # bs, nr, 1024
        img_norm_emb = self.layernorm(img_emb) # bs, np, 1024
        _, np, _ = img_norm_emb.size()

        attn_weights = self.cont_img_attn(     # bs, nr, np
            cont_norm_emb,
            img_norm_emb, 
            img_masks
        )
        relevant_idx = self._get_relevant_index(attn_weights, np, k=self.hparams.top_k) # bs, nr, 16
        img_irr_masks = self._get_irrelevant_mask(img_masks, relevant_idx)          # bs, nr, np

        """Second Stage: Question - Image"""
        _, num_r, _ = cont_emb.size()
        cat_emb = torch.cat([cont_emb, type_emb], dim=-1) # bs, nr, 1024
        cat_q = self.W_cat(cat_emb)                       # bs, nr, 1024
        cat_norm_emb = self.layernorm(cat_q)              # bs, nr, 1024
        
        vis_relevant_feat = []
        # vis_relevant_weight = []

        for nr in range(num_r):
            cat_org = cat_q[:, nr, :].unsqueeze(1)         # bs, 1, 1024
            cat_norm = cat_norm_emb[:, nr, :].unsqueeze(1) # bs, 1, 1024
            img_norm = img_norm_emb                        # bs, np, 1024 
            img_attn, attn_weights = self.cat_img_attn(cat_norm, img_norm, img_irr_masks[:, nr, :])
            img_attn = self.add(cat_org, img_attn)
            img_attn_norm = self.layernorm(img_attn)
            img_trans = self.pnn(img_attn_norm)
            img_attn = self.add(img_attn, img_trans)
            vis_relevant_feat.append(img_attn)
            # vis_relevant_weight.append(attn_weights)
        
        vis_relevant = torch.cat(vis_relevant_feat, dim=1)         # bs, nr, 1024
        vis_ques_emb = torch.cat([vis_relevant, cat_emb], dim=-1)  # bs, nr, 2048
        return vis_ques_emb

    def cont_img_attn(self, cont_norm, img_norm, img_masks):
        q = cont_norm.permute(1, 0, 2)
        k = img_norm.permute(1, 0, 2)
        v = img_norm.permute(1, 0, 2)
        _, attn_weights = self.cont_img_multihead_attn(q, k, v, key_padding_mask=img_masks)
        return attn_weights
    
    def cat_img_attn(self, cat_norm, img_norm, img_masks):
        q = cat_norm.permute(1, 0, 2)
        k = img_norm.permute(1, 0, 2)
        v = img_norm.permute(1, 0, 2)
        img_attn, attn_weights = self.cat_img_multihead_attn(q, k, v, key_padding_mask=img_masks)
        img_attn = img_attn.permute(1, 0, 2)
        return img_attn, attn_weights

    def _get_relevant_index(self, attn_weights, np, k=16):
        if k > np:
            k = np
        _, relevant_idx = torch.topk(attn_weights, k=k, dim=2)
        return relevant_idx

    def _get_irrelevant_mask(self, img_masks, relevant_idx):
        bs, num_p = img_masks.size()
        bs_, nr, k = relevant_idx.size()
        assert bs == bs_

        org_masks = img_masks.long()
        org_masks = [org_masks] * nr
        org_masks = torch.stack(org_masks, dim=1)    # bs, nr, np

        relevant_idx = relevant_idx.view(bs*nr, k)
        rel_masks = torch.ones(bs*nr, num_p).to(img_masks.get_device())
        
        for i in range(bs*nr):
            rel_masks[i][relevant_idx[i]] = 0
        rel_masks = rel_masks.view(bs, nr, num_p).long() # bs, nr, np
        
        new_masks = org_masks + rel_masks
        return (new_masks > 0)

    def add(self, img_org, img_trans):
        assert img_org.size() == img_trans.size()
        return img_org + img_trans


class LinearFusionNetwork(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        dim_in = hparams.img_hidden_size + hparams.cont_emb_size + hparams.type_emb_size
        dim_out = hparams.fusion_out_size
        self.network = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(dim_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim_out)
        )

    def forward(self, input_emb):
        return self.network(input_emb)
