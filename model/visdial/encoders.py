import torch
from torch import nn
from transformers import BertModel

from model.vqa.encoders import PositionwiseFeedForward


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1, max_len, d_model
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        """
            Parameter
            ---------
            x: [shape: batch, seq_len, embed_size]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class HistoryEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(hparams.sent_emb_size, hparams.hist_emb_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # if self.hparams.add_positional_encoding:
        #     self.pos_encoder = PositionalEncoding(hparams.sent_emb_size, hparams.max_round_history + 1)
        
        # self.layernorm = nn.LayerNorm(hparams.hist_emb_size)
        # self.self_multihead_attn = nn.MultiheadAttention(
        #     embed_dim=self.hparams.hist_emb_size,
        #     num_heads=self.hparams.num_heads,
        # )
        # self.pnn = PositionwiseFeedForward(
        #     d_model=hparams.hist_emb_size, 
        #     d_ff=hparams.hist_emb_size//2,
        # )
    
    def forward(self, batch):
        hist, hist_seg = self.init_batch(batch)
        hist_emb, hist_masks = self.encode(hist, hist_seg) 
        return hist_emb, hist_masks

    def init_batch(self, batch):
        return batch["hist"].to(self.device), batch["hist_seg"].to(self.device)

    def _get_attn_masks(self, hist):
        return (hist > 0).long()

    def _get_hist_masks(self, attn_masks):
        return (0 == attn_masks.sum(-1))

    def self_attn(self, hist_norm, hist_masks):
        q = hist_norm.permute(1, 0, 2)
        k = hist_norm.permute(1, 0, 2)
        v = hist_norm.permute(1, 0, 2)
        hist_attn, _ = self.self_multihead_attn(q, k, v, key_padding_mask=hist_masks)
        hist_attn = hist_attn.permute(1, 0, 2)
        return hist_attn

    def add(self, hist_org, hist_trans):
        assert hist_org.size() == hist_trans.size()
        return hist_org + hist_trans

    def encode(self, hist, hist_seg):
        assert hist.size() == hist_seg.size()
        bs, nr, ns, seq_l = hist.size()
        hist = hist.view(-1, seq_l)
        hist_seg = hist_seg.view(-1, seq_l)
        attn_masks = self._get_attn_masks(hist)                           # bs*nr*ns, seq_l
        hist_masks = self._get_hist_masks(attn_masks.view(bs*nr, ns, -1)) # bs*nr, ns
        outputs = self.bert_encoder(hist, attention_mask=attn_masks, token_type_ids=hist_seg)
        outputs = outputs[0]
        hist_emb = outputs[:, 0, :]               # bs*nr*ns, 1, 768
        hist_emb = hist_emb.view(bs*nr, ns, -1)   # bs*nr, ns, 768

        # if self.hparams.add_positional_encoding:
        #     hist_emb = self.pos_encoder(hist_emb) # bs*nr, ns, 768

        hist_emb = self.linear(hist_emb)          # bs*nr, ns, 512
        # hist_norm = self.layernorm(hist_emb)
        # hist_attn = self.self_attn(hist_norm, hist_masks) 
        # hist_emb = self.add(hist_emb, hist_attn)
        # hist_norm = self.layernorm(hist_emb)
        # hist_trans = self.pnn(hist_norm)
        # hist_emb = self.add(hist_emb, hist_trans) # bs*nr, ns, 512

        hist_emb = hist_emb.view(bs, nr, ns, -1)  # bs, nr, ns, 512
        hist_masks = hist_masks.view(bs, nr, ns)  # bs, nr, ns

        return hist_emb, hist_masks


class QAPairEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, batch):
        qa_pairs, qa_seg = self.init_batch(batch)
        sep_emb = self.encode(qa_pairs, qa_seg) 
        return sep_emb
    
    def init_batch(self, batch):
        qa_pairs = batch["qa_pairs"].to(self.device)
        qa_seg = batch["qa_seg"].to(self.device)
        return qa_pairs, qa_seg

    def _get_attn_masks(self, qa_pairs):
        return (qa_pairs > 0).long()

    def _get_SEP_emb(self, qa_pairs, bert_outputs):
        sep_emb = []
        for i in range(qa_pairs.size(0)):
            sep_pos = torch.where(qa_pairs[i] == 102)[0]
            emb = bert_outputs[i, sep_pos[-1], :]
            sep_emb.append(emb.unsqueeze(0))
        sep_emb = torch.cat(sep_emb, dim=0)
        return sep_emb

    def encode(self, qa_pairs, qa_seg):
        assert qa_pairs.size() == qa_seg.size()
        bs, nr, seq_l = qa_pairs.size()
        qa_pairs = qa_pairs.view(-1, seq_l)
        qa_seg = qa_seg.view(-1, seq_l)
    
        attn_masks = self._get_attn_masks(qa_pairs)
        outputs = self.bert_encoder(qa_pairs, attention_mask=attn_masks, token_type_ids=qa_seg)
        outputs = outputs[0]
        sep_emb = self._get_SEP_emb(qa_pairs, outputs)
        sep_emb = sep_emb.view(bs, nr, -1)  # bs, nr, 786
        return sep_emb
