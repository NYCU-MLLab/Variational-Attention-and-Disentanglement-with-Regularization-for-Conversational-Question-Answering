import torch
from torch import nn

class GenerativeDecoder(nn.Module):
    def __init__(self, hparams, vocabulary, options):
        super().__init__()
        self.hparams = hparams
        self.options = options
        self.vocabulary = vocabulary
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.word_embed = nn.Embedding(
            len(vocabulary),
            hparams.word_embedding_size,
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.lstm = nn.LSTM(
            hparams.word_embedding_size,
            hparams.lstm_hidden_size,
            hparams.lstm_num_layers,
            batch_first=True,
            # dropout=hparams.dropout,
        )
        self.lstm_to_words = nn.Linear(hparams.lstm_hidden_size, len(vocabulary))
        self.dropout = nn.Dropout(p=hparams.dropout)
        # self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.vocabulary.PAD_INDEX)

    def forward(self, encoder_output, batch):
        """
            Given `encoder_output`, learn to autoregressively predict ground-truth
            answer word-by-word during training.
            During evaluation, assign log-likelihood scores to all answer options.

            Parameters
            ----------
            encoder_output : torch.Size([batch_size, num_rounds, 512])
            batch          : For detailed dataset structure, please refer to 
                             data/visdial_daset.py
                             data/vqa_dataset.py

            Returns
            -------
            encoder_output: torch.Tensor
                Output from the encoder through its forward pass.
                (batch_size, num_rounds, lstm_hidden_size)
        """

        word_in = batch[self.options + "_in"].to(self.device)
        bs, nr, seq_len = word_in.size()

        word_in = word_in.view(bs * nr, seq_len)

        # Reshape word embeddings input to be set as input format of LSTM.
        word_in_embed = self.word_embed(word_in)       # bs*nr, seq_len, 300

        # Reshape encoder output to be set as initial hidden state of LSTM.
        encoder_output = encoder_output.view(bs * nr, -1)                    # bs*nr, 512
        init_hidden = encoder_output.unsqueeze(0)                            # 1, bs*nr, 512
        init_hidden = init_hidden.repeat(self.hparams.lstm_num_layers, 1, 1) # num_layers, bs*nr, 512
        init_cell = torch.zeros_like(init_hidden)
        
        word_out_emb, (hidden, cell) = self.lstm(word_in_embed, (init_hidden, init_cell))
        word_out_emb = word_out_emb.view(bs, nr, seq_len, -1) # bs, nr, seq_len, 512

        word_out_emb = self.dropout(word_out_emb)
        word_scores = self.lstm_to_words(word_out_emb)        # bs, nr, seq_len, vocabulary_size
        return word_scores

    def criterion(self, decoder_output, batch):
        word_scores = decoder_output.view(-1, decoder_output.size(-1))
        word_out = batch[self.options + "_out"].to(self.device)
        word_out = word_out.view(-1)
        loss_gen = self.loss(word_scores, word_out)
        return loss_gen


class Predictor(nn.Module):
    def __init__(self, hparams, dim_in):
        super().__init__()
        self.hparams = hparams
        self.predicter = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(dim_in, hparams.num_answers),
        )
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input_emb):
        pred_scores = self.predicter(input_emb)
        return pred_scores
    
    def criterion(self, pred_scores, batch):
        pred_scores = pred_scores.view(-1, pred_scores.size(-1))
        ans = batch["gt_ans"].to(self.device)
        ans = ans.view(-1)
        loss_type = self.loss(pred_scores, ans)
        return loss_type
