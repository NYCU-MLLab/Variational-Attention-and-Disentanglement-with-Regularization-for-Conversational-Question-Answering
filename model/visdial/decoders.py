import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DynamicRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """
            A wrapper over pytorch's rnn to handle sequences of variable length.

            Arguments
            ---------
            seq_input : torch.Tensor
                Input sequence tensor (padded) for RNN model.
                Shape: (batch_size, max_sequence_length, embed_size)
            seq_lens : torch.LongTensor
                Length of sequences (b, )
            initial_state : torch.Tensor
                Initial (hidden, cell) states of RNN model.

            Returns
            -------
                Single tensor of shape (batch_size, rnn_hidden_size) corresponding
                to the outputs of the RNN model at the last time step of each input
                sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True
        )

        if initial_state is not None:
            hx = initial_state
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            sorted_hx = None

        self.rnn_model.flatten_parameters()

        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, sorted_hx)

        # pick hidden and cell states of last layer
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length
        )[0].index_select(dim=0, index=bwd_order)

        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().view(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order


class DiscriminativeDecoder(nn.Module):
    def __init__(self, hparams, vocabulary):
        super().__init__()
        self.hparams = hparams

        self.word_embed = nn.Embedding(
            len(vocabulary),
            hparams.word_embedding_size,
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.option_rnn = nn.LSTM(
            hparams.word_embedding_size,
            hparams.lstm_hidden_size,
            hparams.lstm_num_layers,
            batch_first=True,
            dropout=hparams.dropout,
        )

        # Options are variable length padded sequences, use DynamicRNN.
        self.option_rnn = DynamicRNN(self.option_rnn)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, encoder_output, batch):
        """
            Given `encoder_output` + candidate option sequences, predict a score
            for each option sequence.

            Parameters
            ----------
            encoder_output: torch.Tensor
                Output from the encoder through its forward pass.
                (batch_size, num_rounds, lstm_hidden_size)
        """

        options = batch["opt"].to(self.device)
        batch_size, num_rounds, num_options, max_sequence_length = options.size()
        options = options.view(batch_size * num_rounds * num_options, max_sequence_length)

        options_length = batch["opt_len"].to(self.device)
        options_length = options_length.view(batch_size * num_rounds * num_options)

        # Pick options with non-zero length (relevant for test split).
        nonzero_options_length_indices = options_length.nonzero().squeeze()
        nonzero_options_length = options_length[nonzero_options_length_indices]
        nonzero_options = options[nonzero_options_length_indices]

        # shape: (batch_size * num_rounds * num_options, max_sequence_length, word_embedding_size)
        # FOR TEST SPLIT, shape: (batch_size * 1, num_options, max_sequence_length, word_embedding_size)
        nonzero_options_embed = self.word_embed(nonzero_options)

        # shape: (batch_size * num_rounds * num_options, lstm_hidden_size)
        # FOR TEST SPLIT, shape: (batch_size * 1, num_options, lstm_hidden_size)
        _, (nonzero_options_embed, _) = self.option_rnn(nonzero_options_embed, nonzero_options_length)

        options_embed = torch.zeros(
            batch_size * num_rounds * num_options,
            nonzero_options_embed.size(-1),
            device=nonzero_options_embed.device,
        )
        options_embed[nonzero_options_length_indices] = nonzero_options_embed

        # Repeat encoder output for every option.
        # shape: (batch_size, num_rounds, num_options, lstm_hidden_size)
        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, num_options, 1)

        # Shape now same as `options`, can calculate dot product similarity.
        # shape: (batch_size * num_rounds * num_options, lstm_hidden_size)
        encoder_output = encoder_output.view(batch_size * num_rounds * num_options, self.hparams.lstm_hidden_size)

        # shape: (batch_size * num_rounds * num_options)
        scores = torch.sum(options_embed * encoder_output, 1)
        # shape: (batch_size, num_rounds, num_options)
        scores = scores.view(batch_size, num_rounds, num_options)

        return scores

    def criterion(self, disc_output, batch):
        disc_target = batch["ans_ind"].to(self.device)
        loss_disc = self.loss(disc_output.view(-1, disc_output.size(-1)), disc_target.view(-1))
        return loss_disc