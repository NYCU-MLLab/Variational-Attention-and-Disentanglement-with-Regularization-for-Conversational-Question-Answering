import os
import json
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data.preprocess.init_glove import Vocabulary
from data.preprocess.vqa_preprocessors import BertVocabulary
from data.preprocess.visdial_readers import (
    DialogReader,
    DenseAnnotationReader,
)

class VisDialDataset(Dataset):
    """
        A full representation of VisDial v1.0 (train/val/test) dataset. According
        to the appropriate split, it returns dictionary of question, image,
        history, ground truth answer, answer options, dense annotations etc.
    """

    def __init__(
        self,
        hparams,
        dialog_jsonpath: str, 
        dense_annotation_jsonpath: Optional[str]=None,
        overfit: bool=False,
        return_options: bool=True,
    ):
        super().__init__()
        self.hparams = hparams
        self.return_options = return_options
        self.dialogs_reader = DialogReader(dialog_jsonpath)
        
        if "val" in self.split and dense_annotation_jsonpath is not None:
            self.annotations_reader = DenseAnnotationReader(
                dense_annotation_jsonpath
            )
        else:
            self.annotations_reader = None

        self.bert_vocabulary = BertVocabulary()
        self.vocabulary = Vocabulary(
            word_counts_path=hparams.share_word_counts_json, 
            min_count=hparams.vocab_min_count,
        )

        # Keep a list of image_ids as primary keys to access data.
        self.image_ids = list(self.dialogs_reader.dialogs.keys())
        if overfit:
            self.image_ids = self.image_ids[:5]

    @property
    def split(self):
        return self.dialogs_reader.split 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Get image_id, which serves as a primary key for current instance.
        image_id = self.image_ids[index]

        # Retrieve instance for this image_id using json reader.
        visdial_instance = self.dialogs_reader[image_id]
        caption = visdial_instance["caption"]
        dialog = visdial_instance["dialog"]

        # Convert word tokens of caption, question, answer to integers.(bert format)
        # Convert word tokens of answer, answer options to integers for generative
        # stage.(vocabulary format)
        caption = self.bert_vocabulary.add_special_tokens(
            self.bert_vocabulary.to_indices(caption)
        )

        for i in range(len(dialog)):
            dialog[i]["question_bert"] = self.bert_vocabulary.add_special_tokens(
                self.bert_vocabulary.to_indices(dialog[i]["question_bert"])
            )
            dialog[i]["qa_bert"] = self.bert_vocabulary.add_special_tokens(
                self.bert_vocabulary.to_indices(dialog[i]["question_bert"]), 
                self.bert_vocabulary.to_indices(dialog[i]["answer_bert"])
            )
            dialog[i]["qa_segment_ids"] = self.bert_vocabulary.get_segment_ids(
                self.bert_vocabulary.to_indices(dialog[i]["question_bert"]), 
                self.bert_vocabulary.to_indices(dialog[i]["answer_bert"])
            )

            dialog[i]["question_nltk"] = self.vocabulary.to_indices(
                [self.vocabulary.SOS_TOKEN]
                + dialog[i]["question_nltk"]
                + [self.vocabulary.EOS_TOKEN]
            )
            dialog[i]["answer_nltk"] = self.vocabulary.to_indices(
                [self.vocabulary.SOS_TOKEN]
                + dialog[i]["answer_nltk"]
                + [self.vocabulary.EOS_TOKEN]
            )

            if self.return_options:
                for j in range(len(dialog[i]["answer_options"])):
                    dialog[i]["answer_options"][j] = self.vocabulary.to_indices(
                        [self.vocabulary.SOS_TOKEN]
                        + dialog[i]["answer_options"][j]
                        + [self.vocabulary.EOS_TOKEN]
                    )
    
        questions_bert, questions_in, questions_out = [], [], []
        qa_bert, qa_segment_ids, answers_in, answers_out = [], [], [], []
        for dialog_round in dialog:
            questions_bert.append(dialog_round["question_bert"])
            questions_in.append(dialog_round["question_nltk"][:-1])
            questions_out.append(dialog_round["question_nltk"][1:])
            qa_bert.append(dialog_round["qa_bert"])
            qa_segment_ids.append(dialog_round["qa_segment_ids"])
            answers_in.append(dialog_round["answer_nltk"][:-1])
            answers_out.append(dialog_round["answer_nltk"][1:])
    
        questions, question_lengths = self._pad_sequences(
            questions_bert,
            max_sequence_length=self.hparams.max_question_length,
            bert_format=True,
        )
        qa_pairs, _ = self._pad_sequences(
            qa_bert,
            max_sequence_length=self.hparams.max_question_length+self.hparams.max_answer_length,
            bert_format=True,
        )
        qa_segment, _ = self._pad_sequences(
            qa_segment_ids,
            max_sequence_length=self.hparams.max_question_length+self.hparams.max_answer_length,
            bert_format=True,
            segment_format=True,
        )
        history, history_segment, history_lengths = self._get_history(
            caption,
            qa_bert,
            qa_segment_ids,
        )

        # For generative stage.
        answers_in, answer_lengths = self._pad_sequences(
            answers_in, 
            max_sequence_length=self.hparams.max_answer_length,
        )
        answers_out, _ = self._pad_sequences(
            answers_out, 
            max_sequence_length=self.hparams.max_answer_length,
        )
        questions_in, _ = self._pad_sequences(
            questions_in, 
            max_sequence_length=self.hparams.max_question_length,
        )
        questions_out, _ = self._pad_sequences(
            questions_out, 
            max_sequence_length=self.hparams.max_question_length,
        )

        # Collect everything as tensors for ``collate_fn`` of dataloader to
        # work seamlessly questions, history, etc. are converted to
        # LongTensors, for nn.Embedding input.
        item = {}
        item["img_ids"] = torch.tensor(image_id).long()
        item["hist"] = history.long()
        item["hist_seg"] = history_segment.long()
        item["ques"] = questions.long()
        item["qa_pairs"] = qa_pairs.long()
        item["qa_seg"] = qa_segment.long()
        item["ques_in"] = questions_in.long()
        item["ques_out"] = questions_out.long()
        item["ans_in"] = answers_in.long()
        item["ans_out"] = answers_out.long()
        item["ques_len"] = torch.tensor(question_lengths).long()
        item["ans_len"] = torch.tensor(answer_lengths).long()
        item["hist_len"] = torch.tensor(history_lengths).long()
        item["num_rounds"] = torch.tensor(visdial_instance["num_rounds"]).long()

        if self.return_options:
            # For generative stage
            answer_options_in, answer_options_out = [], []
            for dialog_round in dialog:
                options, option_lengths = self._pad_sequences(
                    [
                        option[:-1]
                        for option in dialog_round["answer_options"]
                    ], 
                    max_sequence_length=self.hparams.max_answer_length,
                )
                answer_options_in.append(options)

                options, _ = self._pad_sequences(
                    [
                        option[1:]
                        for option in dialog_round["answer_options"]
                    ], 
                    max_sequence_length = self.hparams.max_answer_length,
                )
                answer_options_out.append(options)

            answer_options_in = torch.stack(answer_options_in, 0)
            answer_options_out = torch.stack(answer_options_out, 0)

            item["opt_in"] = answer_options_in.long()
            item["opt_out"] = answer_options_out.long()

            answer_options = []
            answer_option_lengths = []
            for dialog_round in dialog:
                options, option_lengths = self._pad_sequences(
                    dialog_round["answer_options"], 
                    max_sequence_length=self.hparams.max_answer_length,
                )
                answer_options.append(options)
                answer_option_lengths.append(option_lengths)
            answer_options = torch.stack(answer_options, 0)

            item["opt"] = answer_options.long()
            item["opt_len"] = torch.tensor(answer_option_lengths).long()

            if "test" not in self.split:
                answer_indices = [
                    dialog_round["gt_index"] for dialog_round in dialog
                ]
                item["ans_ind"] = torch.tensor(answer_indices).long()

        # Gather dense annotations.
        if "val" in self.split:
            dense_annotations = self.annotations_reader[image_id]
            item["gt_relevance"] = torch.tensor(
                dense_annotations["gt_relevance"]
            ).float()
            item["round_id"] = torch.tensor(
                dense_annotations["round_id"]
            ).long()

        return item

        # -------------------------------------------------------------------------
        # collate function utilized by dataloader for batching
        # -------------------------------------------------------------------------

    def _pad_sequences(
        self, 
        sequences: List[List[int]],
        max_sequence_length: int,
        bert_format: bool=False, 
        segment_format: bool=False,
    ) -> (torch.tensor, List[int]):
        """
            Given tokenized sequences (either questions, answers or answer
            options, tokenized in ``__getitem__``), padding them to maximum
            specified sequence length. Return as a tensor of size
            ``(*, max_sequence_length)``.

            This method is only called in ``__getitem__``, chunked out separately
            for readability.

            Parameters
            ----------
            sequences : List[List[int]]
                List of tokenized sequences, each sequence is typically a
                List[int].

            Returns
            -------
            torch.Tensor, List[int]
                Tensor of sequences padded to max length, and length of sequences
                before padding.
        """
        sequence_lengths, tensor_sequences = [], []

        # Pad all sequences to max_sequence_length.
        maxpadded_sequences = torch.full(
            (len(sequences), max_sequence_length),
            fill_value=self.vocabulary.PAD_INDEX,
        )

        if bert_format:
            for sequence in sequences:
                sequence = sequence[1: -1]
                sequence = sequence[:max_sequence_length-2]
                sequence_lengths.append(len(sequence))
                if len(sequence) != 0:
                    if not segment_format:
                        tensor_sequences.append(
                            torch.tensor([self.bert_vocabulary.CLS_INDEX]
                                + sequence
                                + [self.bert_vocabulary.SEP_INDEX]
                            )
                        )
                    else:
                        tensor_sequences.append(torch.tensor([0] + sequence + [1]))
                else:
                    tensor_sequences.append(torch.tensor([]))
        else:
            for sequence in sequences:
                sequence = sequence[:max_sequence_length-1]
                sequence_lengths.append(len(sequence))
                tensor_sequences.append(torch.tensor(sequence))

        padded_sequences = pad_sequence(
            tensor_sequences,
            batch_first=True,
            padding_value=self.vocabulary.PAD_INDEX,
        )
        maxpadded_sequences[:, :padded_sequences.size(1)] = padded_sequences
        return maxpadded_sequences, sequence_lengths

    def _get_history(
        self,
        caption: List[int],
        qa_bert: List[List[int]],
        qa_segment_ids: List[List[int]],
    ) -> (torch.tensor, List[int]):
        max_sequence_length = self.hparams.max_question_length + self.hparams.max_answer_length
        
        # Allow length of caption equivalent to a concatenated QA pair.
        if len(caption) > max_sequence_length:
            caption = caption[: max_sequence_length-1]
            caption += [self.bert_vocabulary.SEP_INDEX]
        
        # History for first round is caption.
        # History for each round is QA pair of previous round.
        history, history_segment_id = [], []
        history.append(caption)
        history_segment_id.append([0] * len(caption))
        for i in range(len(qa_bert)):
            if len(qa_bert[i]) > max_sequence_length:
                qa_bert[i] = qa_bert[i][: max_sequence_length-1]
                qa_bert[i] += [self.bert_vocabulary.SEP_INDEX]
                qa_segment_ids[i][: max_sequence_length]
            history.append(qa_bert[i])
            history_segment_id.append(qa_segment_ids[i])
        # Drop last entry from history (there's no eleventh question).
        history = history[:-1] # cap, qa1, qa2, ..., qa9
        history_segment_id = history_segment_id[:-1]

        # Concatenated_history has similar structure as history, except it
        # contains concatenated QA pairs from previous rounds.
        collected_history, collected_segment = [], []
        k = 1
        for i in range(1, len(history) + 1):
            round_history, round_segment = [], []
            round_history.append(history[0])
            round_segment.append(history_segment_id[0])
            if i > self.hparams.max_round_history + 1:
                k += 1
            for j in range(k, i):
                round_history.append(history[j])
                round_segment.append(history_segment_id[j])
            collected_history.append(round_history)
            collected_segment.append(round_segment)    
        history_lengths = [len(round_history) for round_history in collected_history]

        concatenated_history, concatenated_segment = [], []
        for i in range(len(collected_history)):
            while len(collected_history[i]) < self.hparams.max_round_history + 1:
                collected_history[i].append([])
                collected_segment[i].append([])
            assert len(collected_history[i]) == self.hparams.max_round_history + 1
            assert len(collected_segment[i]) == self.hparams.max_round_history + 1
            concatenated_history += collected_history[i]
            concatenated_segment += collected_segment[i]
            
        concatenated_history, _ = self._pad_sequences(
            concatenated_history, 
            max_sequence_length=max_sequence_length,
            bert_format=True,
        )
        concatenated_segment, _ = self._pad_sequences(
            concatenated_segment, 
            max_sequence_length=max_sequence_length,
            bert_format=True,
            segment_format=True,
        )
        
        concatenated_history = concatenated_history.view(
            len(history), 
            self.hparams.max_round_history + 1,
            -1
        )
        concatenated_segment = concatenated_segment.view(
            len(history), 
            self.hparams.max_round_history + 1,
            -1
        )
        return concatenated_history, concatenated_segment, history_lengths