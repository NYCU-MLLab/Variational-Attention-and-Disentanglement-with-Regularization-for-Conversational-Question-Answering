import os
import json
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from transformers import BertTokenizer
from data.preprocess.init_glove import Vocabulary
from data.preprocess.vqa_readers import QuestionsReader, AnnotationsReader

class BertVocabulary(object):
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    PAD_TOKEN = "[PAD]"
    
    CLS_INDEX = 101
    SEP_INDEX = 102
    PAD_INDEX = 0

    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def to_indices(self, words: List[str]) -> List[int]:
        return self.bert_tokenizer.convert_tokens_to_ids(words)
    
    def add_special_tokens(self, sent_st, sent_nd=None):
        if sent_nd is None:
            return self.bert_tokenizer.build_inputs_with_special_tokens(sent_st)
        else:
            return self.bert_tokenizer.build_inputs_with_special_tokens(sent_st, sent_nd)
    
    def get_segment_ids(self, sent_st, sent_nd=None):
        if sent_nd is None:
            return self.bert_tokenizer.create_token_type_ids_from_sequences(sent_st)
        else:
            return self.bert_tokenizer.create_token_type_ids_from_sequences(sent_st, sent_nd)


class VQADataset(Dataset):
    def __init__(
        self,
        hparams,
        questions_jsonpath: str, 
        annotations_jsonpath: Optional[str] = None,
        overfit: bool = False,
    ):
        super().__init__()
        self.hparams = hparams
        
        self.questions_reader = QuestionsReader(questions_jsonpath)
        
        # Keep a list of question_ids as primary keys to access data.
        self.question_ids = list(self.questions_reader.question.keys())
        if overfit:
            self.question_ids = self.question_ids[:5]

        if annotations_jsonpath is not None:
            self.annotations_reader = AnnotationsReader(annotations_jsonpath)
            if "train" in self.questions_reader.split:
                top_ans = self._get_top_answer(hparams.num_answers)
                self.ans_to_id = {w: i for i, w in enumerate(top_ans)}
                self.id_to_ans = {i: w for i, w in enumerate(top_ans)}

                with open(hparams.answer_to_index_json, "w") as json_file: 
                    json.dump(self.ans_to_id, json_file)
                with open(hparams.index_to_answer_json, "w") as json_file: 
                    json.dump(self.id_to_ans, json_file)
                print(f"Answer to index json saved in:\n\t{hparams.answer_to_index_json}\n")
                print(f"Index to answer json saved in:\n\t{hparams.index_to_answer_json}\n")
            else:
                with open(hparams.answer_to_index_json, "r") as json_file:
                    self.ans_to_id = json.load(json_file)
                with open(hparams.index_to_answer_json, "r") as json_file:
                    self.id_to_ans = json.load(json_file)

            # Filter question, which isn't in the top answers.
            self.question_ids = self._filter_question()
        else:
            self.annotations_reader = None
            with open(hparams.answer_to_index_json, "r") as json_file:
                self.ans_to_id = json.load(json_file)
            with open(hparams.index_to_answer_json, "r") as json_file:
                self.id_to_ans = json.load(json_file)
        
        self.vocabulary = Vocabulary(
            word_counts_path=hparams.share_word_counts_json, 
            min_count=hparams.vocab_min_count,
        )
        self.bert_vocabulary = BertVocabulary()
        
    
    @property
    def split(self):
        return self.questions_reader.split 

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, index):
        # Get question_id, which serves as a primary key for current instance.
        question_id = self.question_ids[index]

        # Retrieve instance for this question_id using json reader.
        question_instance = self.questions_reader[question_id]
        question_nltk = question_instance["question_nltk"]
        question_bert = question_instance["question_bert"]
        
        question = self.bert_vocabulary.to_indices(
            [self.bert_vocabulary.CLS_TOKEN]
            + question_bert
            + [self.bert_vocabulary.SEP_TOKEN]
        )
        question_gen = self.vocabulary.to_indices(
            [self.vocabulary.SOS_TOKEN]
            + question_nltk
            + [self.vocabulary.EOS_TOKEN]
        )
        question_in = question_gen[:-1]
        question_out = question_gen[1:]
        
        question, question_length = self._pad_sequences(
            [question], 
            max_sequence_length = self.hparams.max_question_length,
            bert_format = True,
        )
        question_in, _ = self._pad_sequences(
            [question_in], 
            max_sequence_length = self.hparams.max_question_length,
        )
        question_out, _ = self._pad_sequences(
            [question_out], 
            max_sequence_length = self.hparams.max_question_length,
        )
       
        item = {}
        item["ques"] = question.long()
        item["ques_in"] = question_in.long()
        item["ques_out"] = question_out.long()
        item["ques_len"] = torch.tensor(question_length).long()
        item["ques_id"] = torch.tensor(question_instance["question_id"]).long()
        item["img_id"] = torch.tensor(question_instance["image_id"]).long()

        if self.annotations_reader is not None:
            annotation_instance = self.annotations_reader[question_id]
            assert question_instance["image_id"] == annotation_instance["image_id"]

            question_type = annotation_instance["question_type"]
            answer_type = annotation_instance["answer_type"]
            gt_answer = annotation_instance["gt_answer"]
            gt_answer = self.ans_to_id[gt_answer]
            conf_answer = annotation_instance["conf_answer"]
            
            item["gt_ans"] = torch.tensor(gt_answer).long()
            item["ques_type"] = torch.tensor(question_type).long()
            item["ans_type"] = torch.tensor(answer_type).long()

            conf_ans_in, conf_score = [], []
            for dic_item in conf_answer:
                conf_score.append(dic_item["answer_confidence"])
    
            item["conf_ans_score"] = torch.tensor(conf_score).long()
        return item        
    
    def _pad_sequences(
        self, 
        sequences: List[List[int]],
        max_sequence_length: int,
        bert_format: bool=False, 
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
                    tensor_sequences.append(
                        torch.tensor([self.bert_vocabulary.CLS_INDEX]
                            + sequence
                            + [self.bert_vocabulary.SEP_INDEX]
                        )
                    )
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
    
    def _get_top_answer(self, num_ans):
        ans_counts = {}
        for ques_id in self.question_ids:
            annotation_instance = self.annotations_reader[ques_id]
            ans = annotation_instance["gt_answer"]
            ans_counts[ans] = ans_counts.get(ans, 0) + 1
        
        cw = sorted(ans_counts.items(), key=lambda kv: kv[1], reverse=True)
        top_ans = []
        for i in range(num_ans):
            top_ans.append(cw[i][0])
        print("Top answers: {}/{}".format(num_ans, len(cw)))
        return top_ans

    def _filter_question(self):
        filter_ids = []
        for ques_id in self.question_ids:
            annotation_instance = self.annotations_reader[ques_id]
            if annotation_instance["gt_answer"] in self.ans_to_id:
                filter_ids.append(ques_id)
        print("Filter questions: {}/{}".format(len(filter_ids), len(self.question_ids)))
        return filter_ids