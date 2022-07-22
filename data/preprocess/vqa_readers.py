import os
import copy
import json
from typing import Dict, List, Union
from nltk.tokenize import word_tokenize

from tqdm import tqdm
import h5py

import torch
from transformers import BertTokenizer

"""
    A simple reader for VQA v2 data. The json file must have the same structure as
    mentioned on ``https://visualqa.org/download.html``.
"""

class QuestionsReader(object):
    def __init__(self, questions_jsonpath: str):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.nltk_tokenizer = word_tokenize

        with open(questions_jsonpath, "r") as ques_file:
            ques_data = json.load(ques_file)
            self._split = ques_data["data_subtype"]
            self.questions = ques_data["questions"]

            # Question_id serves as key for all three dicts here.
            self.question = {}
            self.question_nltk = {}
            self.question_bert = {}
            self.image_id = {}
            
            for item in self.questions:
                question_id = item["question_id"]
                self.question[question_id] = item["question"]
                self.image_id[question_id] = item["image_id"]
            assert len(self.questions) == len(self.question)

        print(f"[{self._split}] Tokenizing questions...")
        for question_id, question in tqdm(self.question.items()):
            self.question_nltk[question_id] = self.bert_tokenizer.tokenize(question)
            self.question_bert[question_id] = self.nltk_tokenizer(question.lower())

    def __len__(self):
        return len(self.question)

    def __getitem__(self,
        question_id: int
    ) -> Dict[str, Union[int, List[str], int]]:
        return {
            "question_id": question_id,
            "question_nltk": self.question_nltk[question_id],
            "question_bert": self.question_bert[question_id],
            "image_id": self.image_id[question_id]
        }

    def keys(self) -> List[int]:
        return list(self.question_nltk.keys())

    @property
    def split(self):
        return self._split


class AnnotationsReader(object):
    def __init__(self, 
        annotations_jsonpath: str, 
        question_types_txtpath: str="data/v2/mscoco_question_types.txt"
    ):
        question_types = self._get_question_types(question_types_txtpath)
        answer_types = {"yes/no": 0, "number": 1, "other": 2}
        answer_confidence = {"yes": 0, "no": 1, "maybe": 2}
            
        with open(annotations_jsonpath, "r") as anno_file:
            anno_data = json.load(anno_file)
            self._split = anno_data["data_subtype"]
            
            self.annotations = anno_data["annotations"]

            # Question_id serves as key for all three dicts here.
            self.image_id = {}
            self.question_type = {}
            self.answer_type = {}
            self.gt_answer = {}
            self.conf_answer = {}
            
            for item in self.annotations:
                question_id = item["question_id"]
                self.image_id[question_id] = item["image_id"]
                self.question_type[question_id] = question_types[item["question_type"]]
                self.answer_type[question_id] = answer_types[item["answer_type"]]
                self.gt_answer[question_id] = item["multiple_choice_answer"]
                self.conf_answer[question_id] = item["answers"]
            assert len(self.annotations) == len(self.image_id)
            
            print(f"[{self._split}] Parsing answer confidence...")
            for question_id, conf_answer in tqdm(self.conf_answer.items()):
                for item in conf_answer:
                    item["answer_confidence"] = answer_confidence[item["answer_confidence"]]
            
    def __len__(self):
        return len(self.image_id)

    def __getitem__(self,
        question_id: int
    ) -> Dict[str, Union[int, int, int, int, str, List[Dict[str, Union[int, str, int]]]]]:
        return {
            "question_id": question_id,
            "image_id": self.image_id[question_id],
            "question_type": self.question_type[question_id],
            "answer_type": self.answer_type[question_id],
            "gt_answer": self.gt_answer[question_id],
            "conf_answer": self.conf_answer[question_id]
        }

    @property
    def split(self):
        return self._split
    
    def _get_question_types(self, question_types_txtpath):
        types_to_idx = {}
        with open(question_types_txtpath, "r") as types_file:
            lines = types_file.readlines()
            for idx, line in enumerate(lines):
                types_to_idx[line[:-1]] = idx
        return types_to_idx