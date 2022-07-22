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
    A simple reader for VisDial v1.0 dialog data. The json file must have the same structure as
    mentioned on ``https://visualdialog.org/data``.

    Path to a json file containing VisDial v1.0 train, val or test dialog data.
    original code is from https://github.com/yuleiniu/rva (CVPR, 2019)
"""

class DialogReader(object):
    """
        A simple reader for VisDial v1.0 dialog data. The json file must have the same structure as
        mentioned on ``https://visualdialog.org/data``.

        Parameters
        ----------
        dialogs_jsonpath : str
            Path to a json file containing VisDial v1.0 train, val or test dialog data.
    """

    def __init__(self, 
        dialogs_jsonpath: str, 
    ):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.nltk_tokenizer = word_tokenize
            
        with open(dialogs_jsonpath, "r") as visdial_file:
            visdial_data = json.load(visdial_file)
            self._split = visdial_data["split"]

            self.questions = visdial_data["data"]["questions"]
            self.answers = visdial_data["data"]["answers"]

            # Add empty question, answer at the end, useful for padding dialog rounds for test.
            self.questions.append("")
            self.answers.append("")

            # Image_id serves as key for all three dicts here.
            self.captions = {}
            self.dialogs = {}
            self.num_rounds = {}

            for dialog_for_image in visdial_data["data"]["dialogs"]:
                self.captions[dialog_for_image["image_id"]] = dialog_for_image["caption"]
                
                # Record original length of dialog, before padding.
                # 10 for train and val splits, 10 or less for test split.
                self.num_rounds[dialog_for_image["image_id"]] = len(dialog_for_image["dialog"])

                # Pad dialog at the end with empty question and answer pairs (for test split).
                while len(dialog_for_image["dialog"]) < 10:
                    dialog_for_image["dialog"].append({"question": -1, "answer": -1})

                # Add empty answer /answer options if not provided (for test split).
                for i in range(len(dialog_for_image["dialog"])):
                    if "answer" not in dialog_for_image["dialog"][i]:
                        dialog_for_image["dialog"][i]["answer"] = -1
                    if "answer_options" not in dialog_for_image["dialog"][i]:
                        dialog_for_image["dialog"][i]["answer_options"] = [-1] * 100

                self.dialogs[dialog_for_image["image_id"]] = dialog_for_image["dialog"]

        print(f"[{self._split}] Tokenizing questions...")
        self.questions_bert, self.questions_nltk = [], []
        for i in tqdm(range(len(self.questions))):
            self.questions_bert.append(self.bert_tokenizer.tokenize(self.questions[i] + "?"))
            self.questions_nltk.append(self.nltk_tokenizer(self.questions[i].lower() + "?"))

        print(f"[{self._split}] Tokenizing answers...")
        self.answers_bert, self.answers_nltk = [], []
        for i in tqdm(range(len(self.answers))):
            self.answers_bert.append(self.bert_tokenizer.tokenize(self.answers[i]))
            self.answers_nltk.append(self.nltk_tokenizer(self.answers[i].lower()))
        
        print(f"[{self._split}] Tokenizing captions...")
        for image_id, caption in tqdm(self.captions.items()):
            self.captions[image_id] = self.bert_tokenizer.tokenize(caption)

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:
        caption_for_image = self.captions[image_id]
        dialog_for_image = copy.deepcopy(self.dialogs[image_id])
        num_rounds = self.num_rounds[image_id]

        # Replace question and answer indices with actual word tokens.
        for i in range(len(dialog_for_image)):
            dialog_for_image[i]["question_bert"] = self.questions_bert[dialog_for_image[i]["question"]]
            dialog_for_image[i]["question_nltk"] = self.questions_nltk[dialog_for_image[i]["question"]]
            dialog_for_image[i]["answer_bert"] = self.answers_bert[dialog_for_image[i]["answer"]]
            dialog_for_image[i]["answer_nltk"] = self.answers_nltk[dialog_for_image[i]["answer"]]
            del dialog_for_image[i]["answer"]
            del dialog_for_image[i]["question"]
            for j, answer_option in enumerate(dialog_for_image[i]["answer_options"]):
                dialog_for_image[i]["answer_options"][j] = self.answers_nltk[answer_option]

        return {
            "image_id": image_id,
            "caption": caption_for_image,
            "dialog": dialog_for_image,
            "num_rounds": num_rounds
        }

    def keys(self) -> List[int]:
        return list(self.dialogs.keys())

    @property
    def split(self):
        return self._split


class DenseAnnotationReader(object):
    """
        A reader for dense annotations for val split. The json file must have the same structure as mentioned
        on ``https://visualdialog.org/data``.

        Parameters
        ----------
        dense_annotations_jsonpath : str
            Path to a json file containing VisDial v1.0
    """

    def __init__(self, dense_annotations_jsonpath: str):
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
            self._image_ids = [entry["image_id"] for entry in self._visdial_data]
            
    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, List]]:
        index = self._image_ids.index(image_id)
        # keys: {"image_id", "round_id", "gt_relevance"}
        return self._visdial_data[index]

    @property
    def split(self):
        # always
        return "val"
