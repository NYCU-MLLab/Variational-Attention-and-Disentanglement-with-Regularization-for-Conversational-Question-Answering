import os
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize

import sys
import collections
sys.path.append(os.getcwd() + "/../..")
from config.hparams import *

def count(word_counts, words):
    for word in words:
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    return word_counts

def bulid_vqa_wordcount_json(hparams):
    json_files = [
        hparams.questions_json % "train2014",
        hparams.questions_json % "val2014",
    ]

    word_counts = {}
    for json_file in json_files:

        with open(json_file, "r") as ques_file:
            ques_data = json.load(ques_file)
            split = ques_data["data_subtype"]
            questions = ques_data["questions"]
            
            print(f"[{split}] Questions word counting...")
            for item in tqdm(questions):
                words = word_tokenize(item["question"].lower())
                word_counts = count(word_counts, words)
                                  
    with open(hparams.vqa_word_counts_json, "w") as word_counts_file: 
        json.dump(word_counts, word_counts_file)

def combine_wordcount_json(hparams):
    with open(hparams.vqa_word_counts_json, "r") as word_counts_file:
        vqa_word_counts = json.load(word_counts_file)
    with open(hparams.visdial_word_counts_json, "r") as word_counts_file:
        visdial_word_counts = json.load(word_counts_file)

    word_counts = {}
    for word, count in visdial_word_counts.items():
        if word in vqa_word_counts:
            word_counts[word] = count + vqa_word_counts[word]
            del vqa_word_counts[word]
        else:
            word_counts[word] = count
    for word, count in vqa_word_counts.items():
        word_counts[word] = count

    with open(hparams.share_word_counts_json, "w") as word_counts_file: 
        json.dump(word_counts, word_counts_file)


if __name__ == "__main__":
    hparams = HPARAMS
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

    os.chdir("../../")
    if not os.path.exists(hparams.visdial_word_counts_json):
        raise FileNotFoundError(f"VisDial word counts do not exist at {hparams.visdial_word_counts_json}")
    
    print("VQA WORD COUNTING ...")
    bulid_vqa_wordcount_json(hparams)
    print(f"VQA word counts json saved in:\n\t{hparams.vqa_word_counts_json}\n")
    
    if not os.path.exists("data/share_text/"):
        os.mkdir("data/share_text/")
    
    combine_wordcount_json(hparams)
    print(f"Share word counts json saved in:\n\t{hparams.share_word_counts_json}\n")