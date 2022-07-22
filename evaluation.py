import os
import copy
import json
import torch
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
from typing import Dict, List, Union

from config.hparams import *
from pretrain import VQAModel
from train import VisDialModel
from evaluate.vqa_evaluate import VQAEvaluator
from evaluate.visdial_evaluate import MultiEvaluation


def evaluate_vqa(args, hparams):    
    load_pthpath = hparams.save_dirpath + "/vqa/" + args.folder + "/checkpoint_vqa.pth"
    fig_dirpath = hparams.fig_dirpath + "/vqa/" + args.folder
    vqa_eval = VQAEvaluator(hparams, args.split, load_pthpath, fig_dirpath)
    vqa_eval.run_acc()
    vqa_eval.run_tSNE()
    torch.cuda.empty_cache()

def evaluate_visdial(args, hparams):
    load_pthpath = hparams.save_dirpath + "/vqa/" + "VQA_FOLDER" + "/checkpoint_vqa.pth"
    fig_dirpath = hparams.fig_dirpath + "/vqa/" + args.folder
    vqa_eval = VQAEvaluator(hparams, args.split, load_pthpath, fig_dirpath)
    vqa_eval.run_evaluate()
            

def main(args):
    hparams = HPARAMS
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

    if args.evaluate == "vqa":
        print("VQA EVALUATION")
        evaluate_vqa(args, hparams)
    else:
        print("VISUAL DIALOG EVALUATION")
        evaluate_visdial(args, hparams)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Variational Attention and Disentanglement with Regularization for Conversational Question Answering")
    arg_parser.add_argument("--evaluate", dest="evaluate", type=str, default="vqa", help="Evaluation model")
    arg_parser.add_argument("--eval_split", dest="split", type=str, default="val", help="Evaluation split")
    arg_parser.add_argument("--folder", dest="folder", type=str, help="Evaluation folder")
    
    args = arg_parser.parse_args()
    main(args)