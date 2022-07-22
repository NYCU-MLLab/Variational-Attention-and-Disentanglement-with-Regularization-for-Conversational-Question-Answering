import os
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from sklearn import manifold
from matplotlib import pyplot as plt

# import sys
# import collections
# sys.path.append(os.getcwd() + "/..")
# from config.hparams import *

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.vqa_dataset import VQADataset
from model.vqa.modules import VQAEncodeModule, VQADecodeModule
from model.vqa.model import EncoderDecoderModel

class VQAEvaluator(object):
    def __init__(self, hparams, split, load_pthpath, fig_dirpath):
        self.split = split
        # self.load_ep = load_pthpath[41:-4]
        self.hparams = hparams
        self.fig_dirpath = fig_dirpath
        torch.manual_seed(hparams.random_seed[0])
        torch.cuda.manual_seed_all(hparams.random_seed[0])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._build_dataloader(split)
        self._bulid_model()
        self._load_checkpoint(load_pthpath)

    def _build_dataloader(self, split):
        print("* Loading dataset ...\n")
        self.dataset = VQADataset(
            self.hparams,
            overfit=self.hparams.overfit,
            split=split, 
        )

        collate_fn = None
        if "dan" in self.hparams.img_feature_type:
            collate_fn = self.dataset.collate_fn

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.cpu_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn
        )
        print("\t# --------------------------------------------------")
        print("\t#   DATALOADER FINISHED")
        print("\t# --------------------------------------------------\n")

    def _bulid_model(self):
        print("* Building model ...\n")
        # mi_estimator = MIEstimator(self.hparams)
        vqa_encoder = VQAEncodeModule(self.hparams)
        vqa_decoder = VQADecodeModule(self.hparams, self.dataset.vocabulary)
        self.model = EncoderDecoderModel(vqa_encoder, vqa_decoder)
        self.model = self.model.to(self.device)        

    def _load_checkpoint(self, load_pthpath):
        checkpoint = torch.load(load_pthpath)
        self.model.encoder.textual_encoder.load_state_dict(checkpoint["textual_encoder"])
        self.model.encoder.visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        self.model.encoder.vis_ques_encoder.load_state_dict(checkpoint["vis_ques_encoder"])
        self.model.encoder.fusion_nn.load_state_dict(checkpoint["fusion_nn"])
        self.model.decoder.load_state_dict(checkpoint["decoder"])
        print("\t- Loaded model weights from {}".format(load_pthpath))
        print("\t# --------------------------------------------------")
        print("\t#   MODEL BULID FINISHED")
        print("\t# --------------------------------------------------\n")

    def run_acc(self):
        print("* Evaluate model accuracy ...")
        dataloader = itertools.chain(self.dataloader)
        total_iterations = len(self.dataset) // self.hparams.eval_batch_size + 1
        
        print("Total Iter:", total_iterations)
        tqdm_batch_iterator = tqdm(dataloader)

        cor_yn, cor_num, cor_oth = 0, 0, 0
        num_yn, num_num, num_oth = 0, 0, 0
        for i, batch in enumerate(tqdm_batch_iterator):

            with torch.no_grad():
                self.model.eval()
                _, _, word_scores = self.model(batch)
            _, pred_ans = torch.max(word_scores, dim=-1)
            pred_ans = pred_ans.view(-1)

            for i in range(len(pred_ans)):
                if batch["ans_type"][i] == 0: # yes/no
                    num_yn += 1
                    if pred_ans[i] == batch["gt_ans"][i]:
                        cor_yn += 1
                elif batch["ans_type"][i] == 1: # num
                    num_num += 1
                    if pred_ans[i] == batch["gt_ans"][i]:
                        cor_num += 1
                elif batch["ans_type"][i] == 2: # other
                    num_oth += 1
                    if pred_ans[i] == batch["gt_ans"][i]:
                        cor_oth += 1
        
        acc_yn = cor_yn / num_yn
        acc_num = cor_num / num_num
        acc_oth = cor_oth / num_oth
        acc_all = (cor_yn + cor_num + cor_oth) / (num_yn + num_num + num_oth)
        print("Acc of Yes/No: ", acc_yn)
        print("Acc of Number: ", acc_num)
        print("Acc of Other : ", acc_oth)
        print("Acc of All   : ", acc_all)

    def run_tSNE(self):
        print("* Evaluate latent distribution ...\n")
        dataloader = itertools.chain(self.dataloader)
        total_iterations = len(self.dataset) // self.hparams.eval_batch_size + 1
        
        print("Total Iter:", total_iterations)
        tqdm_batch_iterator = tqdm(dataloader)

        cont_record, type_record = [], []
        label_record = []
        for i, batch in enumerate(tqdm_batch_iterator):
            with torch.no_grad():
                self.model.eval()
                cont_emb, type_emb, _ = self.model(batch)

            cont_record.append(cont_emb.view(-1, cont_emb.size(-1)))
            type_record.append(type_emb.view(-1, type_emb.size(-1)))
            label_record.append(batch["ans_type"].view(-1))
            if i == 50:
                break
        cont_emb = (torch.cat(cont_record, dim=0).cpu()).numpy()
        type_emb = (torch.cat(type_record, dim=0).cpu()).numpy()
        label = (torch.cat(label_record, dim=0).cpu()).numpy()

        save_name = self.fig_dirpath + "/"
        self._plot_tSNE(cont_emb, label, save_name + "content.png")
        self._plot_tSNE(type_emb, label, save_name + "style.png")

    def _plot_tSNE(self, emb, label, save_name):
        X_tsne = manifold.TSNE(
            n_components=2,
            # perplexity=25,
            # n_iter=5000,
            init="random", 
            random_state=5, 
            verbose=1
        ).fit_transform(emb)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
        yn_x, yn_y = [], []
        num_x, num_y = [], []
        oth_x, oth_y = [], []
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            # plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]), 
            #         fontdict={"weight": "bold", "size": 9})
            if label[i] == 0:
                yn_x.append(X_norm[i, 0])
                yn_y.append(X_norm[i, 1])
            elif label[i] == 1:
                num_x.append(X_norm[i, 0])
                num_y.append(X_norm[i, 1])
            elif label[i] == 2:
                oth_x.append(X_norm[i, 0])
                oth_y.append(X_norm[i, 1])

        plt.scatter(yn_x, yn_y, facecolors='none', edgecolors='#d62728', marker='o', label="Yes/No")
        plt.scatter(num_x, num_y, facecolors='none', edgecolors='#1f77b4', marker='^', label="Number")
        plt.scatter(oth_x, oth_y, color='#2ca02c', marker="x", label="Other")
        plt.legend(loc="upper right")
        # plt.xticks([])
        # plt.yticks([])
        plt.savefig(save_name)

    def _to_words(self, predict_idx):
        bs, nr, seq_len = predict_idx.size()
        predict_words = []

        for i in range(bs):
            predict_words.append([])
            for j in range(nr):
                predict_words[i].append([])
                item = predict_idx[i][j].to("cpu")
                item = item.numpy()
                predict_words[i][j].append(self.val_dataset.vocabulary.to_words(item))
        return predict_words 

    def _record(self, cor_yn, cor_num, cor_oth, num_yn, num_num, num_oth):
        self.num_correct["yes/no"] += cor_yn
        self.num_correct["num"] += cor_num
        self.num_correct["other"] += cor_oth
        self.num_total["yes/no"] += num_yn
        self.num_total["num"] += num_num
        self.num_total["other"] += num_oth

    def _show_acc(self):
        acc_ys = self.num_correct["yes/no"] / self.num_total["yes/no"]
        acc_num = self.num_correct["num"] / self.num_total["num"]
        acc_oth = self.num_correct["other"] / self.num_total["other"]
        
        cor_all = self.num_correct["yes/no"] + self.num_correct["num"] + self.num_correct["other"]
        num_all = self.num_total["yes/no"] + self.num_total["num"] + self.num_total["other"]
        acc_all = cor_all / num_all
        print("ACCURACY : ")
        print(f"\tOVERALL  : {acc_all}")
        print(f"\tYES / NO : {acc_ys}")
        print(f"\tNUMBER   : {acc_num}")
        print(f"\tOTHER    : {acc_oth}")

    def run(self, batch, model):
        _, _, decoder_output = model(batch)
        _, pred_ans = torch.max(decoder_output, dim=-1)
        pred_ans = pred_ans.view(-1)

        cor_yn, cor_num, cor_oth = 0, 0, 0
        num_yn, num_num, num_oth = 0, 0, 0
        for i in range(len(pred_ans)):
            if batch["ans_type"][i] == 0: # yes/no
                num_yn += 1
                if pred_ans[i] == batch["gt_ans"][i]:
                    cor_yn += 1
            elif batch["ans_type"][i] == 1: # num
                num_num += 1
                if pred_ans[i] == batch["gt_ans"][i]:
                    cor_num += 1
            elif batch["ans_type"][i] == 2: # other
                num_oth += 1
                if pred_ans[i] == batch["gt_ans"][i]:
                    cor_oth += 1
        
        return cor_yn, cor_num, cor_oth, num_yn, num_num, num_oth
        
    def run_valid(self, checkpoint_pthpath):
        self._build_dataloader()
        self._bulid_model()
        self.num_correct = {"yes/no": 0, "num": 0, "other": 0}
        self.num_total = {"yes/no": 0, "num": 0, "other": 0}

        _, _, model_state_dict, _ = load_checkpoint(checkpoint_pthpath)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        tqdm_batch_iterator = tqdm(self.val_dataloader)
        for i, batch in enumerate(tqdm_batch_iterator):
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # print("\n-------------------------------------------------------------")
            # for key in list(batch.keys()):
            #     print("{:<15}{} {}".format(key, type(batch[key]), batch[key].size()))
            # print("-------------------------------------------------------------\n")
            
            with torch.no_grad():
                cor_yn, cor_num, cor_oth, num_yn, num_num, num_oth = self.run(batch, self.model)
                self._record(cor_yn, cor_num, cor_oth, num_yn, num_num, num_oth)

            # decoder_output = self.model.decoder.ques_gen(cont_emb, batch)
            # prob, pred = torch.max(decoder_output, dim=-1)
            # pred = batch["ques_out"]
            # word = self._to_words(pred)
            # for i in word:
            #     for r in i:
            #         print(r)
            #         print("-----------------------")
            self._show_acc()
        
        self._show_acc()


# def main(args):
#     hparams = HPARAMS
#     hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

#     os.chdir("../")

#     for i in range(22, 38):
#         load_pthpath = "checkpoints/testTypeRegMItune/checkpoint_" + str(i) + ".pth"
#         vqa_eval = VQAEvaluator(hparams, args.split, load_pthpath)
#         # vqa_eval.run_acc()
#         vqa_eval.run_tSNE()
#         torch.cuda.empty_cache()

# if __name__ == "__main__":
#     arg_parser = argparse.ArgumentParser(description="VQA Evaluation")
#     arg_parser.add_argument("--load_pth", dest="load_pth", type=str, help="Checkpoint file")
#     arg_parser.add_argument("--split", dest="split", type=str,  help="Evaluate split")
#     # arg_parser.add_argument("--save_folder", dest="save_folder", type=str, default="results/vqa/fig/", help="Save dir")
    
#     args = arg_parser.parse_args()
#     main(args)



