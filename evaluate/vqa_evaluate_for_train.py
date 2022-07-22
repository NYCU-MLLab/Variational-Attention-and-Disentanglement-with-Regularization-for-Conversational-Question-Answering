import os
import itertools
import numpy as np
from tqdm import tqdm
from sklearn import manifold
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.vqa_dataset import VQADataset
from model.vqa.modules import VQAEncodeModule, VQADecodeModule
from model.vqa.model import EncoderDecoderModel

class VQAEvaluator(object):
    def __init__(self, hparams, split, epoch, load_pthpath, writer):
        self.split = split
        self.epoch = epoch
        self.hparams = hparams
        self.writer = writer
        torch.manual_seed(hparams.random_seed[0])
        torch.cuda.manual_seed_all(hparams.random_seed[0])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.acc_dic = {
            "num_yn": 0, "cor_yn": 0,
            "num_num": 0, "cor_num": 0,
            "num_oth": 0, "cor_oth": 0,
        }
        self.cont_record, self.type_record = [], []
        self.label_record = []

        self._build_dataloader(split)
        self._bulid_model()
        self._load_checkpoint(load_pthpath)

        if not os.path.exists(hparams.fig_dirpath):
            os.makedirs(hparams.fig_dirpath)

    def _build_dataloader(self, split):
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
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

    def _bulid_model(self):
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
        self.model.decoder.type_predictor.load_state_dict(checkpoint["type_predictor"])
        self.model.decoder.ques_gen.load_state_dict(checkpoint["ques_gen"])
        self.model.decoder.ans_predictor.load_state_dict(checkpoint["ans_predictor"])

    def run(self):
        dataloader = itertools.chain(self.dataloader)
        total_iterations = len(self.dataset) // self.hparams.eval_batch_size + 1
    
        print(f"Evaluating for epoch: {self.epoch}", "\tTotal Iter:", total_iterations)
        tqdm_batch_iterator = tqdm(dataloader)
        num_draw_points = 0
        for i, batch in enumerate(tqdm_batch_iterator):

            with torch.no_grad():
                self.model.eval()
                cont_emb, type_emb, word_scores = self.model(batch)

            _, pred_ans = torch.max(word_scores, dim=-1)
            pred_ans = pred_ans.view(-1)

            for i in range(len(pred_ans)):
                if batch["ans_type"][i] == 0: # yes/no
                    self.acc_dic["num_yn"] += 1
                    if pred_ans[i] == batch["gt_ans"][i]:
                        self.acc_dic["cor_yn"] += 1
                elif batch["ans_type"][i] == 1: # num
                    self.acc_dic["num_num"] += 1
                    if pred_ans[i] == batch["gt_ans"][i]:
                        self.acc_dic["cor_num"] += 1
                elif batch["ans_type"][i] == 2: # other
                    self.acc_dic["num_oth"] += 1
                    if pred_ans[i] == batch["gt_ans"][i]:
                        self.acc_dic["cor_oth"] += 1

            if num_draw_points <= 5000:
                self.cont_record.append(cont_emb.view(-1, cont_emb.size(-1)))
                self.type_record.append(type_emb.view(-1, type_emb.size(-1)))
                self.label_record.append(batch["ans_type"].view(-1))
                num_draw_points += self.hparams.eval_batch_size

        self._run_acc()
        self._run_tSNE()

    def _run_acc(self):
        acc_yn = self.acc_dic["cor_yn"] / self.acc_dic["num_yn"]
        acc_num = self.acc_dic["cor_num"] / self.acc_dic["num_num"]
        acc_oth = self.acc_dic["cor_oth"] / self.acc_dic["num_oth"]
        acc_all = (self.acc_dic["cor_yn"] + self.acc_dic["cor_num"] + self.acc_dic["cor_oth"]) \
                / (self.acc_dic["num_yn"] + self.acc_dic["num_num"] + self.acc_dic["num_oth"])
        # print("\tAcc of Yes/No: ", acc_yn)
        # print("\tAcc of Number: ", acc_num)
        # print("\tAcc of Other : ", acc_oth)
        # print("\tAcc of All   : ", acc_all)

        self.writer.add_scalar("Accuracy/Val Overall", acc_all, self.epoch)
        self.writer.add_scalar("Accuracy/Val Yes_No", acc_yn, self.epoch)
        self.writer.add_scalar("Accuracy/Val Number", acc_num, self.epoch)
        self.writer.add_scalar("Accuracy/Val Other", acc_oth, self.epoch)

    def _run_tSNE(self):
        cont_emb = (torch.cat(self.cont_record, dim=0).cpu()).numpy()
        type_emb = (torch.cat(self.type_record, dim=0).cpu()).numpy()
        label = (torch.cat(self.label_record, dim=0).cpu()).numpy()

        save_name = self.hparams.fig_dirpath + "/" + self.split + "_" + str(self.epoch) + "_"
        self._plot_tSNE(cont_emb, label, save_name + "cont.png")
        self._plot_tSNE(type_emb, label, save_name + "type.png")

    def _plot_tSNE(self, emb, label, save_name):
        X_tsne = manifold.TSNE(
            n_components=2, 
            init="random", 
            random_state=5, 
            verbose=1
        ).fit_transform(emb)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]), 
                    fontdict={"weight": "bold", "size": 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_name)

