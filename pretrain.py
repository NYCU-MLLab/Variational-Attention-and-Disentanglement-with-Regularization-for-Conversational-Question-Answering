import os
import itertools
import numpy as np
from tqdm import tqdm
from bisect import bisect
from datetime import datetime

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.vqa_dataset import VQADataset
from model.vqa.modules import (
    VQAEncodeModule, VQADecodeModule,
    MIEstimator, Regularizer
)
from model.vqa.model import EncoderDecoderModel
from model.vqa.checkpointing import CheckpointManager
from evaluate.vqa_evaluate import VQAEvaluator


class VQAModel(object):
    def __init__(self, hparams):
        self.hparams = hparams        
        torch.manual_seed(hparams.random_seed[0])
        torch.cuda.manual_seed_all(hparams.random_seed[0])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_dataloader(self):

        # ====================================================================
        #   SETUP DATASET, DATALOADER
        # ====================================================================

        print("* Loading dataset ...\n")
        self.train_dataset = VQADataset(
            self.hparams,
            overfit=self.hparams.overfit,
            split="train", 
        )
        
        collate_fn = None
        if "dan" in self.hparams.img_feature_type:
            collate_fn = self.train_dataset.collate_fn

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.cpu_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
    
        print("\t# --------------------------------------------------")
        print("\t#   DATALOADER FINISHED")
        print("\t# --------------------------------------------------\n")

    def _build_model(self):

        # ====================================================================
        #   MODELS : Mutual Information Estimator,
        #            VQA Encoder,
        #            VQA Decoder
        # ====================================================================

        print("* Building model ...\n")
        self.mi_estimator = MIEstimator(self.hparams).to(self.device)
        vqa_encoder = VQAEncodeModule(self.hparams)
        vqa_decoder = VQADecodeModule(self.hparams, self.train_dataset.vocabulary)
        self.model = EncoderDecoderModel(vqa_encoder, vqa_decoder).to(self.device)
        self.regularizer = Regularizer()
                
        self.iterations = len(self.train_dataset) // self.hparams.train_batch_size + 1
    
        # ====================================================================
        #   OPTIMIZER
        # ====================================================================

        self.mi_optimizer = optim.Adam(self.mi_estimator.parameters(), lr=self.hparams.initial_lr)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.initial_lr)
            
        print("\t# --------------------------------------------------")
        print("\t#   MODEL BULID FINISHED")
        print("\t# --------------------------------------------------\n")

    def _setup_training(self):
        print("* Setup training ...\n")
        self.checkpoint_manager = CheckpointManager(
            self.mi_estimator,
            self.model.encoder.textual_encoder,
            self.model.encoder.visual_encoder,
            self.model.encoder.vis_ques_encoder,
            self.model.encoder.fusion_nn,
            self.model.decoder.type_predictor,
            self.model.decoder.ques_gen,
            self.model.decoder.ans_predictor,
            self.hparams.save_dirpath, 
        )
        self.writer = SummaryWriter(self.hparams.result_dirpath)
        self.writer_dict = {
            "mi_estimate": 0.0,
            "mi_nll": 0.0,
            "cont_loss": 0.0,
            "type_loss": 0.0,
            "reg_loss": 0.0,
            "vqa_loss": 0.0,
            "acc_all": 0.0,
            "acc_yn": 0.0,
            "acc_num": 0.0,
            "acc_oth": 0.0,
        }
        self.start_epoch = 1

        print("\t# --------------------------------------------------")
        print("\t#   SETUP TRINING FINISHED")
        print("\t# --------------------------------------------------\n")

    def _record(self, iterations: int):
        self.writer.add_scalar("Objective/MI Estimate", self.writer_dict["mi_estimate"], iterations)
        self.writer.add_scalar("Objective/MI NLL", self.writer_dict["mi_nll"], iterations)
        self.writer.add_scalar("Objective/Cont loss", self.writer_dict["cont_loss"], iterations)
        self.writer.add_scalar("Objective/Type loss", self.writer_dict["type_loss"], iterations)
        self.writer.add_scalar("Objective/Reg loss", self.writer_dict["reg_loss"], iterations)
        self.writer.add_scalar("Objective/Answer loss", self.writer_dict["vqa_loss"], iterations)
        self.writer.add_scalar("Accuracy/Overall", self.writer_dict["acc_all"], iterations)
        self.writer.add_scalar("Accuracy/Yes_No", self.writer_dict["acc_yn"], iterations)
        self.writer.add_scalar("Accuracy/Number", self.writer_dict["acc_num"], iterations)
        self.writer.add_scalar("Accuracy/Other", self.writer_dict["acc_oth"], iterations)

    def _beta_scheduler(self, beta):
        beta += 0.1
        if beta >= 1:
            return 1
        else:
            return beta

    def _run_eval(self, batch):
        with torch.no_grad():
            self.model.eval()
            _, _, word_scores = self.model(batch)
        _, pred_ans = torch.max(word_scores, dim=-1)
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
        
        acc_yn = cor_yn / num_yn
        acc_num = cor_num / num_num
        acc_oth = cor_oth / num_oth
        acc_all = (cor_yn + cor_num + cor_oth) / (num_yn + num_num + num_oth)
        return acc_yn, acc_num, acc_oth, acc_all

    def train(self):
        self._build_dataloader()
        self._build_model()
        self._setup_training()
    
        print("* Model Training ...\n")
        beta = self.hparams.initial_beta
        iterations = 0
        for epoch in range(self.start_epoch , self.hparams.num_epochs):
            
            # ====================================================================
            #   ON EPOCH START  (combine dataloaders if training on train + val)
            # ====================================================================
            
            combined_dataloader = itertools.chain(self.train_dataloader)
            
            print(f"Training for epoch: {epoch}", "\tTotal Iter:", self.iterations)
            tqdm_batch_iterator = tqdm(combined_dataloader)

            for i, batch in enumerate(tqdm_batch_iterator):
                # for key in batch:
                #     batch[key] = batch[key].to(self.device)
                
                # print("\n-------------------------------------------------------------")
                # for key in list(batch.keys()):
                #     print("{:<15}{} {}".format(key, type(batch[key]), batch[key].size()))
                # print("-------------------------------------------------------------\n")
                
                self.model.train()
                self.mi_estimator.eval()
                
                # Update VQA Network.
                cont_emb, type_emb, word_scores = self.model(batch)
                loss_cont, loss_type, loss_vqa = self.model.criterion(cont_emb, type_emb, word_scores, batch)
                mi_est = self.mi_estimator(cont_emb, type_emb)
                loss_reg = self.regularizer(batch, self.model)
                loss = beta * (loss_cont + loss_type + 0.001 * mi_est) + loss_vqa + 0.5 * loss_reg
                self.model_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.model_optimizer.step()

                # Update Inference Network.
                ave_mi_loss = 0
                for i in range(self.hparams.num_optim_mi):
                    self.mi_estimator.train()
                    batch_mi_loss = self.mi_estimator.learning_loss(cont_emb, type_emb)
                    self.mi_optimizer.zero_grad()
                    batch_mi_loss.backward(retain_graph=True)
                    self.mi_optimizer.step()
                    ave_mi_loss += batch_mi_loss.item()
                ave_mi_loss /= self.hparams.num_optim_mi

                acc_yn, acc_num, acc_oth, acc_all = self._run_eval(batch)
                
                iterations += 1
                self.writer_dict["mi_nll"] = ave_mi_loss
                self.writer_dict["mi_estimate"] = mi_est.item()
                self.writer_dict["type_loss"] = loss_type.item()
                self.writer_dict["cont_loss"] = loss_cont.item()
                self.writer_dict["reg_loss"] = loss_reg.item()
                self.writer_dict["vqa_loss"] = loss_vqa.item()
                self.writer_dict["acc_all"] = acc_all
                self.writer_dict["acc_yn"] = acc_yn
                self.writer_dict["acc_num"] = acc_num
                self.writer_dict["acc_oth"] = acc_oth
                self._record(iterations)

            beta = self._beta_scheduler(beta)
            model_path = self.checkpoint_manager.step(epoch)
            
            # print("\n* Model Evaluating ...\n")
            # torch.cuda.empty_cache()
            # evaluator = VQAEvaluator(self.hparams, "val", epoch, model_path, self.writer)
            # evaluator.run()
            # torch.cuda.empty_cache()
            