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

from data.visdial_dataset import VisDialDataset
from model.vqa.modules import MIEstimator
from model.visdial.checkpointing import CheckpointManager
from model.visdial.modules import VisDialEncodeModule, VisDialDecodeModule
from model.visdial.model import EncoderDecoderModel

class VisDialModel(object):
    def __init__(self, hparams):
        self.hparams = hparams        
        torch.manual_seed(hparams.random_seed[0])
        torch.cuda.manual_seed_all(hparams.random_seed[0])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_dataloader(self):
        print("* Loading dataset ...\n")

        self.train_dataset = VisDialDataset(
            self.hparams,
            overfit=self.hparams.overfit,
            split="train", 
        )
        self.val_dataset = VisDialDataset(
            self.hparams,
            overfit=self.hparams.overfit,
            split="val", 
        )

        collate_fn = None
        if "dan" in self.hparams.img_feature_type:
            collate_fn = self.train_dataset.collate_fn

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.cpu_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.cpu_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn
        )

        print("\t# --------------------------------------------------")
        print("\t#   DATALOADER FINISHED")
        print("\t# --------------------------------------------------\n")

    def _build_model(self):

        print("* Building model ...\n")

        self.mi_estimator = MIEstimator(self.hparams).to(self.device)
        visdial_encoder = VisDialEncodeModule(self.hparams)
        visdial_decoder = VisDialDecodeModule(self.hparams, self.train_dataset.vocabulary)
        self.model = EncoderDecoderModel(visdial_encoder, visdial_decoder)
        self.model.load_pretrained(self.hparams.load_pthpath)
        self.model = self.model.to(self.device)
        
        # Total Iterations -> for learning rate scheduler
        if self.hparams.training_splits == "trainval":
            self.iterations = (len(self.train_dataset) + len(self.valid_dataset)) // self.hparams.train_batch_size + 1
        else:
            self.iterations = len(self.train_dataset) // self.hparams.train_batch_size + 1
        self.val_iterations = len(self.val_dataset) // self.hparams.eval_batch_size + 1

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
            self.model.encoder,
            self.model.decoder,
            self.hparams.save_dirpath, 
        )
        self.writer = SummaryWriter(self.hparams.result_dirpath)
        self.writer_dict = {
            "mi_estimate": 0.0,
            "mi_nll": 0.0,
            "cont_loss": 0.0,
            "disc_loss": 0.0,
            "KL_div": 0.0, 
        } 
        self.start_epoch = 1

        print("\t# --------------------------------------------------")
        print("\t#   SETUP TRINING FINISHED")
        print("\t# --------------------------------------------------\n")

    def _record(self, iterations: int):
        self.writer.add_scalar("Objective/MI Estimate", self.writer_dict["mi_estimate"], iterations)
        self.writer.add_scalar("Objective/MI NLL", self.writer_dict["mi_nll"], iterations)
        self.writer.add_scalar("Objective/Cont loss", self.writer_dict["cont_loss"], iterations)
        self.writer.add_scalar("Objective/Disc loss", self.writer_dict["disc_loss"], iterations)
        self.writer.add_scalar("Objective/KL div", self.writer_dict["KL_div"], iterations)
        
    def _get_val_loss(self, beta):
        pass

    def _beta_scheduler(self, beta):
        beta += 0.1
        if beta >= 1:
            return 1
        else:
            return beta

    def train(self):
        self._build_dataloader()
        self._build_model()
        self._setup_training()

        # Evaluation Setup
        # evaluator = VQAEvaluator()

        print("* Model Training ...\n")
        beta = 0.5
        iterations = 0
        for epoch in range(self.start_epoch , self.hparams.num_epochs):
            
            # ====================================================================
            #   ON EPOCH START  (combine dataloaders if training on train + val)
            # ====================================================================
            
            if self.hparams.training_splits == "trainval":
                combined_dataloader = itertools.chain(self.train_dataloader, self.valid_dataloader)
            else:
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
                
                # Update VisDial Network.
                cont_emb, type_emb, KLD, pred_score = self.model(batch)
                loss_cont, loss_disc = self.model.criterion(cont_emb, pred_score, batch)
                mi_est = self.mi_estimator(cont_emb, type_emb.detach())
                loss = beta * (loss_cont + 0.001 * mi_est) + loss_disc + KLD
                loss = loss / self.hparams.accumulation_steps
                loss.backward(retain_graph=True)

                # Update Inference Network.
                ave_mi_loss = 0
                for i in range(self.hparams.num_optim_mi):
                    self.mi_estimator.train()
                    batch_mi_loss = self.mi_estimator.learning_loss(cont_emb, type_emb.detach())
                    batch_mi_loss = batch_mi_loss / self.hparams.accumulation_steps
                    batch_mi_loss.backward(retain_graph=True)
                    ave_mi_loss += batch_mi_loss.item() * self.hparams.accumulation_steps
                ave_mi_loss /= self.hparams.num_optim_mi

                if (iterations+1) % self.hparams.accumulation_steps == 0:
                    self.model_optimizer.step()
                    self.model_optimizer.zero_grad()
                    self.mi_optimizer.step()
                    self.mi_optimizer.zero_grad()

                # acc_yn, acc_num, acc_oth, acc_all = evaluator.run(batch, self.model.eval())
                
                iterations += 1
                self.writer_dict["mi_nll"] = ave_mi_loss
                self.writer_dict["mi_estimate"] = mi_est.item()
                self.writer_dict["cont_loss"] = loss_cont.item()
                self.writer_dict["disc_loss"] = loss_disc.item()
                self.writer_dict["KL_div"] = KLD.item()
                self._record(iterations)

            beta = self._beta_scheduler(beta)
            self.checkpoint_manager.step(epoch)
