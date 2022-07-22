import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.visdial_dataset import VisDialDataset
from model.visdial.modules import VisDialEncodeModule, VisDialDecodeModule
from model.visdial.model import EncoderDecoderModel
from evaluate.metrics import SparseGTMetrics, NDCG, scores_to_ranks

class MultiEvaluation(object):
    def __init__(self, hparams, split, pre_pthpath):
        self.hparams = hparams
        self.split = split
        self.pre_pthpath = pre_pthpath
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        do_valid, do_test = False, False
        if split == "val":
            do_valid = True
        else:
            do_test = True
        self._build_dataloader(do_valid=do_valid, do_test=do_test)
        self._dataloader = self.valid_dataloader if split == 'val' else self.test_dataloader

        
        self._build_model()

        self.sparse_metrics = SparseGTMetrics()
        self.ndcg = NDCG()

    def _build_dataloader(self, do_valid=False, do_test=False):
        if do_valid:
            self.valid_dataset = VisDialDataset(
                self.hparams,
                overfit=self.hparams.overfit,
                split="val"
            )
            collate_fn = None
            if "dan" in self.hparams.img_feature_type:
                collate_fn = self.valid_dataset.collate_fn
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=self.hparams.eval_batch_size,
                num_workers=self.hparams.cpu_workers,
                drop_last=False,
                collate_fn=collate_fn,
            )

        if do_test:
            self.test_dataset = VisDialDataset(
                self.hparams,
                overfit=self.hparams.overfit,
                split="test"
            )

            collate_fn = None
            if "dan" in self.hparams.img_feature_type:
                collate_fn = self.test_dataset.collate_fn

            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.hparams.eval_batch_size,
                num_workers=self.hparams.cpu_workers,
                drop_last=False,
                collate_fn=collate_fn
            )

    def _build_model(self):
        vocabulary = self.valid_dataset.vocabulary if self.split == "val" else self.test_dataset.vocabulary
        encoder = VisDialEncodeModule(self.hparams)
        decoder = VisDialDecodeModule(self.hparams, self.valid_dataset.vocabulary)
        self.model = EncoderDecoderModel(encoder, encoder).to(self.device)
        self.model.load_pretrained(self.pre_pthpath)

    def load_checkpoint(self, evaluation_path):
        checkpoint = torch.load(evaluation_path)
        self.model.encoder.load_state_dict(checkpoint["encoder"])
        # self.model.decoder.load_state_dict(checkpoint["decoder"])

    def run_evaluate(self, evaluation_path, global_iteration_step=0, eval_json_path=None, eval_seed=None):

        self.load_checkpoint(evaluation_path)
        print("Evaluation model loading completes! ->", evaluation_path)
        
        self.eval_seed = self.hparams.random_seed[0] if eval_seed is None else eval_seed
        self.model.eval()

        ranks_json = []
        self.prob_dist_json = []

        for i, batch in enumerate(tqdm(self._dataloader)):
            
            with torch.no_grad():
                cont_emb, type_emb, pred_score = self.model(batch, test=True)

            batch_size, num_dial, _ = batch['ques'].size()
            ranks = None

            pred_score = pred_score.view(batch_size, num_dial, -1)
            disc_ranks = scores_to_ranks(pred_score)
            ranks = disc_ranks
            output = pred_score

            for i in range(len(batch["img_ids"])):
                # Cast into types explicitly to ensure no errors in schema.
                # Round ids are 1-10, not 0-9
                if self.split == "test":
                    ranks_json.append(
                        {
                            "image_id": batch["img_ids"][i].item(),
                            "round_id": int(batch["num_rounds"][i].item()),
                            "ranks": [rank.item() for rank in ranks[i][batch["num_rounds"][i] - 1]],
                        }
                    )
                else:
                    for j in range(batch["num_rounds"][i]):
                        ranks_json.append(
                            {
                                "image_id": batch["img_ids"][i].item(),
                                "round_id": int(j + 1),
                                "ranks": [rank.item() for rank in ranks[i][j]],
                            }
                        )

                if self.split == "val":
                    self.sparse_metrics.observe(output, batch["ans_ind"])
                    if "gt_relevance" in batch:  # version 1.0
                        output = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
                        self.ndcg.observe(output, batch["gt_relevance"])
               
    
