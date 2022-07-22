import h5py
import pickle
import json
import numpy as np

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data.preprocess.vqa_preprocessors import (
    Vocabulary,
    BertVocabulary
)

class VisDialDataset(Dataset):
    """
        A full representation of VisDial v1.0 (train/val/test) dataset. According
        to the appropriate split, it returns dictionary of question, image,
        history, ground truth answer, answer options, dense annotations etc.

        Number of data in each split:
            train : 123287
            val   : 2064
            test  : 8000
    """

    def __init__(
        self,
        hparams,
        overfit: bool = False,
        split: str = "",
    ):
        super().__init__()
        self.hparams = hparams
        self.split = split
        self.bert_vocabulary = BertVocabulary()
        self.vocabulary = Vocabulary(
            word_counts_path=self.hparams.share_word_counts_json, 
            min_count=self.hparams.vocab_min_count,
        )
        
        # train, val, test
        text_features_hdf5_path = hparams.visdial_text_features_hdf5 % (self.split)
        img_features_h5_path = hparams.img_features_h5 % (self.hparams.img_feature_type, self.split)

        self.hdf_reader = DataHdfReader(hparams, text_features_hdf5_path, img_features_h5_path, self.split)
                
        # Keep a list of image_ids as primary keys to access data.
        self.text_feat_image_ids = self.hdf_reader.text_feature_id_l
        # print("num of images :", len(self.text_feat_image_ids))
        
        if overfit:
            self.text_feat_image_ids = self.text_feat_image_ids[:self.hparams.num_pick_data]
        self.float_variables = ["img_feats", "gt_relevance", "img_boxes", "img_sp_feats"]
        
    def __len__(self):
        return len(self.text_feat_image_ids)

    def __getitem__(self, index):
        # Get image_id, which serves as a primary key for current instance.
        # Get image features for this image_id using hdf reader.
        curr_features = self.hdf_reader[index]

        for f_key in curr_features.keys():
            if f_key in self.float_variables:
                curr_features[f_key] = torch.tensor(curr_features[f_key]).float()
                continue
            curr_features[f_key] = torch.tensor(curr_features[f_key]).long()

        return curr_features

    def collate_fn(self, batch):
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        max_np = max(merged_batch["num_proposals"])
        for key in merged_batch:
            if key in ["img_feats", "img_boxes", "img_sp_feats"]:
                for batch_idx, features in enumerate(merged_batch[key]):
                    pad_features = torch.zeros((max_np - len(features), features.size()[1])).float()
                    merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
            merged_batch[key] = torch.stack(merged_batch[key], 0)

        return merged_batch


class DataHdfReader(object):
    def __init__(self, 
        hparams,
        text_features_hdf5_path: str,
        img_features_h5_path: str,
        split=None
    ):
        self.text_features_hdf5_path = text_features_hdf5_path
        self.img_features_h5_path = img_features_h5_path
        self._split = split
        self.hparams = hparams

        # Text
        with h5py.File(self.text_features_hdf5_path, "r") as text_features_hdf5:
            self.feature_keys = list(text_features_hdf5.keys())
            # print("feature_keys", self.feature_keys)

            self._split = split
            assert split == self._split
            # print("data split :", self._split)

            self.text_feature_id_l = list(text_features_hdf5["img_ids"])
            
        # Image
        if hparams.img_feature_type == "dan_faster_rcnn_x101":
            # get imgid2id dicionary
            with open(hparams.imgid2idx_path % split, "rb") as imgid2idx_pkl:
                self.img_feature_id_l = list(pickle.load(imgid2idx_pkl))
            with h5py.File(self.img_features_h5_path, "r") as features_h5:
                self.pos_boxes = np.array(features_h5.get("pos_boxes"))
        else:
            with h5py.File(self.img_features_h5_path, "r") as img_features_h5:
                self.img_feature_id_l = list(img_features_h5["image_id"])
    
        assert len(self.text_feature_id_l) == len(self.img_feature_id_l)
    
    def __len__(self):
        return len(self.text_feature_id_l)

    def __getitem__(self, index: int):
        """            
            Parameters
            ----------
            index : int
                Index of data in dataloader.

            Returns
            ----------
            features : Dict
                |   split | train, val, test
                |--- "img_ids"       [shape: (num_images, )]
                |--- "hist"          [shape: (num_images, num_rounds, num_sents, seq_length)]
                |--- "hist_seg"      [shape: (num_images, num_rounds, num_sents, seq_length)]
                |--- "ques"          [shape: (num_images, num_rounds, seq_length)]
                |--- "qa_pairs"      [shape: (num_images, num_rounds, seq_length)]
                |--- "qa_seg"        [shape: (num_images, num_rounds, seq_length)]
                |--- "ans_in"        [shape: (num_images, num_rounds, seq_length)]
                |--- "ans_out"       [shape: (num_images, num_rounds, seq_length)]
                |--- "ques_in"       [shape: (num_images, num_rounds, seq_length)]
                |--- "ques_out"      [shape: (num_images, num_rounds, seq_length)]
                |--- "ques_len"      [shape: (num_images, num_rounds)]
                |--- "ans_len"       [shape: (num_images, num_rounds)]
                |--- "hist_len"      [shape: (num_images, num_sents)]
                |--- "num_rounds"    [shape: (num_images, )]
                |
                |--- "opt_in"        [shape: (num_images, num_rounds, num_options, seq_length)]
                |--- "opt_out"       [shape: (num_images, num_rounds, num_options, seq_length)]
                |--- "opt"           [shape: (num_images, num_rounds, num_options, seq_length)]
                |--- "opt_len"       [shape: (num_images, num_rounds, num_options)]
                |
                |   split | train, val
                |--- "ans_ind"       [shape: (num_images, num_rounds)]
                |    
                |   split | val
                |--- "gt_relevance"  [shape: (num_images, num_options)]
                |--- "round_id"      [shape: (num_images, )]
                |
                |   img_feature_type | faster_rcnn_x101, dan_faster_rcnn_x101
                |--- "img_feats"     [shape: (num_images, num_proposals, 2048)]
                |    
                |   img_feature_type | dan_faster_rcnn_x101
                |--- "num_proposals" [shape: (num_images, )]
                └--- "img_sp_feats"  [shape: (num_images, num_proposals, 6)]
        """
        features = {}
        text_feature_index = index

        # Text
        with h5py.File(self.text_features_hdf5_path, "r") as text_features_hdf5:
            for f_key in self.feature_keys:
                features[f_key] = text_features_hdf5[f_key][text_feature_index]
            image_id = text_features_hdf5["img_ids"][text_feature_index]
        assert image_id == self.text_feature_id_l[index]
        assert image_id in self.img_feature_id_l
    
        # Image
        img_feature_index = self.img_feature_id_l.index(image_id)
        if self.hparams.img_feature_type == "dan_faster_rcnn_x101":
            with h5py.File(self.img_features_h5_path, "r") as features_h5:
                image_features = features_h5["image_features"][
                    self.pos_boxes[img_feature_index][0]: 
                    self.pos_boxes[img_feature_index][1], :]
                bounding_boxes = [features_h5["image_bb"][feature_idx]
                    for feature_idx in range(self.pos_boxes[img_feature_index][0], self.pos_boxes[img_feature_index][1])]
                spatial_features = [features_h5["spatial_features"][feature_idx]
                    for feature_idx in range(self.pos_boxes[img_feature_index][0], self.pos_boxes[img_feature_index][1])]
    
                features["img_feats"] = image_features
                features["num_proposals"] = len(image_features)
                features["img_boxes"] = np.array(bounding_boxes)
                features["img_sp_feats"] = np.array(spatial_features)
        else:
            with h5py.File(self.img_features_h5_path, "r") as img_features_h5:
                assert image_id == img_features_h5["image_id"][img_feature_index]
                features["img_feats"] = img_features_h5["features"][img_feature_index]
        return features

    def keys(self) -> List[int]:
        return self.text_feature_id_l

    @property
    def split(self):
        return self._split