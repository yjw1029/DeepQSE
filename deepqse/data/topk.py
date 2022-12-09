import os
import torch
import itertools
import logging
import random
import math
import numpy as np
from pathlib import Path

from ..utils import MODEL_CLASSES

from .pair import DEPairDataset, PairCollate


class TopKDataset(DEPairDataset):
    def __init__(self, args, data_path):
        super().__init__(args, data_path)

        logging.info("[+] start loading labels of teacher model")

        if self.args.mode in ["train", "test"]:
            if "train" in str(data_path):
                infer_file = (
                    Path(args.infer_path) / args.teacher_name / "train" / "result.tsv"
                )
            elif "dev" in str(data_path):
                infer_file = (
                    Path(args.infer_path) / args.teacher_name / "dev" / "result.tsv"
                )
            elif "test" in str(data_path):
                infer_file = (
                    Path(args.infer_path) / args.teacher_name / "test" / "result.tsv"
                )
        elif self.args.mode == "combine_test":
            infer_file = (
                Path(args.infer_path) / args.teacher_name / "test" / "result.tsv"
            )

        with open(infer_file, "r") as f:
            self.teacher_scores = f.readlines()

        assert len(self.teacher_scores) == len(
            self.filter_lines
        ), "Number of teacher samples not equal to teacher scores"
        logging.info("[-] finish loading labels of teacher model")

    def __getitem__(self, index):
        line = self.filter_lines[index]
        score_line = self.teacher_scores[index]
        return self.parse_line(line, score_line)

    def parse_line(self, line, score_line):
        query, pair_body_sents, start_index, lang = super().parse_line(line)

        _, scores = score_line.strip("\n").split("\t")
        scores = np.array([float(score) for score in scores.split()])
        topk_num = min(len(scores), self.args.step1_topk)

        scores[start_index] = -1e4
        topk_indices = np.argsort(scores)[::-1][: topk_num - 1]
        topk_indices = np.concatenate((np.array([start_index]), topk_indices))

        topk_body_sents = [pair_body_sents[i] for i in topk_indices]

        # query's position is 0
        position_ids = topk_indices + 1
        return query, topk_body_sents, 0, lang, position_ids


class TopKCollate(PairCollate):
    def tokenize_body_sents(self, bodies):
        """
        bodies: list of (query, sent) pairs
        """
        pair_num = [len(body) for body in bodies]
        max_sents_num = max(pair_num)

        # pad every body to max_sents_num
        flatten_pair_sents = [("", "")]
        body_indexes = []
        for body in bodies:
            body = body[:max_sents_num]
            body_index = []
            for sent in body:
                body_index.append(len(flatten_pair_sents))
                flatten_pair_sents.append(sent)
            body_index = body_index + [0] * (max_sents_num - len(body))
            body_indexes.extend(body_index)

        querys, bodies = list(zip(*flatten_pair_sents))

        tokenized_body = self.tokenizers(
            querys,
            bodies,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_sent_length,
        )

        sents_num = torch.LongTensor(pair_num)
        sents_mask = self.len2mask(sents_num, max_sents_num)
        sents_mask = torch.FloatTensor(sents_mask)
        return tokenized_body, sents_num, body_indexes, sents_mask

    def parse_pos_ids(self, batch_pos):
        sents_num = [len(pos) for pos in batch_pos]
        max_sents_num = max(sents_num)
        max_pos = 511

        batch_padded_pos = []

        for pos in batch_pos:
            pos = pos.tolist()
            pad_pos = pos[:max_sents_num] + [max_pos] * (max_sents_num - len(pos))
            batch_padded_pos.append(pad_pos)

        return torch.LongTensor(batch_padded_pos)

    def __call__(self, data):
        queries, bodies, start_indexes, langs, batch_pos = zip(*data)

        tokenized_queries = self.tokenizers(
            queries, padding=True, truncation=True, return_tensors="pt"
        )
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
        ) = self.tokenize_body_sents(bodies)
        start_indexes = torch.LongTensor(start_indexes)
        body_indexes = torch.LongTensor(body_indexes)

        batch_padded_pos = self.parse_pos_ids(batch_pos)

        inputs = {
            "queries": tokenized_queries,
            "bodies": tokenized_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "start_indexes": start_indexes,
            "body_indexes": body_indexes,
            "pos_ids": batch_padded_pos,
        }
        return inputs, langs
