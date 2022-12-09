from deepqse.data.topk_test import TopKTestDataset


import os
import torch
import itertools
import logging
import random
import math
import numpy as np
from pathlib import Path

from deepqse.data.title.pair import TitlePairDataset



class TopKTitleTestDataset(TitlePairDataset):
    def __init__(self, args, data_path):
        super().__init__(args, data_path)

        logging.info("[+] start loading labels of teacher model")

        if self.args.mode in ["train", "test"]:
            infer_file = (
                Path(args.infer_path) / args.teacher_name / args.mode / "result.tsv"
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

        topk_indices = np.argsort(scores)[::-1][:topk_num]

        topk_body_sents = [pair_body_sents[i] for i in topk_indices]

        position_ids = topk_indices + 1
        return (
            query,
            topk_body_sents,
            start_index,
            lang,
            position_ids,
            topk_indices,
            len(scores),
        )


