import torch
import logging
import numpy as np
from pathlib import Path

from ..utils import MODEL_CLASSES

from .pair import DEPairDataset
from .topk import TopKCollate


class TopKTestDataset(DEPairDataset):
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


class TopKTestCollate(TopKCollate):
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
        (
            queries,
            bodies,
            start_indexes,
            langs,
            batch_pos,
            batch_topk_indices,
            batch_raw_sents_num,
        ) = zip(*data)

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
            "top_indices": batch_topk_indices,
            "raw_sents_num": batch_raw_sents_num,
        }
        return inputs, langs
