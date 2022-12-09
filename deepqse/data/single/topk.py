import torch
from torch.utils.data import Dataset

import logging
from pathlib import Path
import numpy as np

from deepqse.data.single.querykv_title import QueryKVTitleCollate


class QueryKVTitleTopKDataset(Dataset):
    def __init__(self, args, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.args = args

        logging.info("[+] start filtering lines")
        self.filter_lines = []

        cnt_max_start = 0
        cnt_no_title = 0
        for line in self.lines:
            _, title, _, _, _, _, start_index, _ = line.split("\t")
            start_index = int(start_index)

            if start_index >= args.max_sent_num:
                cnt_max_start += 1
                continue

            if title == "":
                cnt_no_title += 1
                continue
            self.filter_lines.append(line)

        logging.info(
            f"[+] finish filtering lines, {cnt_max_start} exceed max_sent_len, {cnt_no_title} have no title, last {len(self.filter_lines)} lines"
        )

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
        ), f"Number of teacher samples {len(self.filter_lines)} not equal to teacher scores {len(self.teacher_scores)}"
        logging.info("[-] finish loading labels of teacher model")

    def __len__(self):
        return len(self.filter_lines)

    def __getitem__(self, index):
        line = self.filter_lines[index]
        score_line = self.teacher_scores[index]
        return self.parse_line(line, score_line)

    def parse_line(self, line, score_line):
        query, title, snippet, body, url, lang, start_index, end_index = line.split(
            "\t"
        )

        body_sents = body.split("<sep>")
        start_index, end_index = int(start_index), int(end_index)

        _, scores = score_line.strip("\n").split("\t")
        scores = np.array([float(score) for score in scores.split()])
        topk_num = min(len(scores), self.args.step1_topk)

        scores[start_index] = -1e4
        topk_indices = np.argsort(scores)[::-1][: topk_num - 1]
        topk_indices = np.concatenate((np.array([start_index]), topk_indices))

        topk_body_sents = [body_sents[i] for i in topk_indices]

        # query's position is 0
        position_ids = topk_indices + 1
        return query, topk_body_sents, title, 0, lang, position_ids


class QueryKVTitleTopKCollate(QueryKVTitleCollate):
    def tokenize_body_sents(self, bodies):
        sents_num = [len(body) for body in bodies]
        max_sents_num = max(sents_num)

        # pad every body to max_sents_num
        flatten_body_sents = [""]
        body_indexes = []
        query_indexes = [0]
        for cnt, body in enumerate(bodies):
            body = body[:max_sents_num]
            body_index = []
            for sent in body:
                body_index.append(len(flatten_body_sents))
                flatten_body_sents.append(sent)
                query_indexes.append(cnt + 1)
            body_index = body_index + [0] * (max_sents_num - len(body))
            body_indexes.extend(body_index)

        tokenized_body = self.tokenizers(
            flatten_body_sents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_sent_length,
        )

        sents_num = torch.LongTensor(sents_num)
        sents_mask = self.len2mask(sents_num, max_sents_num)
        sents_mask = torch.FloatTensor(sents_mask)
        body_indexes = torch.LongTensor(body_indexes)
        query_indexes = torch.LongTensor(query_indexes)
        return tokenized_body, sents_num, body_indexes, sents_mask, query_indexes

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
        queries, bodies, titles, start_indexes, langs, batch_pos = zip(*data)

        tokenized_queries = self.tokenize_query_title(queries, titles)
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
            query_indexes,
        ) = self.tokenize_body_sents(bodies)
        start_indexes = torch.LongTensor(start_indexes)

        batch_padded_pos = self.parse_pos_ids(batch_pos)

        inputs = {
            "queries": tokenized_queries,
            "bodies": tokenized_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "start_indexes": start_indexes,
            "body_indexes": body_indexes,
            "query_indexes": query_indexes,
            "pos_ids": batch_padded_pos,
        }
        return inputs, langs
