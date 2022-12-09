import os
import torch
import itertools
import logging

from torch.utils.data import Dataset, DataLoader
from ..utils import MODEL_CLASSES


class DEDataset(Dataset):
    def __init__(self, args, data_path):
        with open(data_path, "r") as f:
            self.lines = f.readlines()
        self.args = args

        logging.info("[+] start filtering lines")
        self.filter_lines = []
        for line in self.lines:
            _, _, _, _, _, start_index, _ = line.split("\t")
            start_index = int(start_index)

            if start_index >= args.max_sent_num:
                continue
            self.filter_lines.append(line)
        logging.info(
            f"[+] finish filtering lines, contains {len(self.filter_lines)} lines"
        )

    def __getitem__(self, index):
        line = self.filter_lines[index]
        return self.parse_line(line)

    def __len__(self):
        return len(self.filter_lines)

    def parse_line(self, line):
        query, snippet, body, url, lang, start_index, end_index = line.split("\t")

        body_sents = body.split("<sep>")
        start_index, end_index = int(start_index), int(end_index)

        return query, body_sents, start_index, lang


class DECollate:
    def __init__(self, args):
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.tokenizers = tokenizer_class.from_pretrained(
            os.path.join(self.args.bert_path, args.tokenizer_name)
            if self.args.bert_name is None
            else self.args.bert_name,
            do_lower_case=True,
        )

    def len2mask(self, lens, max_len):
        mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
        return mask.float()

    def tokenize_body_sents(self, bodies):
        sents_num = [len(body) for body in bodies]
        max_sents_num = min(max(sents_num), self.args.max_sent_num)

        # pad every body to max_sents_num
        flatten_body_sents = [""]
        body_indexes = []
        for body in bodies:
            body = body[:max_sents_num]
            body_index = []
            for sent in body:
                body_index.append(len(flatten_body_sents))
                flatten_body_sents.append(sent)
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
        return tokenized_body, sents_num, body_indexes, sents_mask

    def __call__(self, data):
        queries, bodies, start_indexes, langs = zip(*data)

        tokenized_queries = self.tokenizers(
            list(queries), padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_sent_length
        )
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
        ) = self.tokenize_body_sents(bodies)
        start_indexes = torch.LongTensor(start_indexes)
        body_indexes = torch.LongTensor(body_indexes)

        inputs = {
            "queries": tokenized_queries,
            "bodies": tokenized_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "start_indexes": start_indexes,
            "body_indexes": body_indexes,
        }
        return inputs, langs
