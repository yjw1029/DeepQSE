import torch
import os

from deepqse.data.title import DETitleDataset
from deepqse.data.base import DECollate
from deepqse.utils import MODEL_CLASSES


class QueryKVDataset(DETitleDataset):
    """directly apply DETitleDataset"""

    pass


class QueryKVCollate(DECollate):
    def tokenize_body_sents(self, bodies):
        sents_num = [len(body) for body in bodies]
        max_sents_num = min(max(sents_num), self.args.max_sent_num)

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

    def tokenize_queries(self, queries):
        # add one query for padding sentences
        queries = [""] + list(queries)
        tokenized_queries = self.tokenizers(
            queries, padding=True, truncation=True, return_tensors="pt"
        )
        return tokenized_queries

    def __call__(self, data):
        queries, bodies, start_indexes, langs = zip(*data)

        tokenized_queries = self.tokenize_queries(queries)
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
            query_indexes,
        ) = self.tokenize_body_sents(bodies)
        start_indexes = torch.LongTensor(start_indexes)

        inputs = {
            "queries": tokenized_queries,
            "bodies": tokenized_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "start_indexes": start_indexes,
            "body_indexes": body_indexes,
            "query_indexes": query_indexes,
        }
        return inputs, langs
