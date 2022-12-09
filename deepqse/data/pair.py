import os
import torch
import itertools
import logging
import random
import math

from ..utils import MODEL_CLASSES

from .base import DEDataset, DECollate


class DEPairDataset(DEDataset):
    def parse_line(self, line):
        query, body_sents, start_index, lang = super().parse_line(line)
        pair_body_sents = [(query, sent) for sent in body_sents]
        return query, pair_body_sents, start_index, lang


class PairCollate(DECollate):
    """Should be used with xxPairDataset"""

    def tokenize_body_sents(self, bodies):
        """
        bodies: list of (query, sent) pairs
        """
        pair_num = [len(body) for body in bodies]
        max_sents_num = min(max(pair_num), self.args.max_sent_num)

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
