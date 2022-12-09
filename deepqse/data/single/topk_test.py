import torch
import numpy as np
from .topk import QueryKVTitleTopKDataset, QueryKVTitleTopKCollate


class QueryKVTitleTopKTestDataset(QueryKVTitleTopKDataset):
    def parse_line(self, line, score_line):
        query, title, snippet, body, url, lang, start_index, end_index = line.split(
            "\t"
        )

        body_sents = body.split("<sep>")
        start_index, end_index = int(start_index), int(end_index)

        _, scores = score_line.strip("\n").split("\t")
        scores = np.array([float(score) for score in scores.split()])
        topk_num = min(len(scores), self.args.step1_topk)

        topk_indices = np.argsort(scores)[::-1][:topk_num]

        topk_body_sents = [body_sents[i] for i in topk_indices]

        # query's position is 0
        position_ids = topk_indices + 1
        return (
            query,
            topk_body_sents,
            title,
            start_index,
            lang,
            position_ids,
            topk_indices,
            len(scores),
        )


class QueryKVTitleTopKTestCollate(QueryKVTitleTopKCollate):
    def __call__(self, data):
        (
            queries,
            bodies,
            titles,
            start_indexes,
            langs,
            batch_pos,
            batch_topk_indices,
            batch_raw_sents_num,
        ) = zip(*data)

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
            "top_indices": batch_topk_indices,
            "raw_sents_num": batch_raw_sents_num,
        }
        return inputs, langs
