import torch
import os

from deepqse.data.title import DETitleDataset
from deepqse.data.base import DECollate
from deepqse.utils import MODEL_CLASSES


class QueryKVPosDataset(DETitleDataset):
    pass


class QueryKVPosCollate(DECollate):
    def tokenize_body_sents(self, bodies, queries_lens):
        sents_num = [len(body) for body in bodies]
        max_sents_num = min(max(sents_num), self.args.max_sent_num)

        # pad every body to max_sents_num
        flatten_body_sents = [""]
        body_indexes = []
        query_indexes = [0]
        position_ids = [
            torch.arange(
                queries_lens[0],
                queries_lens[0] + self.args.max_sent_length,
                dtype=torch.long,
            )
        ]
        for cnt, body in enumerate(bodies):
            body = body[:max_sents_num]
            body_index = []
            for sent in body:
                body_index.append(len(flatten_body_sents))
                flatten_body_sents.append(sent)
                query_indexes.append(cnt + 1)
                position_ids.append(
                    torch.arange(
                        queries_lens[cnt + 1],
                        queries_lens[cnt + 1] + self.args.max_sent_length,
                        dtype=torch.long,
                    )
                )
            body_index = body_index + [0] * (max_sents_num - len(body))
            body_indexes.extend(body_index)

        tokenized_body = self.tokenizers(
            flatten_body_sents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_sent_length,
        )

        tokenized_body.data["position_ids"] = torch.stack(position_ids, dim=0)[
            :, : tokenized_body["input_ids"].size(-1)
        ]

        sents_num = torch.LongTensor(sents_num)
        sents_mask = self.len2mask(sents_num, max_sents_num)
        sents_mask = torch.FloatTensor(sents_mask)
        body_indexes = torch.LongTensor(body_indexes)
        query_indexes = torch.LongTensor(query_indexes)
        return tokenized_body, sents_num, body_indexes, sents_mask, query_indexes

    def tokenize_queries(self, queries):
        tokenized_toks = [["<s>", "</s>"]]
        query_lens = [2]

        for query in queries:
            tokenized_query = self.tokenizers.tokenize(query)
            toks = ["<s>"] + tokenized_query + ["</s>"]

            tokenized_toks.append(toks)
            query_lens.append(len(toks))

        max_len = max([len(toks) for toks in tokenized_toks])

        input_ids = []
        attention_masks = []
        position_ids = (
            torch.arange(max_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(len(tokenized_toks), -1)
        )

        for toks in tokenized_toks:
            input_id = self.tokenizers.convert_tokens_to_ids(toks)
            pad_num = max_len - len(input_id)
            attention_mask = [1] * len(input_id) + [0] * pad_num
            input_id = input_id + [0] * pad_num

            input_ids.append(input_id)
            attention_masks.append(attention_mask)

        tokenized_query = {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_masks),
            "position_ids": position_ids,
        }

        return tokenized_query, query_lens

    def __call__(self, data):
        queries, bodies, start_indexes, langs = zip(*data)

        tokenized_queries, query_lens = self.tokenize_queries(queries)
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
            query_indexes,
        ) = self.tokenize_body_sents(bodies, query_lens)
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
