from deepqse.data.title import TitleSentDataset, TitleSentCollate

import logging
import torch


class WOQuerySentDataset(TitleSentDataset):
    pass


class WOQueryCollate(TitleSentCollate):
    def tokenize_query(self, queries, titles):
        tokenized_toks = []

        for query, title in zip(queries, titles):
            tokenized_query = self.tokenizers.tokenize(query)
            tokenized_title = self.tokenizers.tokenize(title)[
                : self.args.max_title_length
            ]
            toks = ["<s>"] + tokenized_title + ["</s>"]

            tokenized_toks.append(toks)

        max_len = max([len(toks) for toks in tokenized_toks])

        input_ids = []
        attention_masks = []

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
        }

        return tokenized_query

    def __call__(self, data):
        queries, bodies, titles, start_indexes, langs = zip(*data)

        tokenized_queries = self.tokenize_query(queries, titles)

        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
        ) = self.tokenize_body_sents(bodies, titles)
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
