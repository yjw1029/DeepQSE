import torch
from .pair import DECollate, PairCollate

class TwoStepCollate(DECollate):
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
            flatten_pair_sents.extend(body)

        querys, bodies = list(zip(*flatten_pair_sents))

        tokenized_query_body = self.tokenizers(
            querys,
            bodies,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_sent_length,
        )

        tokenized_body = self.tokenizers(
            bodies,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_body_length,
        )

        sents_num = torch.LongTensor(pair_num)
        sents_mask = self.len2mask(sents_num, max_sents_num)
        sents_mask = torch.FloatTensor(sents_mask)
        return tokenized_query_body, tokenized_body, sents_num, body_indexes, sents_mask

    def __call__(self, data):
        queries, bodies, start_indexes, langs = zip(*data)

        tokenized_queries = self.tokenizers(
            queries, padding=True, return_tensors="pt"
        )
        (   
            tokenized_query_bodies,
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
            "query_bodies": tokenized_query_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "start_indexes": start_indexes,
            "body_indexes": body_indexes,
        }
        return inputs, langs
