import torch

from .querykv import QueryKVTitleJudgeDataset
from deepqse.data.base import DECollate

class QuerySentJudgeDataset(QueryKVTitleJudgeDataset):
    pass

class QuerySentJudgeCollate(DECollate):
    def __call__(self, data):
        queries, bodies, ref_start_indexes, test_start_indexes, labels, titles, langs = zip(
            *data
        )

        tokenized_queries = self.tokenizers(
            list(queries), padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_sent_length
        )

        # concatenate sentence with query and title
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
        ) = self.tokenize_body_sents(bodies)
        ref_start_indexes = torch.LongTensor(ref_start_indexes)
        test_start_indexes = torch.LongTensor(test_start_indexes)
        body_indexes = torch.LongTensor(body_indexes)
        labels = torch.FloatTensor(labels)

        inputs = {
            "queries": tokenized_queries,
            "bodies": tokenized_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "body_indexes": body_indexes,
            "ref_start_indexes": ref_start_indexes,
            "test_start_indexes": test_start_indexes,
            "labels": labels
        }
        return inputs, langs

