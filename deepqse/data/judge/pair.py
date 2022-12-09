# for birch and parad
import torch

from .dssm import DSSMJudgeDataset
from deepqse.data.pair import PairCollate


class PairJudgeDataset(DSSMJudgeDataset):
    def parse_line(self, line):
        (
            query,
            url,
            _,
            _,
            judgement_data_int,
            lang,
            body,
            ref_start,
            _,
            test_start,
            _,
            title,
        ) = line.strip("\n").split("\t")

        body_sents = body.split("<sep>")
        pair_body_sents = [(query, sent) for sent in body_sents]
        ref_start, test_start = int(ref_start), int(test_start)
        label = 0 if float(judgement_data_int) < 3 else 1
        confidence = abs(float(judgement_data_int) - 3.0)

        return query, pair_body_sents, ref_start, test_start, label, confidence, lang


class PairJudgeCollate(PairCollate):
    def __call__(self, data):
        queries, bodies, ref_start_indexes, test_start_indexes, labels, confidence_scores, langs = zip(
            *data
        )

        tokenized_queries = self.tokenizers(
            list(queries),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_sent_length,
        )
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
        ) = self.tokenize_body_sents(bodies)

        ref_start_indexes = torch.LongTensor(ref_start_indexes)
        test_start_indexes = torch.LongTensor(test_start_indexes)
        body_indexes = torch.LongTensor(body_indexes)
        confidence_scores = torch.FloatTensor(confidence_scores)
        labels = torch.FloatTensor(labels)

        inputs = {
            "queries": tokenized_queries,
            "bodies": tokenized_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "ref_start_indexes": ref_start_indexes,
            "test_start_indexes": test_start_indexes,
            "labels": labels,
            "confidence_scores": confidence_scores,
            "body_indexes": body_indexes,
        }
        return inputs, langs
