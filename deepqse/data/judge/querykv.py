import logging
import torch
from torch.utils.data import Dataset

from deepqse.data.single import QueryKVTitleCollate

class QueryKVTitleJudgeDataset(Dataset):
    def __init__(self, args, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.args = args

        logging.info("[+] start filtering lines")
        self.filter_lines = []

        cnt_max_start = 0
        cnt_no_title = 0
        for line in self.lines:
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
            ref_start = int(ref_start)
            test_start = int(test_start)

            if test_start >= args.max_sent_num or ref_start >= args.max_sent_num:
                cnt_max_start += 1
                continue

            if title == "":
                cnt_no_title += 1
                continue
            self.filter_lines.append(line)

        logging.info(
            f"[+] finish filtering lines, {cnt_max_start} exceed max_sent_len, {cnt_no_title} have no title, last {len(self.filter_lines)} lines"
        )
    
    def __getitem__(self, index):
        line = self.filter_lines[index]
        return self.parse_line(line)

    def __len__(self):
        return len(self.filter_lines)
        
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
        ref_start, test_start = int(ref_start), int(test_start)
        label = 0 if float(judgement_data_int) < 3 else 1

        return query, body_sents, ref_start, test_start, label, title, lang


class QueryKVTitleJudgeCollate(QueryKVTitleCollate):
    def __call__(self, data):
        queries, bodies, ref_start_indexes, test_start_indexes, labels, titles, langs = zip(
            *data
        )

        tokenized_queries = self.tokenize_query_title(queries, titles)
        (
            tokenized_bodies,
            sents_num,
            body_indexes,
            sents_mask,
            query_indexes,
        ) = self.tokenize_body_sents(bodies)

        ref_start_indexes = torch.LongTensor(ref_start_indexes)
        test_start_indexes = torch.LongTensor(test_start_indexes)
        labels = torch.FloatTensor(labels)

        inputs = {
            "queries": tokenized_queries,
            "bodies": tokenized_bodies,
            "sents_num": sents_num,
            "sents_mask": sents_mask,
            "body_indexes": body_indexes,
            "query_indexes": query_indexes,
            "ref_start_indexes": ref_start_indexes,
            "test_start_indexes": test_start_indexes,
            "labels": labels,
        }
        return inputs, langs

