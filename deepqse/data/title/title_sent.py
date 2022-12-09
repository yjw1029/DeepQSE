from deepqse.data.base import DEDataset, DECollate

import logging
import torch


class TitleSentDataset(DEDataset):
    def __init__(self, args, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.args = args

        logging.info("[+] start filtering lines")
        self.filter_lines = []

        cnt_max_start = 0
        cnt_no_title = 0
        for line in self.lines:
            _, title, _, _, _, _, start_index, _ = line.split("\t")
            start_index = int(start_index)

            if start_index >= args.max_sent_num:
                cnt_max_start += 1
                continue

            if title == "":
                cnt_no_title += 1
                continue
            self.filter_lines.append(line)

        logging.info(
            f"[+] finish filtering lines, {cnt_max_start} exceed max_sent_len, {cnt_no_title} have no title, last {len(self.filter_lines)} lines"
        )

    def parse_line(self, line):
        query, title, snippet, body, url, lang, start_index, end_index = line.split(
            "\t"
        )

        body_sents = body.split("<sep>")
        start_index, end_index = int(start_index), int(end_index)

        return query, body_sents, title, start_index, lang


class TitleSentCollate(DECollate):
    def tokenize_body_sents(self, bodies, titles):
        sents_num = [len(body) for body in bodies]
        max_sents_num = min(max(sents_num), self.args.max_sent_num)

        # pad every body to max_sents_num
        tokenized_toks = [["<s>", "</s>" "</s>"]]
        body_indexes = []
        for body_sents, title in zip(bodies, titles):
            body_sents = body_sents[:max_sents_num]
            body_index = []

            tokenized_title = self.tokenizers.tokenize(title)[
                : self.args.max_title_length
            ]
            for sent in body_sents:
                body_index.append(len(tokenized_toks))

                # truncate sent and title and concat them
                tokenized_sent = self.tokenizers.tokenize(sent)[
                    : self.args.max_sent_length
                ]
                tokenized_tok = (
                    ["<s>"] + tokenized_sent + ["</s>"] + tokenized_title + ["</s>"]
                )

                tokenized_toks.append(tokenized_tok)

            body_index = body_index + [0] * (max_sents_num - len(body_sents))
            body_indexes.extend(body_index)

        input_ids = []
        attention_masks = []
        max_len = max([len(toks) for toks in tokenized_toks])

        for toks in tokenized_toks:
            input_id = self.tokenizers.convert_tokens_to_ids(toks)
            pad_num = max_len - len(input_id)
            attention_mask = [1] * len(input_id) + [0] * pad_num
            input_id = input_id + [0] * pad_num

            input_ids.append(input_id)
            attention_masks.append(attention_mask)

        tokenized_body = {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_masks),
        }

        sents_num = torch.LongTensor(sents_num)
        sents_mask = self.len2mask(sents_num, max_sents_num)
        sents_mask = torch.FloatTensor(sents_mask)
        return tokenized_body, sents_num, body_indexes, sents_mask

    def __call__(self, data):
        queries, bodies, titles, start_indexes, langs = zip(*data)

        tokenized_queries = self.tokenizers(
            queries, padding=True, truncation=True, return_tensors="pt"
        )
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
