from deepqse.data.base import DEDataset, DECollate

import logging


class DETitleDataset(DEDataset):
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

        return query, body_sents, start_index, lang
