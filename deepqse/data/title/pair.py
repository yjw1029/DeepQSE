from deepqse.data.title.title_sent import TitleSentDataset


class TitlePairDataset(TitleSentDataset):
    def parse_line(self, line):
        query, body_sents, title, start_index, lang = super().parse_line(line)
    
        pair_body_sents = [(query, sent) for sent in body_sents]
        return query, pair_body_sents, start_index, lang
