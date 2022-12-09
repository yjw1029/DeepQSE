import torch
import logging
from collections import defaultdict


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def precision(y_true, y_hat, k=3):
    k = min(k, y_hat.shape[-1])
    _, indices = torch.topk(y_hat, k=k)
    hit = torch.sum(indices == y_true.unsqueeze(-1))
    return hit.data.float() * 1.0


class BaseMetric:
    def __init__(self, name, func, *func_args, **func_kwargs):
        self.name = name
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.metric_val = defaultdict(int)

    def update(self, y_true, y_hat, lang):
        score = self.func(y_true, y_hat, *self.func_args, **self.func_kwargs)

        self.metric_val[lang] += score
        self.metric_val["all"] += score

    def get_rslt(self, cnt):
        rslt = {}
        for key in cnt:
            rslt[key] = self.metric_val[key] / cnt[key]
        return rslt


class PrecMetric(BaseMetric):
    def __init__(self, name="prec", func=precision, k=1, *func_args, **func_kwargs):
        self.name = f"{name}@{k}"
        super().__init__(self.name, func, k, *func_args, **func_kwargs)


class MetricPool:
    def __init__(self):
        self.metrics = []
        self.lang_cnt = defaultdict(int)

    def regist(self, metric):
        assert isinstance(metric, BaseMetric), "input wrong metrics type"
        self.metrics.append(metric)

    def update(self, y_true, y_hat, lang):
        self.lang_cnt["all"] += 1
        self.lang_cnt[lang] += 1

        for metric in self.metrics:
            metric.update(y_true, y_hat, lang)

    def report(self):
        rslts = {}
        for metric in self.metrics:
            rslts[metric.name] = metric.get_rslt(self.lang_cnt)

        logging.info(" ----------------- start logging result --------------")
        for lang in self.lang_cnt:
            lang_rslt = (
                f"cnt: {self.lang_cnt[lang]}\tlang: {lang}"
                + "\t"
                + "\t".join(
                    [
                        f"{metric_name}: {rslts[metric_name][lang]:.4f}"
                        for metric_name in rslts
                    ]
                )
            )
            logging.info(lang_rslt)
        logging.info(" ----------------- start logging result --------------")
