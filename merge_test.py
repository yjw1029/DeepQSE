import argparse
import re
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--mode", type=str, default="test")
args = parser.parse_args()

rank_rslt = {}
rank_cnt = defaultdict(int)
with open(f"log.{args.name}.{args.mode}.txt", "r") as f:
    for line in f:
        z = re.match(
            r"\[(\d+)\] \[.*\] cnt: (\d+).lang: all.prec@1: (0.\d+).prec@3: (0.\d+).prec@5: (0.\d+).prec@10: (0.\d+).prec@20: (0.\d+).*",
            line,
        )

        if z is None:
            continue
        rank, cnt, prec1, prec3, prec5, prec10, prec20 = z.groups()
        rank, cnt = int(rank), int(cnt)
        prec1, prec3, prec5, prec10, prec20 = [
            float(i) for i in [prec1, prec3, prec5, prec10, prec20]
        ]

        if cnt > rank_cnt[rank]:
            rank_cnt[rank] = cnt
            rank_rslt[rank] = (prec1, prec3, prec5, prec10, prec20)


rslt = [rank_rslt[rank] for rank in rank_rslt]
rslt_mean = []
for metrics in zip(*rslt):
    rslt_mean.append(np.mean(metrics))


print("\t".join([f"{metric:.4f}" for metric in rslt_mean]))
