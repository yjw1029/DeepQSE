import argparse
import re
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--mode", type=str, default="judge")
args = parser.parse_args()

rank_rslt = {}
rank_cnt = defaultdict(int)
with open(f"log.{args.name}.{args.mode}.txt", "r") as f:
    for line in f:
        z = re.match(
            r"\[(\d+)\] \[.*\] \[(\d+)\] judge result: (0.\d+).*",
            line,
        )

        if z is None:
            continue
        rank, cnt, acc = z.groups()
        rank, cnt = int(rank), int(cnt)
        acc = float(acc)

        if cnt >= rank_cnt[rank]:
            rank_cnt[rank] = cnt
            rank_rslt[rank] = (acc,)


rslt = [rank_rslt[rank] for rank in rank_rslt]
rslt_mean = []
for metrics in zip(*rslt):
    rslt_mean.append(np.mean(metrics))


print("\t".join([f"{metric:.4f}" for metric in rslt_mean]))
with open(f"log.{args.name}.{args.mode}.txt", "a") as f:
    f.write("\t".join([f"{metric:.4f}" for metric in rslt_mean]) + "\n")
