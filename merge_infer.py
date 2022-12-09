from pathlib import Path
import re
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--infer_mode", type=str, choices=["train", "dev", "test"])
args = parser.parse_args()

infer_path = Path.home() / "blob/DeepExtract/infer" / args.name / args.infer_mode

rank2file = {}
for file in infer_path.iterdir():
    f = open(file, "r")
    match_obj = re.match(r"^result.(\d)+.tsv$", file.name)

    if match_obj:
        rank = int(match_obj.group(1))
        rank2file[rank] = f

f_out = open(
    Path.home() / "blob/DeepExtract/infer" / args.name / args.infer_mode / "result.tsv",
    "w",
)

line_num = 0
read_flag = {k: False for k in rank2file}
file_end_num = 0
while True:
    for i in range(len(read_flag)):
        if read_flag[i]:
            continue
        line = rank2file[i].readline()

        if line == "":
            rank2file[i].close()
            read_flag[i] = True
            file_end_num += 1
            continue

        num, _ = line.split("\t")

        if int(num) != line_num:
            rank2file[i].close()
            read_flag[i] = True
            file_end_num += 1
            continue

        line_num += 1
        f_out.write(line)

    if file_end_num == len(read_flag):
        break

f_out.close()
