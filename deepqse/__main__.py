import torch.multiprocessing as mp
import torch
import sys
from pathlib import Path
import shutil

from .parameters import parse_args
from .main import train, test, speed, infer, combine_test, judge

args = parse_args()
# print(sys.path)
# exit()

# make save_skpt_dir if not exist
save_ckpt_dir = Path(args.save_ckpt_dir) / args.name
save_ckpt_dir.mkdir(exist_ok=True, parents=True)

n_gpus = torch.cuda.device_count()
assert (
    n_gpus >= args.world_size
), f"Requires at least {args.word_size} GPUs to run, but got {n_gpus}"

if args.mode == "train":
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
elif args.mode == "test":
    mp.spawn(test, args=(args,), nprocs=args.world_size, join=True)
elif args.mode == "infer":
    infer_path = Path(args.infer_path) / args.name / args.infer_mode
    if infer_path.exists() and infer_path.is_dir():
        shutil.rmtree(infer_path)
    infer_path.mkdir(exist_ok=True, parents=True)
    mp.spawn(infer, args=(args,), nprocs=args.world_size, join=True)
elif args.mode == "combine_test":
    mp.spawn(combine_test, args=(args,), nprocs=args.world_size, join=True)
elif args.mode == "speed":
    speed(args)
elif args.mode == "judge":
    mp.spawn(judge, args=(args,), nprocs=args.world_size, join=True)
