from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig

import os
import sys
import logging
import torch
import torch.distributed as dist
import numpy as np
import argparse
from collections import OrderedDict

from deepqse.model_utils import CrossRobertaModel, CrossBertModel


class MiniTULRTokenizer(XLMRobertaTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        special_token_file = os.path.join(
            os.path.dirname(pretrained_model_name_or_path), "special_tokens.txt"
        )

        special_tokens_dict = {"additional_special_tokens": []}

        with open(special_token_file, "r") as f:
            for line in f:
                word = line.strip()
                special_tokens_dict["additional_special_tokens"].append(word)

        num_added_tokes = tokenizer.add_special_tokens(special_tokens_dict)
        logging.info(f"add {num_added_tokes} special tokens.")

        return tokenizer


class MiniTULRModel(XLMRobertaModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        prefix=None,
        strip_prefix=True,
        **kwargs,
    ):
        model_dict = torch.load(pretrained_model_name_or_path)
        """add prefix and strip_prefix for partially load statedict"""
        if prefix is not None:
            new_model_dict = OrderedDict()
            for key in model_dict:
                if key.startswith(prefix):
                    if strip_prefix:
                        new_model_dict[key[len(prefix) :]] = model_dict[key]
                    else:
                        new_model_dict[key] = model_dict[key]
        else:
            new_model_dict = model_dict

        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            state_dict=new_model_dict,
            **kwargs,
        )


MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "tulr": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "minitulr": (XLMRobertaConfig, MiniTULRModel, MiniTULRTokenizer),
    "cross-minitulr": (XLMRobertaConfig, CrossRobertaModel, MiniTULRTokenizer),
    "cross-bert": (BertConfig, CrossBertModel, BertTokenizer),
}


def cleanup():
    dist.destroy_process_group()


def setup(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.nccl_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # intialize logging
    setuplogger(rank, args)

    torch.cuda.set_device(rank)
    logging.info(args)


def setuplogger(rank, args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[{rank}] [%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    fh = logging.FileHandler(f"log.{args.name}.{args.mode}.txt")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split(".")[-2].split("-")[-1]): x for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory, all_checkpoints[max(all_checkpoints.keys())])


def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None
