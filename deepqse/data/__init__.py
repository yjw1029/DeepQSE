from .base import DEDataset, DECollate
from .negsam import NegSamDataset
from .pair import DEPairDataset, NegSamPairDataset, PairCollate
from .flatten import WholeBodyDataset, WholeBodyCollate
from .flatten_mean import WBMeanDataset, WBMeanCollate
from .two_step import TwoStepCollate
from .kd import KDPairDataset, KDPairCollate
from .topk import TopKDataset, TopKCollate
from .topk_test import TopKTestDataset, TopKTestCollate

from .title import *
from .single import *
from .baseline import *
from .judge import *

def get_dataset(args):
    return eval(args.dataset_name), eval(args.collate_name)
