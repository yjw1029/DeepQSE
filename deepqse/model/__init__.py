from .toptf import TopTFModel
from .toptf_load import TopTFModelLoad
from .two_step import TwoStepModel
from .topk import TopKModel

from .single_speed import *


def get_model(args):
    return eval(args.model_name)(args)
