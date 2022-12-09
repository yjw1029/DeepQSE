import argparse
import logging

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--nccl_port", type=str, default="12355")
    parser.add_argument("--enable_gpu", type=str2bool, default=True)
    parser.add_argument("--enable_ddp", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=True)

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "infer", "combine_test", "speed", "judge"],
    )
    parser.add_argument(
        "--data_path", type=str, default="/home/v-jingweiyi/blob/DeepExtract/data"
    )
    parser.add_argument(
        "--train_file", type=str, default="Alldata.en.train.withTitle.tsv"
    )
    parser.add_argument("--dev_file", type=str, default="Alldata.en.dev.withTitle.tsv")
    parser.add_argument(
        "--test_file", type=str, default="Alldata.en.test.withTitle.tsv"
    )
    # parser.add_argument("--log_file", type=str, default="log.txt")
    parser.add_argument("--bert_name", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="tulr")
    parser.add_argument(
        "--bert_path",
        type=str,
        default="/home/v-jingweiyi/blob/DeepExtract/DeepExtract_RankLMv6_6/",
    )
    parser.add_argument("--config_name", type=str, default="config.json")
    parser.add_argument("--model_name_or_path", type=str, default="pytorch_model.bin")
    parser.add_argument("--tokenizer_name", type=str, default="sentencepiece.bpe.model")
    parser.add_argument("--bert_output_layer", type=int, default=6)
    parser.add_argument(
        "--save_ckpt_dir", type=str, default="/home/v-jingweiyi/blob/DeepExtract/model"
    )

    parser.add_argument("--max_sent_num", type=int, default=160)
    parser.add_argument("--max_sent_length", type=int, default=64)
    parser.add_argument("--max_title_length", type=int, default=32)

    parser.add_argument("--epoch_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--load_ckpt_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--model_name", type=str, default="DESingleModel")
    parser.add_argument("--dataset_name", type=str, default="NegSamDataset")
    parser.add_argument("--collate_name", type=str, default="DECollate")
    parser.add_argument("--name", type=str, default="")

    parser.add_argument("--npratio", type=int, default=9)

    # for the first step of two steps model
    parser.add_argument("--max_body_length", type=int, default=32)
    parser.add_argument("--step1_topk", type=int, default=20)

    kd_param = parser.add_argument_group(
        title="inference", description="Parameters used for inferencing on KD"
    )
    kd_param.add_argument(
        "--infer_path", type=str, default="/home/v-jingweiyi/blob/DeepExtract/infer"
    )
    kd_param.add_argument("--infer_mode", type=str, default="train")
    kd_param.add_argument("--teacher_name", type=str, default="")
    kd_param.add_argument("--kl_T", type=float, default=1.0)
    kd_param.add_argument("--beta", type=float, default=1.0)

    args = parser.parse_args()
    logging.info(args)

    return args
