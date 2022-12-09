import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.optim as optim
import transformers

from torch.cuda.amp import GradScaler, autocast

from deepqse import utils
from deepqse.model import get_model
from deepqse.data import get_dataset
from deepqse.utils import setup, cleanup
from deepqse.metrics import acc, PrecMetric, MetricPool

# try:
#     from apex.optimizers import FP16_Optimizer
#     from apex.optimizers import FusedAdam
# except ImportError:
#     raise ImportError(
#         "Please install apex from https://www.github.com/nvidia/apex to run this."
#     )


def train(rank, args):
    setup(rank, args)

    save_ckpt_dir = Path(args.save_ckpt_dir) / args.name

    logging.info(f"[+] start initializing {args.dataset_name} dataloader")
    data_path = Path(args.data_path)

    ds_cls, collate_cls = get_dataset(args)
    ds = ds_cls(args, data_path=data_path / args.train_file)
    sampler = DistributedSampler(ds, drop_last=True, shuffle=True)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_cls(args),
        num_workers=args.num_workers,
    )
    logging.info(f"[-] finish initializing {args.dataset_name} dataloader")

    logging.info(f"[+] start initializing {args.model_name} model")
    if args.enable_gpu:
        model = get_model(args).to(rank)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        # Only support for testing
        model = get_model(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        ddp_model = model

    logging.info(f"[-] finish initializing {args.model_name} model")

    scaler = GradScaler()

    ddp_model.train()
    for ep in range(args.epoch_num):
        loss = 0
        accuary = 0

        for cnt, (data, _) in enumerate(dl):
            optimizer.zero_grad()

            if args.enable_gpu:
                for key in data:
                    if isinstance(
                        data[key], transformers.tokenization_utils_base.BatchEncoding
                    ) or isinstance(data[key], dict):
                        for key2 in data[key]:
                            # print(key, key2, data[key][key2].shape)
                            data[key][key2] = data[key][key2].cuda()
                    else:
                        # print(key, data[key].shape)
                        data[key] = data[key].cuda()

            with autocast():
                bz_loss, prob = ddp_model(data)
                loss += bz_loss.data.float()
                accuary += acc(data["start_indexes"], prob)

            scaler.scale(bz_loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            if (cnt + 1) % args.log_steps == 0:
                logging.info(
                    "Ed: {}, train_loss: {:.5f}, acc: {:.5f}".format(
                        (cnt + 1) * args.batch_size,
                        loss.data / (cnt + 1),
                        accuary / (cnt + 1),
                    )
                )

            if rank == 0 and cnt % args.save_steps == 0:
                ckpt_path = save_ckpt_dir / f"epoch-{ep+1}-{cnt}.pt"
                torch.save(
                    {
                        "model_state_dict": ddp_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                logging.info(f"Model saved to {ckpt_path}")

        if rank == 0:
            ckpt_path = save_ckpt_dir / f"epoch-{ep+1}.pt"
            torch.save(
                {
                    "model_state_dict": ddp_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            logging.info(f"Model saved to {ckpt_path}")

        # evaluate model each epoch
        # evaluate(rank, args.dev_file, args, ddp_model)

    cleanup()


def evaluate(rank, test_file, args, ddp_model):
    logging.info("[+] start initializing dataloader")
    data_path = Path(args.data_path)

    ds_cls, collate_cls = get_dataset(args)
    ds = ds_cls(args, data_path=data_path / test_file)
    sampler = DistributedSampler(ds, shuffle=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_cls(args),
        num_workers=args.num_workers,
    )
    logging.info("[-] finish initializing dataloader")

    metric_pool = MetricPool()
    metric_pool.regist(PrecMetric(name="prec", k=1))
    metric_pool.regist(PrecMetric(name="prec", k=3))
    metric_pool.regist(PrecMetric(name="prec", k=10))
    metric_pool.regist(PrecMetric(name="prec", k=20))

    infer_time = 0
    ddp_model.eval()
    for cnt, (data, langs) in enumerate(dl):
        if args.enable_gpu:
            for key in data:
                if isinstance(
                    data[key], transformers.tokenization_utils_base.BatchEncoding
                ) or isinstance(data[key], dict):
                    for key2 in data[key]:
                        # print(key, key2, data[key][key2].shape)
                        data[key][key2] = data[key][key2].cuda()
                else:
                    # print(key, data[key].shape)
                    data[key] = data[key].cuda()

        start_time = time.time()

        with autocast():
            probs = ddp_model(data, compute_loss=False)

        end_time = time.time()
        infer_time += end_time - start_time

        for y_true, y_hat, lang in zip(data["start_indexes"], probs, langs):
            metric_pool.update(y_true, y_hat, lang)

        if (cnt + 1) % args.log_steps == 0:
            metric_pool.report()

    metric_pool.report()
    cleanup()


def test(rank, args):
    setup(rank, args)

    logging.info("[+] start initializing dataloader")
    data_path = Path(args.data_path)

    ds_cls, collate_cls = get_dataset(args)
    ds = ds_cls(args, data_path=data_path / args.test_file)
    sampler = DistributedSampler(ds, shuffle=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_cls(args),
        num_workers=args.num_workers,
    )
    logging.info("[-] finish initializing dataloader")

    save_ckpt_dir = Path(args.save_ckpt_dir) / args.name

    logging.info(f"[+] start loading {args.model_name} model from {save_ckpt_dir}")

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(save_ckpt_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(save_ckpt_dir)

    assert ckpt_path is not None, "No ckpt found"

    if args.enable_gpu:
        model = get_model(args).to(rank)
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        # Only support for testing
        ddp_model = get_model(args)

    if args.enable_gpu:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        ddp_model.load_state_dict(
            torch.load(ckpt_path, map_location=map_location)["model_state_dict"]
        )

    logging.info(f"[-] finish loading {args.model_name} model from {ckpt_path}")

    metric_pool = MetricPool()
    metric_pool.regist(PrecMetric(name="prec", k=1))
    metric_pool.regist(PrecMetric(name="prec", k=3))
    metric_pool.regist(PrecMetric(name="prec", k=5))
    metric_pool.regist(PrecMetric(name="prec", k=10))
    metric_pool.regist(PrecMetric(name="prec", k=20))

    infer_time = 0
    ddp_model.eval()
    for cnt, (data, langs) in enumerate(dl):
        if args.enable_gpu:
            for key in data:
                if isinstance(
                    data[key], transformers.tokenization_utils_base.BatchEncoding
                ) or isinstance(data[key], dict):
                    for key2 in data[key]:
                        # print(key, key2, data[key][key2].shape)
                        data[key][key2] = data[key][key2].cuda()
                else:
                    # print(key, data[key].shape)
                    data[key] = data[key].cuda()

        start_time = time.time()

        with autocast():
            probs = ddp_model(data, compute_loss=False)

        end_time = time.time()
        infer_time += end_time - start_time

        for y_true, y_hat, lang in zip(data["start_indexes"], probs, langs):
            metric_pool.update(y_true, y_hat, lang)

        if (cnt + 1) % args.log_steps == 0:
            metric_pool.report()

    metric_pool.report()
    logging.info(f"time: {infer_time}")

    time.sleep(30)
    cleanup()


def infer(rank, args):
    from deepqse.data.infer import InferDataset, InferCollate

    # used before kd to generate soft labels of teacher model
    setup(rank, args)

    logging.info("[+] start initializing dataloader")
    data_path = Path(args.data_path)

    if args.infer_mode == "train":
        ds = InferDataset(args, data_path=data_path / args.train_file)
    elif args.infer_mode == "test":
        ds = InferDataset(args, data_path=data_path / args.test_file)
    elif args.infer_mode == "dev":
        ds = InferDataset(args, data_path=data_path / args.dev_file)
    collate = InferCollate(args)

    sampler = DistributedSampler(ds, shuffle=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    logging.info("[-] finish initializing dataloader")

    save_ckpt_dir = Path(args.save_ckpt_dir) / args.name

    logging.info(f"[+] start loading {args.model_name} model from {save_ckpt_dir}")
    model = get_model(args).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(save_ckpt_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(save_ckpt_dir)

    assert ckpt_path is not None, "No ckpt found"

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    ddp_model.load_state_dict(
        torch.load(ckpt_path, map_location=map_location)["model_state_dict"]
    )

    logging.info(f"[+] finish loading {args.model_name} model from {ckpt_path}")

    infer_time = 0
    infer_file = (
        Path(args.infer_path) / args.name / args.infer_mode / f"result.{rank}.tsv"
    )

    lines = []
    ddp_model.eval()
    for cnt, (data, langs, indexes) in enumerate(dl):
        if args.enable_gpu:
            for key in data:
                if isinstance(
                    data[key], transformers.tokenization_utils_base.BatchEncoding
                ) or isinstance(data[key], dict):
                    for key2 in data[key]:
                        # print(key, key2, data[key][key2].shape)
                        data[key][key2] = data[key][key2].cuda()
                else:
                    # print(key, data[key].shape)
                    data[key] = data[key].cuda()

        start_time = time.time()
        with autocast():
            probs = ddp_model(data, compute_loss=False)
        # torch.cuda.synchronize()
        end_time = time.time()
        infer_time += end_time - start_time

        for y_hat, sent_num, index in zip(probs, data["sents_num"].cpu(), indexes):
            y_hat = y_hat[:sent_num]
            y_hat_line = " ".join([f"{score:.4f}" for score in y_hat])
            line = f"{index}\t{y_hat_line}\n"
            lines.append(line)

        if (cnt + 1) % args.log_steps == 0:
            logging.info(f"finish inference {cnt + 1} batch")
            with open(infer_file, "a") as f:
                f.writelines(lines)
            lines = []

    logging.info(f"finish inference {cnt + 1} batch")
    with open(infer_file, "a") as f:
        f.writelines(lines)
    lines = []

    logging.info(f"time: {infer_time}")

    cleanup()


def combine_test(rank, args):
    setup(rank, args)

    logging.info("[+] start initializing dataloader")
    data_path = Path(args.data_path)

    ds_cls, collate_cls = get_dataset(args)
    ds = ds_cls(args, data_path=data_path / args.test_file)
    sampler = DistributedSampler(ds, shuffle=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_cls(args),
        num_workers=args.num_workers,
    )
    logging.info("[-] finish initializing dataloader")

    save_ckpt_dir = Path(args.save_ckpt_dir) / args.name

    logging.info(f"[+] start loading {args.model_name} model from {save_ckpt_dir}")
    model = get_model(args).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(save_ckpt_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(save_ckpt_dir)

    assert ckpt_path is not None, "No ckpt found"

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    ddp_model.load_state_dict(
        torch.load(ckpt_path, map_location=map_location)["model_state_dict"]
    )

    logging.info(f"[+] finish loading {args.model_name} model from {ckpt_path}")

    metric_pool = MetricPool()
    metric_pool.regist(PrecMetric(name="prec", k=1))
    metric_pool.regist(PrecMetric(name="prec", k=3))
    metric_pool.regist(PrecMetric(name="prec", k=5))
    metric_pool.regist(PrecMetric(name="prec", k=10))
    metric_pool.regist(PrecMetric(name="prec", k=20))

    infer_time = 0
    ddp_model.eval()
    for cnt, (data, langs) in enumerate(dl):
        if args.enable_gpu:
            for key in data:
                if isinstance(
                    data[key], transformers.tokenization_utils_base.BatchEncoding
                ) or isinstance(data[key], dict):
                    for key2 in data[key]:
                        # print(key, key2, data[key][key2].shape)
                        data[key][key2] = data[key][key2].cuda()
                elif isinstance(data[key], list) or isinstance(data[key], tuple):
                    continue
                else:
                    # print(key, data[key].shape)
                    data[key] = data[key].cuda()

        start_time = time.time()
        with autocast():
            probs = ddp_model(data, compute_loss=False)
        # torch.cuda.synchronize()
        end_time = time.time()
        infer_time += end_time - start_time

        for y_true, y_hat, raw_sents_num, top_indices, lang in zip(
            data["start_indexes"].detach().cpu(),
            probs.detach().cpu(),
            data["raw_sents_num"],
            data["top_indices"],
            langs,
        ):
            dummy_y_hat = torch.ones(raw_sents_num).float() * -1e4
            dummy_y_hat[top_indices] = y_hat[: len(top_indices)]
            metric_pool.update(y_true, dummy_y_hat, lang)

        if (cnt + 1) % args.log_steps == 0:
            metric_pool.report()

    metric_pool.report()
    cleanup()
    logging.info(f"time: {infer_time}")


def speed(args, rank=0):
    # TODO: for speed test
    setup(rank, args)

    logging.info("[+] start initializing dataloader")
    data_path = Path(args.data_path)

    ds_cls, collate_cls = get_dataset(args)
    ds = ds_cls(args, data_path=data_path / args.test_file)
    sampler = DistributedSampler(ds, shuffle=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_cls(args),
        num_workers=args.num_workers,
    )
    logging.info("[-] finish initializing dataloader")

    save_ckpt_dir = Path(args.save_ckpt_dir) / args.name

    logging.info(f"[+] start loading {args.model_name} model from {save_ckpt_dir}")

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(save_ckpt_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(save_ckpt_dir)

    assert ckpt_path is not None, "No ckpt found"

    if args.enable_gpu:
        model = get_model(args).to(rank)
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        # Only support for testing
        ddp_model = get_model(args)

    if args.enable_gpu:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        ddp_model.load_state_dict(
            torch.load(ckpt_path, map_location=map_location)["model_state_dict"]
        )

    logging.info(f"[-] finish loading {args.model_name} model from {ckpt_path}")

    infer_time = 0
    global_steps = 0
    ddp_model.eval()
    for cnt, (data, langs) in enumerate(dl):
        if args.enable_gpu:
            for key in data:
                if isinstance(
                    data[key], transformers.tokenization_utils_base.BatchEncoding
                ) or isinstance(data[key], dict):
                    for key2 in data[key]:
                        # print(key, key2, data[key][key2].shape)
                        data[key][key2] = data[key][key2].cuda()
                else:
                    # print(key, data[key].shape)
                    data[key] = data[key].cuda()

        global_steps += data["start_indexes"].shape[0]
        start_time = time.time()
        _ = ddp_model(data, compute_loss=False)

        end_time = time.time()
        infer_time += end_time - start_time

        if (cnt + 1) % args.log_steps == 0:
            logging.info(f"inference time: {infer_time / global_steps} s")

    logging.info(f"time: {infer_time}")


def judge(rank, args):
    setup(rank, args)

    logging.info("[+] start initializing dataloader")
    data_path = Path(args.data_path)

    ds_cls, collate_cls = get_dataset(args)
    ds = ds_cls(args, data_path=data_path / args.test_file)
    sampler = DistributedSampler(ds, shuffle=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_cls(args),
        num_workers=args.num_workers,
    )
    logging.info("[-] finish initializing dataloader")

    save_ckpt_dir = Path(args.save_ckpt_dir) / args.name

    logging.info(f"[+] start loading {args.model_name} model from {save_ckpt_dir}")

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(save_ckpt_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(save_ckpt_dir)

    assert ckpt_path is not None, "No ckpt found"

    if args.enable_gpu:
        model = get_model(args).to(rank)
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        # Only support for testing
        ddp_model = get_model(args)

    if args.enable_gpu:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        ddp_model.load_state_dict(
            torch.load(ckpt_path, map_location=map_location)["model_state_dict"]
        )

    logging.info(f"[-] finish loading {args.model_name} model from {ckpt_path}")

    acc = 0.0
    cnt = 0
    ddp_model.eval()
    for step, (data, langs) in enumerate(dl):
        if args.enable_gpu:
            for key in data:
                if isinstance(
                    data[key], transformers.tokenization_utils_base.BatchEncoding
                ) or isinstance(data[key], dict):
                    for key2 in data[key]:
                        # print(key, key2, data[key][key2].shape)
                        data[key][key2] = data[key][key2].cuda()
                else:
                    data[key] = data[key].cuda()

        with autocast():
            probs = ddp_model(data, compute_loss=False).detach().cpu()

        for ref_index, test_index, probs, label in zip(
            data["ref_start_indexes"], data["test_start_indexes"], probs, data["labels"]
        ):
            ref_prob = probs[ref_index]
            test_prob = probs[test_index]

            acc_score = (ref_prob > test_prob) * (1 - label) + (
                ref_prob < test_prob
            ) * label
            acc += acc_score
            cnt += 1

        if (step + 1) % args.log_steps == 0:
            logging.info(f"[{step+1}] judge result: {acc / cnt:.4f}")

    logging.info(f"[{step+1}] judge result: {acc / cnt:.4f}")
    time.sleep(10)
    cleanup()
