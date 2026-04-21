import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import csv
import time
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.optim import AdamW

from src.utils.utils import Logger, set_all_seeds
from src.training.dataset_pretrain import PretrainDataset
from src.training.dataset import Dataset
from src.models.modeling_strats import Strats
from src.models.modeling_gru import GRU_TS
from src.models.modeling_tcn import TCN_TS
from src.models.modeling_sand import SAND
from src.models.modeling_grud import GRUD_TS
from src.models.modeling_interpnet import InterpNet

from src.models.models import count_parameters
from src.training.evaluator import Evaluator
from src.training.evaluator_pretrain import PretrainEvaluator

def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_results_base_dir() -> Path:
    return get_repo_root() / "results"

def parse_file_tag(file_name: str) -> dict:
    """
    Supports:
      physionet_2012_unbalanced_10
      physionet_2012_subsampled_10_0
      mimic_iii_sparsified-tsid-varid_90_3
      physionet_2012_fold_0_unbalanced_10
      mimic_iii_fold_2_unbalanced_1
    """
    parts = file_name.split("_")

    result = {
        "dataset": None,
        "fold": None,
        "perturbation": "unknown",
        "pct": None,
        "seed": None,
    }

    if file_name.startswith("physionet_2012"):
        result["dataset"] = "physionet_2012"
        rest = parts[2:]
    elif file_name.startswith("mimic_iii"):
        result["dataset"] = "mimic_iii"
        rest = parts[2:]
    else:
        result["dataset"] = parts[0] if parts else "unknown"
        rest = parts[1:]

    if len(rest) >= 4 and rest[0] == "fold" and rest[1].isdigit():
        result["fold"] = int(rest[1])
        tail = rest[2:]
    else:
        tail = rest

    if len(tail) >= 2 and tail[-1].isdigit() and tail[-2].isdigit():
        result["pct"] = int(tail[-2])
        result["seed"] = int(tail[-1])
        pert_parts = tail[:-2]
    elif len(tail) >= 1 and tail[-1].isdigit():
        result["pct"] = int(tail[-1])
        result["seed"] = None
        pert_parts = tail[:-1]
    else:
        pert_parts = tail

    if len(pert_parts) > 0:
        result["perturbation"] = "_".join(pert_parts)

    return result

def infer_perturbation_from_file(file_name: str) -> str:
    return parse_file_tag(file_name)["perturbation"]

def get_experiment_dir(args) -> Path:
    dataset = getattr(args, "dataset", "unknown_dataset")
    target = getattr(args, "target", "unknown_target")
    model = getattr(args, "model_type", "unknown_model")
    file_name = getattr(args, "file", "unknown_file")

    info = parse_file_tag(file_name)
    perturb = info["perturbation"]
    fold = info["fold"]

    base = get_results_base_dir()

    if fold is None:
        return base / dataset / target / model / perturb / file_name
    return base / dataset / target / model / perturb / f"fold_{fold}" / file_name

def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--dataset', type=str, default='physionet_2012')
    parser.add_argument('--file', type=str, default='physionet_2012')
    parser.add_argument('--train_frac', type=float, default=0.5)
    parser.add_argument('--run', type=str, default='1o10')
    parser.add_argument('--target', type=str, default=None)
    # model related arguments
    parser.add_argument('--model_type', type=str, default='strats',
                        choices=['gru','tcn','sand','grud','interpnet',
                                 'strats','istrats'])
    parser.add_argument('--load_ckpt_path', type=str, default=None)
    ##  strats and istrats
    parser.add_argument('--max_obs', type=int, default=880)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)
    ## gru: hid_dim, dropout
    ## tcn: dropout, filters=hid_dim
    parser.add_argument('--kernel_size', type=int, default=4)
    ## sand: num_layers, hid_dim, num_heads, dropout
    parser.add_argument('--r', type=int, default=24)
    parser.add_argument('--M', type=int, default=12)
    ## grud: hid_dim, dropout
    parser.add_argument('--max_timesteps', type=int, default=880)
    ## interpnet: hid_dim
    parser.add_argument('--hours_look_ahead', type=int, default=24)
    parser.add_argument('--ref_points', type=int, default=24)

    # training/eval realated arguments
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--print_train_loss_every', type=int, default=100)
    parser.add_argument('--validate_after', type=int, default=-1)
    parser.add_argument('--validate_every', type=int, default=None)

    args = parser.parse_args()
    args.run_start_ts = time.time()

    exp_dir = get_experiment_dir(args)
    # Force everything downstream to use this experiment folder
    if args.output_dir is None or args.output_dir == "":
        args.output_dir = str(exp_dir)

    if args.output_dir_prefix is None or args.output_dir_prefix == "":
        args.output_dir_prefix = str(exp_dir)

    return args


def set_output_dir(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"

    file_tag = args.file
    info = parse_file_tag(file_tag)

    perturbation = info["perturbation"]
    fold = info["fold"]

    target = getattr(args, "target", None) or "in_hospital_mortality"

    if fold is None:
        exp_dir = results_root / str(args.dataset) / str(target) / str(args.model_type) / str(perturbation) / str(file_tag)
    else:
        exp_dir = results_root / str(args.dataset) / str(target) / str(args.model_type) / str(perturbation) / f"fold_{fold}" / str(file_tag)

    prefix = (getattr(args, "output_dir_prefix", None) or "").strip()
    outdir = (getattr(args, "output_dir", None) or "").strip()
    base_dir = Path(prefix or outdir or exp_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir = str(base_dir)
    args.output_dir_prefix = str(base_dir)

    args.log_path = str(base_dir / "log.txt")
    args.best_ckpt_path = str(base_dir / "best.pt")
    args.last_ckpt_path = str(base_dir / "last.pt")

def collect_paper_hyperparams(args):
    """
    Returns a dict of hyperparameters for saving to results CSV.
    Safe: missing attributes become None.
    """
    keys = [
        "model_type", "hid_dim", "dropout", "attention_dropout", "lr",
        "num_layers", "num_heads", "kernel_size", "r", "M",
        "max_timesteps", "hours_look_ahead", "ref_points",
        "train_batch_size", "eval_batch_size", "gradient_accumulation_steps",
        "max_epochs", "patience", "seed", "train_frac", "run", "target",
        "max_obs", "pretrain", "output_dir", "output_dir_prefix",
        "load_ckpt_path",
    ]
    return {k: getattr(args, k, None) for k in keys}

def save_results_csv(args, best_val_res, best_test_res):
    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"

    file_tag = getattr(args, "file", "unknown_file")
    info = parse_file_tag(file_tag)

    perturbation = info["perturbation"]
    pct = info["pct"]
    seed = info["seed"] if info["seed"] is not None else getattr(args, "seed", None)
    fold = info["fold"]

    base_dir = Path(
        (getattr(args, "output_dir_prefix", None) or "").strip()
        or (getattr(args, "output_dir", None) or "").strip()
        or results_root
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f"{file_tag}.csv"

    row = {
        "dataset": getattr(args, "dataset", None),
        "target": getattr(args, "target", None),
        "model": getattr(args, "model_type", None),
        "perturbation": perturbation,
        "file": file_tag,
        "fold": fold,
        "pct": pct,
        "seed": seed,

        "start_time": getattr(args, "run_start_time", None),
        "end_time": getattr(args, "run_end_time", None),
        "duration_sec": getattr(args, "run_duration_sec", None),
        "device": getattr(args, "device_str", str(getattr(args, "device", None))),

        "lr": getattr(args, "lr", None),
        "hid_dim": getattr(args, "hid_dim", None),
        "dropout": getattr(args, "dropout", None),
        "attention_dropout": getattr(args, "attention_dropout", None),
        "num_layers": getattr(args, "num_layers", None),
        "num_heads": getattr(args, "num_heads", None),
        "kernel_size": getattr(args, "kernel_size", None),
        "r": getattr(args, "r", None),
        "M": getattr(args, "M", None),
        "train_frac": getattr(args, "train_frac", None),
        "max_epochs": getattr(args, "max_epochs", None),
        "patience": getattr(args, "patience", None),
    }

    # all paper hyperparameter columns (filled where possible, else None)
    if "collect_paper_hyperparams" in globals():
        row.update(collect_paper_hyperparams(args))

    # metrics
    if best_val_res is not None:
        for k, v in best_val_res.items():
            row[f"val_{k}"] = float(v) if hasattr(v, "__float__") else v

    if best_test_res is not None:
        for k, v in best_test_res.items():
            row[f"test_{k}"] = float(v) if hasattr(v, "__float__") else v

    # ---------- append to CSV ----------
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"✔ Results saved to {csv_path}")


if __name__ == "__main__":
   
    args = parse_args()
    perturbation = infer_perturbation_from_file(args.file)

    if args.target is None or args.target.strip() == "":
        args.target = "unbalanced" if perturbation == "unbalanced" else "in_hospital_mortality"

    args.run_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.run_start_ts = time.time()

    set_output_dir(args)

    args.logger = Logger(args.output_dir, "log.txt")
    args.logger.write("\n" + str(args))

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_str = str(args.device)
    set_all_seeds(args.seed + int(args.run.split("o")[0]))

    model_path_best = args.best_ckpt_path

    dataset = PretrainDataset(args) if args.pretrain == 1 else Dataset(args)

    model_class = {
        "strats": Strats,
        "istrats": Strats,
        "gru": GRU_TS,
        "tcn": TCN_TS,
        "sand": SAND,
        "grud": GRUD_TS,
        "interpnet": InterpNet,
    }
    model = model_class[args.model_type](args)
    model.to(args.device)
    count_parameters(args.logger, model)

   
    if args.load_ckpt_path is not None:
        curr_state_dict = model.state_dict()
        pt_state_dict = torch.load(args.load_ckpt_path, map_location="cpu")
        for k, v in pt_state_dict.items():
            if k in curr_state_dict:
                curr_state_dict[k] = v
        model.load_state_dict(curr_state_dict)

    num_train = len(dataset.splits["train"])

    # batches per epoch must be an integer >= 1
    num_batches_per_epoch = int(np.ceil(num_train / args.train_batch_size))
    num_batches_per_epoch = max(1, num_batches_per_epoch)

    args.logger.write("\nNo. of training batches per epoch = " + str(num_batches_per_epoch))

    args.max_steps = num_batches_per_epoch * args.max_epochs
    if args.validate_every is None:
        args.validate_every = num_batches_per_epoch

    cum_train_loss, num_steps, num_batches_trained = 0, 0, 0
    wait = args.patience
    best_val_metric = -np.inf
    best_val_res, best_test_res = None, None

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    train_bar = tqdm(range(args.max_steps))
    evaluator = PretrainEvaluator(args) if args.pretrain == 1 else Evaluator(args)

    if args.validate_after < 0:
        evaluator.evaluate(model, dataset, "val", train_step=-1)
        if not args.pretrain:
            evaluator.evaluate(model, dataset, "eval_train", train_step=-1)
            evaluator.evaluate(model, dataset, "test", train_step=-1)

    model.train()
    for step in train_bar:
        batch = dataset.get_batch()
        batch = {k: v.to(args.device) for k, v in batch.items()}

        loss = model(**batch)

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        cum_train_loss += loss.item()
        num_steps += 1
        num_batches_trained += 1

        train_bar.set_description(str(np.round(cum_train_loss / num_batches_trained, 5)))

        if num_steps % args.print_train_loss_every == 0:
            args.logger.write(
                "\nTrain-loss at step " + str(num_steps) + ": " + str(cum_train_loss / num_batches_trained)
            )
            cum_train_loss, num_batches_trained = 0, 0

        # validation
        if (num_steps >= args.validate_after) and (num_steps % args.validate_every == 0):
            val_res = evaluator.evaluate(model, dataset, "val", train_step=step)

            if not args.pretrain:
                evaluator.evaluate(model, dataset, "eval_train", train_step=step)
                test_res = evaluator.evaluate(model, dataset, "test", train_step=step)
            else:
                test_res = None

            model.train(True)

            curr_val_metric = val_res["loss_neg"] if args.pretrain else (val_res["auprc"] + val_res["auroc"])

            if curr_val_metric > best_val_metric:
                best_val_metric = curr_val_metric
                best_val_res, best_test_res = val_res, test_res

                args.logger.write("\nSaving ckpt at " + model_path_best)
                torch.save(model.state_dict(), model_path_best)
                wait = args.patience
            else:
                wait -= 1
                args.logger.write("Updating wait to " + str(wait))
                if wait == 0:
                    args.logger.write("Patience reached")
                    break

    args.logger.write("Final val res: " + str(best_val_res))
    args.logger.write("Final test res: " + str(best_test_res))

    if not args.pretrain:
        args.run_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        args.run_duration_sec = round(time.time() - args.run_start_ts, 3)
        save_results_csv(args, best_val_res, best_test_res)