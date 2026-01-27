import argparse
import os
import csv

import time
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers.optimization import AdamW

from utils import Logger, set_all_seeds
from dataset_pretrain import PretrainDataset
from dataset import Dataset
from modeling_strats import Strats
from modeling_gru import GRU_TS
from modeling_tcn import TCN_TS
from modeling_sand import SAND
from modeling_grud import GRUD_TS
from modeling_interpnet import InterpNet
import numpy as np
from tqdm import tqdm
from transformers.optimization import AdamW
from models import count_parameters
from evaluator import Evaluator
from evaluator_pretrain import PretrainEvaluator
from pathlib import Path


def get_repo_root() -> Path:
    # .../strats-development/src/main.py -> .../strats-development
    return Path(__file__).resolve().parent.parent


def get_results_base_dir() -> Path:
    # results folder under strats-development
    return get_repo_root() / "results"


def infer_perturbation_from_file(file_name: str) -> str:
    """
    Expected file format:
      <dataset>_<perturbation>_<pct>_<seed>
    Examples:
      physionet_2012_subsampled_10_0
      mimic_iii_sparsified-patientwise_50_0
      mimic_iii_sparsified-tsid-varid_90_0
    """
    parts = file_name.split("_")
    if len(parts) < 4:
        return "unknown"

    # dataset is usually first two parts: physionet_2012 / mimic_iii
    rest = parts[2:]
    if len(rest) < 3:
        return "unknown"

    perturbation = "_".join(rest[:-2])  # everything except pct and seed
    return perturbation


def get_experiment_dir(args) -> Path:
    dataset = getattr(args, "dataset", "unknown_dataset")
    target = getattr(args, "target", "unknown_target")
    model = getattr(args, "model_type", "unknown_model")
    file_name = getattr(args, "file", "unknown_file")

    perturb = "unknown_perturb"
    try:
        perturb = infer_perturbation_from_file(file_name)
    except Exception:
        pass

    base = get_results_base_dir()
    exp_dir = base / dataset / target / model / perturb / file_name
    return exp_dir


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--dataset', type=str, default='physionet_2012')
    parser.add_argument('--file', type=str, default='physionet_2012')
    parser.add_argument('--train_frac', type=float, default=0.5)
    parser.add_argument('--run', type=str, default='1o10')
    parser.add_argument('--target', type=str, default='in_hospital_mortality')
    # model related arguments
    parser.add_argument('--model_type', type=str, default='strats',
                        choices=['gru', 'tcn', 'sand', 'grud', 'interpnet',
                                 'strats', 'istrats'])
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

    exp_dir = get_experiment_dir(args)
    # Force everything downstream to use this experiment folder
    args.output_dir = str(exp_dir)
    args.output_dir_prefix = str(exp_dir)
    return args


def set_output_dir(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"

    file_tag = args.file
    parts = file_tag.split("_")
    perturbation = "_".join(parts[2:-2]) if len(parts) >= 5 else (parts[2] if len(parts) > 2 else "none")

    target = getattr(args, "target", "in_hospital_mortality")

    exp_dir = results_root / str(args.dataset) / str(target) / str(args.model_type) / str(perturbation) / str(file_tag)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Keep output_dir for anything that uses it
    args.output_dir = str(exp_dir)

    # Point to file paths (not folders)
    args.log_path = str(exp_dir / "log.txt")  # use Logger(args.log_path)
    args.best_ckpt_path = str(exp_dir / "best.pt")
    args.last_ckpt_path = str(exp_dir / "last.pt")


def collect_paper_hyperparams(args):
    """
    Returns a dict of hyperparameters (columns) for saving to results CSV.
    Safe: if an attribute doesn't exist in args, we store None.
    """
    keys = [
        # common
        "model_type", "hid_dim", "dropout", "attention_dropout", "lr",
        "num_layers", "num_heads", "kernel_size", "r", "M",
        "max_timesteps", "hours_look_ahead", "ref_points",
        "train_batch_size", "eval_batch_size", "gradient_accumulation_steps",
        "max_epochs", "patience", "seed", "train_frac", "run", "target",
        "max_obs", "pretrain", "output_dir", "output_dir_prefix",
        "load_ckpt_path",
    ]
    out = {}
    for k in keys:
        out[k] = getattr(args, k, None)
    return out


def save_results_csv(args, best_val_res, best_test_res):
    """
    Save best validation and test results to a CSV file (append mode).
    One row = one experiment run.

    Folder structure:
      results/<dataset>/<target>/<model>/<perturbation>/<file>/<file>.csv
    """

    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"

    file_tag = args.file
    parts = file_tag.split("_")

    perturbation = "_".join(parts[2:-2]) if len(parts) >= 5 else (parts[2] if len(parts) > 2 else "none")
    pct = int(parts[-2]) if len(parts) >= 2 and parts[-2].isdigit() else None
    seed = int(parts[-1]) if len(parts) >= 1 and parts[-1].isdigit() else args.seed

    exp_dir = (
            results_root
            / str(args.dataset)
            / str(getattr(args, "target", "in_hospital_mortality"))
            / str(args.model_type)
            / str(perturbation)
            / str(file_tag)
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = exp_dir / f"{args.file}.csv"

    # ---------- build row ----------
    row = {
        "dataset": args.dataset,
        "target": getattr(args, "target", None),
        "model": args.model_type,
        "perturbation": perturbation,
        "file": file_tag,
        "pct": pct,
        "seed": seed,

        # NEW: run meta
        "start_time": getattr(args, "run_start_time", None),
        "end_time": getattr(args, "run_end_time", None),
        "duration_sec": getattr(args, "run_duration_sec", None),
        "device": getattr(args, "device_str", str(getattr(args, "device", None))),

        # basic hparams
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
    # Preliminary setup.
    args = parse_args()

    # infer perturbation early (from args.file)
    perturbation = infer_perturbation_from_file(args.file)

    # enforce target rule
    if perturbation == "unbalanced":
        args.target = "length_of_stay"
    else:
        args.target = "in_hospital_mortality"

    args.run_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.run_start_ts = time.time()

    set_output_dir(args)
    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n' + str(args))
    # args.device = torch.device('cuda')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_str = str(args.device)
    set_all_seeds(args.seed + int(args.run.split('o')[0]))
    model_path_best = os.path.join(args.output_dir, 'checkpoint_best.bin')

    # load data
    dataset = PretrainDataset(args) if args.pretrain == 1 else Dataset(args)

    # load model
    model_class = {'strats': Strats, 'istrats': Strats, 'gru': GRU_TS, 'tcn': TCN_TS,
                   'sand': SAND, 'grud': GRUD_TS, 'interpnet': InterpNet}
    model = model_class[args.model_type](args)
    model.to(args.device)
    count_parameters(args.logger, model)
    if args.load_ckpt_path is not None:
        curr_state_dict = model.state_dict()
        pt_state_dict = torch.load(args.load_ckpt_path)
        for k, v in pt_state_dict.items():
            if k in curr_state_dict:
                curr_state_dict[k] = v
        model.load_state_dict(curr_state_dict)

    # training loop
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = num_train / args.train_batch_size
    args.logger.write('\nNo. of training batches per epoch = '
                      + str(num_batches_per_epoch))
    args.max_steps = int(round(num_batches_per_epoch) * args.max_epochs)
    if args.validate_every is None:
        args.validate_every = int(np.ceil(num_batches_per_epoch))
    cum_train_loss, num_steps, num_batches_trained = 0, 0, 0
    wait, patience_reached = args.patience, False
    best_val_metric = -np.inf
    best_val_res, best_test_res = None, None
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    train_bar = tqdm(range(args.max_steps))
    evaluator = PretrainEvaluator(args) if args.pretrain == 1 else Evaluator(args)

    # results before any training
    if args.validate_after < 0:
        results = evaluator.evaluate(model, dataset, 'val', train_step=-1)
        if not (args.pretrain):
            evaluator.evaluate(model, dataset, 'eval_train', train_step=-1)
            evaluator.evaluate(model, dataset, 'test', train_step=-1)

    model.train()
    for step in train_bar:
        # Load batch
        batch = dataset.get_batch()
        batch = {k: v.to(args.device) for k, v in batch.items()}

        # Forward
        loss = model(**batch)

        # Backward
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Track loss
        cum_train_loss += loss.item()
        num_steps += 1
        num_batches_trained += 1

        # Log training losses.
        train_bar.set_description(str(np.round(cum_train_loss / num_batches_trained, 5)))
        if (num_steps) % args.print_train_loss_every == 0:
            args.logger.write('\nTrain-loss at step ' + str(num_steps) + ': '
                              + str(cum_train_loss / num_batches_trained))
            cum_train_loss, num_batches_trained = 0, 0

        # run validatation
        if (num_steps >= args.validate_after) and (num_steps % args.validate_every == 0):
            # get metrics on test and validation splits
            val_res = evaluator.evaluate(model, dataset, 'val', train_step=step)
            if not (args.pretrain):
                evaluator.evaluate(model, dataset, 'eval_train', train_step=step)
                test_res = evaluator.evaluate(model, dataset, 'test', train_step=step)
            else:
                test_res = None

            model.train(True)

            # Save ckpt if there is an improvement.
            curr_val_metric = val_res['loss_neg'] if args.pretrain \
                else val_res['auprc'] + val_res['auroc']
            if curr_val_metric > best_val_metric:
                best_val_metric = curr_val_metric
                best_val_res, best_test_res = val_res, test_res

                args.logger.write("\nSaving ckpt at " + model_path_best)
                torch.save(model.state_dict(), model_path_best)
                wait = args.patience
            else:
                wait -= 1
                args.logger.write('Updating wait to ' + str(wait))
                if wait == 0:
                    args.logger.write('Patience reached')
                    break

    # print final res
    args.logger.write('Final val res: ' + str(best_val_res))
    args.logger.write('Final test res: ' + str(best_test_res))

if not args.pretrain:
    args.run_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.run_duration_sec = round(time.time() - args.run_start_ts, 3)
    save_results_csv(args, best_val_res, best_test_res)
