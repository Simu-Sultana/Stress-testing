import argparse
import os
from utils import Logger, set_all_seeds
import torch
import pandas as pd
from dataset_pretrain import PretrainDataset
from dataset import Dataset
from modeling_strats import Strats
from modeling_gru import GRU_TS
from modeling_tcn import TCN_TS
from modeling_sand import SAND
from modeling_grud import GRUD_TS
from modeling_interpnet import InterpNet
import numpy as np
import time
from tqdm import tqdm
from transformers.optimization import AdamW
from models import count_parameters
from evaluator import Evaluator
from evaluator_pretrain import PretrainEvaluator


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

    # training/eval related arguments
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
    return args


def set_output_dir(args: argparse.Namespace) -> None:
    """Function to automatically set output dir if it is not passed in args."""
    if args.output_dir is None:
        if args.pretrain:
            args.output_dir = '../outputs/' + args.dataset + '/' + args.output_dir_prefix + 'pretrain/'
        else:
            if args.load_ckpt_path is not None:
                args.output_dir_prefix = 'finetune_' + args.output_dir_prefix
            args.output_dir = '../outputs/' + args.dataset + '/' + args.output_dir_prefix
            args.output_dir += args.model_type
            if args.model_type == 'strats':
                for param in ['num_layers', 'hid_dim', 'num_heads', 'dropout', 'attention_dropout', 'lr']:
                    args.output_dir += ',' + param + ':' + str(getattr(args, param))
            for param in ['train_frac', 'run']:
                args.output_dir += '|' + param + ':' + str(getattr(args, param))
    os.makedirs(args.output_dir, exist_ok=True)


def _get_results_dir() -> str:
    """
    Always save results under:
      strats-development/results/
    regardless of current working directory.
    This file lives in:
      strats-development/src/main.py
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))          # .../strats-development/src
    results_dir = os.path.normpath(os.path.join(base_dir, "..", "results"))  # .../strats-development/results
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _parse_perturbation_from_file_tag(file_tag: str, fallback_seed: int):
    """
    Expected patterns (best case):
      physionet_2012_<perturbation>_<pct>_<seed>
    But we handle missing parts gracefully.
    """
    parts = (file_tag or "").split("_")

    perturbation = "none"
    pct = None
    seed = fallback_seed

    # Example:
    # ["physionet", "2012", "sparsified", "10", "0"]
    if len(parts) > 2:
        perturbation = parts[2]

    if len(parts) > 3:
        try:
            pct = int(parts[3])
        except Exception:
            pct = None

    if len(parts) > 4:
        try:
            seed = int(parts[4])
        except Exception:
            seed = fallback_seed

    return perturbation, pct, seed


def save_results_csv(args, best_val_res, best_test_res,
                     total_sec: float, train_sec: float, eval_sec: float):
    """
    Save best validation and test results + timing to a CSV file (append mode).
    One row = one experiment run.
    CSV saved under: strats-development/results/
    """
    results_dir = _get_results_dir()

    file_tag = args.file
    perturbation, pct, seed = _parse_perturbation_from_file_tag(file_tag, args.seed)

    # Base row information
    row = {
        "dataset": args.dataset,
        "file_tag": args.file,
        "model": args.model_type,
        "perturbation": perturbation,
        "pct": pct,
        "seed": seed,
        "lr": args.lr,
        "hid_dim": args.hid_dim,
        "dropout": args.dropout,
        "train_frac": args.train_frac,
        "run": args.run,

        # timing columns (seconds)
        "time_total_sec": round(float(total_sec), 6),
        "time_train_loop_sec": round(float(train_sec), 6),
        "time_eval_sum_sec": round(float(eval_sec), 6),
    }

    # Add validation metrics
    if best_val_res is not None:
        for k, v in best_val_res.items():
            row[f"val_{k}"] = v

    # Add test metrics
    if best_test_res is not None:
        for k, v in best_test_res.items():
            row[f"test_{k}"] = v

    df = pd.DataFrame([row])

    # One CSV per dataset + model + perturbation
    csv_path = os.path.join(results_dir, f"{args.dataset}_{args.model_type}_{perturbation}.csv")

    df.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False
    )

    print(f"✔ Results saved to {csv_path}")


if __name__ == "__main__":
    # -------------------------
    # TIMING: total start
    # -------------------------
    t_total_start = time.perf_counter()
    t_eval_total = 0.0
    t_train_start = None
    t_train_end = None

    # Preliminary setup.
    args = parse_args()
    set_output_dir(args)
    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n' + str(args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        _t0 = time.perf_counter()
        results = evaluator.evaluate(model, dataset, 'val', train_step=-1)
        t_eval_total += (time.perf_counter() - _t0)

        if not (args.pretrain):
            _t0 = time.perf_counter()
            evaluator.evaluate(model, dataset, 'eval_train', train_step=-1)
            t_eval_total += (time.perf_counter() - _t0)

            _t0 = time.perf_counter()
            evaluator.evaluate(model, dataset, 'test', train_step=-1)
            t_eval_total += (time.perf_counter() - _t0)

    model.train()

    # -------------------------
    # TIMING: training loop start
    # -------------------------
    t_train_start = time.perf_counter()

    for step in train_bar:
        # load batch
        batch = dataset.get_batch()
        batch = {k: v.to(args.device) for k, v in batch.items()}

        # forward pass
        loss = model(**batch)

        # backward pass
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # add to cum loss
        cum_train_loss += loss.item()
        num_steps += 1
        num_batches_trained += 1

        # Log training losses.
        train_bar.set_description(str(np.round(cum_train_loss / num_batches_trained, 5)))
        if (num_steps) % args.print_train_loss_every == 0:
            args.logger.write('\nTrain-loss at step ' + str(num_steps) + ': '
                              + str(cum_train_loss / num_batches_trained))
            cum_train_loss, num_batches_trained = 0, 0

        # run validation
        if (num_steps >= args.validate_after) and (num_steps % args.validate_every == 0):
            # get metrics on test and validation splits
            _t0 = time.perf_counter()
            val_res = evaluator.evaluate(model, dataset, 'val', train_step=step)
            t_eval_total += (time.perf_counter() - _t0)

            if not (args.pretrain):
                _t0 = time.perf_counter()
                evaluator.evaluate(model, dataset, 'eval_train', train_step=step)
                t_eval_total += (time.perf_counter() - _t0)

                _t0 = time.perf_counter()
                test_res = evaluator.evaluate(model, dataset, 'test', train_step=step)
                t_eval_total += (time.perf_counter() - _t0)
            else:
                test_res = None

            model.train(True)

            # Save ckpt if there is an improvement.
            curr_val_metric = val_res['loss_neg'] if args.pretrain else val_res['auprc'] + val_res['auroc']
            if curr_val_metric > best_val_metric:
                best_val_metric = curr_val_metric
                best_val_res, best_test_res = val_res, test_res
                args.logger.write('\nSaving ckpt at ' + model_path_best)
                torch.save(model.state_dict(), model_path_best)
                wait = args.patience
            else:
                wait -= 1
                args.logger.write('Updating wait to ' + str(wait))
                if wait == 0:
                    args.logger.write('Patience reached')
                    break

    # -------------------------
    # TIMING: training loop end
    # -------------------------
    t_train_end = time.perf_counter()

    # print final res
    args.logger.write('Final val res: ' + str(best_val_res))
    args.logger.write('Final test res: ' + str(best_test_res))

    # -------------------------
    # TIMING: total end + summary
    # -------------------------
    t_total_end = time.perf_counter()
    total_sec = t_total_end - t_total_start
    train_sec = (t_train_end - t_train_start) if (t_train_start is not None and t_train_end is not None) else float('nan')
    eval_sec = t_eval_total

    timing_msg = (
        "\n==========================================\n"
        "⏱ TIMING SUMMARY\n"
        f"Total runtime (end-to-end): {total_sec:.2f} sec\n"
        f"Training loop time:         {train_sec:.2f} sec\n"
        f"Evaluation time (sum):      {eval_sec:.2f} sec\n"
        "==========================================\n"
    )
    print(timing_msg)
    args.logger.write(timing_msg)

    # Save final results to CSV (only for supervised training)
    if not args.pretrain:
        save_results_csv(args, best_val_res, best_test_res, total_sec, train_sec, eval_sec)
