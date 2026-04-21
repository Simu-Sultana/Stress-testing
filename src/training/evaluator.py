from tqdm import tqdm
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, f1_score, fbeta_score
)
import numpy as np
from sklearn.utils.multiclass import type_of_target


class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = ' + split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(
            range(0, num_samples, self.args.eval_batch_size),
            desc='running forward pass'
        )

        true, pred = [], []

        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples, start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)

            true.append(batch['labels'])
            del batch['labels']

            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            with torch.no_grad():
                pred.append(model(**batch).cpu())

        # Concatenate all batches
        true = torch.cat(true)
        pred = torch.cat(pred)

        # ----------------------------
        # STRaTS original metrics (KEEP SAME RESULT FORMAT)
        # ----------------------------
        y_type = type_of_target(true)

        if y_type == "binary":
            precision_curve, recall_curve, thresholds = precision_recall_curve(true, pred)
            pr_auc = auc(recall_curve, precision_curve)
            minrp = np.minimum(precision_curve, recall_curve).max()
            roc_auc = roc_auc_score(true, pred)
        else:
            print(f"Skipping STRaTS AUROC/AUPRC/minrp: target type is '{y_type}' (not binary).")
            roc_auc = np.nan
            pr_auc = np.nan
            minrp = np.nan

        # ----------------------------
        # Additional metrics @ threshold = 0.5 (binary only)
        # ----------------------------
        if y_type == "binary":
            threshold = 0.5
            y_true = true.cpu().numpy()
            y_prob = pred.cpu().numpy()
            y_pred = (y_prob >= threshold).astype(int)

            precision_val = precision_score(y_true, y_pred)
            recall_val = recall_score(y_true, y_pred)
            accuracy_val = accuracy_score(y_true, y_pred)
            balanced_acc_val = balanced_accuracy_score(y_true, y_pred)
            f1_val = f1_score(y_true, y_pred)
            f2_val = fbeta_score(y_true, y_pred, beta=2)
        else:
            precision_val = np.nan
            recall_val = np.nan
            accuracy_val = np.nan
            balanced_acc_val = np.nan
            f1_val = np.nan
            f2_val = np.nan

        # ----------------------------
        # Final result dictionary (same keys always)
        # ----------------------------
        result = {
            'auroc': roc_auc,
            'auprc': pr_auc,
            'minrp': minrp,
            'precision@0.5': precision_val,
            'recall@0.5': recall_val,
            'accuracy@0.5': accuracy_val,
            'balanced_accuracy@0.5': balanced_acc_val,
            'f1@0.5': f1_val,
            'f2@0.5': f2_val
        }

        # Optional: prettier logging (does not change result values)
        if train_step is not None:
            def fmt(v):
                if isinstance(v, (float, np.floating)) and not np.isnan(v):
                    return f"{v:.4f}"
                return v

            pretty = {k: fmt(v) for k, v in result.items()}

            self.args.logger.write(
                'Result on ' + split + ' split at train step '
                + str(train_step) + ': ' + str(pretty)
            )

        return result
