import os
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
def print_split_stats(cohort, splits):
    for split_name, ids in splits.items():
        y = cohort.loc[cohort['ts_id'].astype(str).isin(ids), 'in_hospital_mortality']

        print(f"{split_name}: n={len(y)}")

        # binary stats
        if y.nunique() <= 2:
            pos_rate = y.mean()
            print(f"  positives: {y.sum()}, rate: {pos_rate:.3f}")
        # multiclass stats
        else:
            counts = y.value_counts()
            for cls, cnt in counts.items():
                print(f"  class {cls}: {cnt} ({cnt/len(y):.3f})")


def compute_splits_stf(dataset, data_paths, OUT_PATH=None, n_splits=3, train_frac=0.8, seed=123):

    # File paths
    filepath = data_paths[dataset]
    if OUT_PATH is None:
        OUT_PATH = os.path.join(os.path.dirname(filepath), "folds")
    os.makedirs(OUT_PATH, exist_ok=True)

    # Load dataset
    with open(filepath, "rb") as f:
        ts, oc, train_ids, val_ids, test_ids = pickle.load(f)

    X = oc['ts_id']
    y = oc['in_hospital_mortality']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        test_ids = X.iloc[test_idx].astype(str).values
        train_valid_ids = X.iloc[train_idx].astype(str).values
        y_train_valid = y.iloc[train_idx]

        # Stratified 80/20 split for train/val
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
        train_idx_inner, val_idx_inner = next(sss.split(train_valid_ids, y_train_valid))
        train_ids_fold = train_valid_ids[train_idx_inner]
        val_ids_fold   = train_valid_ids[val_idx_inner]

        # -----------------------------
        # Check for overlap
        # -----------------------------
        overlap_train_val = set(train_ids_fold) & set(val_ids_fold)
        overlap_train_test = set(train_ids_fold) & set(test_ids)
        overlap_val_test = set(val_ids_fold) & set(test_ids)

        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                f"Overlap detected in fold {i}!\n"
                f"train/val: {overlap_train_val}\n"
                f"train/test: {overlap_train_test}\n"
                f"val/test: {overlap_val_test}"
            )

        # Print stats
        print(f"############ FOLD {i} ############")
        print_split_stats(oc, {"Train": train_ids_fold, "Val": val_ids_fold, "Test": test_ids})

        # Save fold
        out_file = os.path.join(OUT_PATH, f'{dataset}_fold_{i}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump([ts, oc, train_ids_fold, val_ids_fold, test_ids], f)
        print(f"Saved fold {i} to {out_file}\n")
if __name__ == "__main__":
    data_paths = {
        "physionet_2012": "/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/data/processed/physionet_2012.pkl",
        "mimic_iii": "/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/data/processed/mimic_iii.pkl"
    }

    compute_splits_stf(
        "physionet_2012",
        data_paths,
        OUT_PATH="/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/data/cv_splits/physionet_2012_in_hospital_mortality_unbalanced"
    )

    compute_splits_stf(
        "mimic_iii",
        data_paths,
        OUT_PATH="/home/hpc/iwbn/iwbn127h/projects/STRaTS_christel/strats-development/data/cv_splits/mimic_iii_in_hospital_mortality_unbalanced"
    )
