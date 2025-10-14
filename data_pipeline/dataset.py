"""
data_pipeline/dataset.py
Refactor of generate_dataset.py

Usage:
    python -m data_pipeline.dataset --map_csv path/to/file_map.csv --out processed_gcn_dataset.npz --csv_out combined.csv

The map_csv must be a CSV with two columns: "disease" and "path".
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from .utils import save_npz, get_logger

logger = get_logger("dataset")


def build_combined(file_map: dict,
                   subj_col: str = 'subject_id',
                   disease_col: str = 'disease',
                   study_col: str = 'study',
                   class_col: str = 'class'):
    """
    Load per-disease CSVs, align feature columns, create multi-hot labels, and aggregate duplicate subjects.
    Returns:
        combined (pd.DataFrame), X_raw (np.ndarray), Y (np.ndarray), feature_cols (list), label_cols (list)
    """
    dfs = []
    all_features = set()

    # 1) Load each CSV and create initial label column
    for name, path in file_map.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found for disease {name}")
        df = pd.read_csv(p)
        label_col = f'label_{name}'
        # If disease column missing, create safe string for contains
        if disease_col in df.columns:
            df[label_col] = df[disease_col].astype(str).str.contains(name, case=False, na=False).astype(int)
        else:
            # fallback: if dataset already contains label column or disease not provided
            if label_col in df.columns:
                df[label_col] = df[label_col].astype(int)
            else:
                # assume none
                df[label_col] = 0
        dfs.append(df)

        # collect feature columns (exclude metadata and label_*)
        feat_cols = [c for c in df.columns
                     if c not in [subj_col, class_col, disease_col, study_col]
                     and not c.startswith('label_')]
        all_features.update(feat_cols)

    all_features = sorted(list(all_features))
    label_cols = [f'label_{d}' for d in file_map.keys()]

    # 2) Align features and labels in each df
    for i, df in enumerate(dfs):
        # add missing feature columns as zeros
        missing_feats = [f for f in all_features if f not in df.columns]
        if missing_feats:
            df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing_feats)], axis=1)

        # add missing label columns (default 0)
        for lbl in label_cols:
            if lbl not in df.columns:
                df[lbl] = 0

        # reorder: metadata -> features -> labels
        meta_cols = [c for c in [subj_col, class_col, disease_col, study_col] if c in df.columns]
        ordered = meta_cols + all_features + label_cols
        df = df[ordered]
        dfs[i] = df

    # 3) Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)

    # 4) Handle duplicate subject_id: max for labels, mean for features
    def agg_func(series):
        if series.name in label_cols:
            return series.max()
        elif series.name in all_features:
            return series.mean()
        else:
            nonnull = series.dropna()
            return nonnull.iloc[0] if len(nonnull) > 0 else np.nan

    if subj_col in combined.columns:
        combined = combined.groupby(subj_col, sort=False).agg(agg_func).reset_index()

    # 5) Ensure labels are integers
    combined[label_cols] = combined[label_cols].fillna(0).astype(int)

    # 6) Extract raw feature matrix and label matrix
    X_raw = combined[all_features].values.astype(float)
    Y = combined[label_cols].values.astype(int)

    return combined, X_raw, Y, all_features, label_cols


def preprocess_log_zscore(X: np.ndarray, eps: float = 1e-6):
    """
    Log10 transform and z-score each feature (column-wise).
    Returns: X_z, mean, std_safe
    """
    X_log = np.log10(X + eps)
    mean = X_log.mean(axis=0)
    std = X_log.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_z = (X_log - mean) / std_safe
    return X_z, mean, std_safe


def load_map_csv(path: Path):
    """
    Expect CSV with columns: disease,path
    Returns dict {disease: path}
    """
    df = pd.read_csv(path)
    if 'disease' not in df.columns or 'path' not in df.columns:
        raise ValueError("map_csv must contain columns 'disease' and 'path'")
    return dict(zip(df['disease'], df['path']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build combined dataset and preprocess for GCN")
    parser.add_argument('--map_csv', required=True,
                        help='CSV mapping: columns [disease,path]')
    parser.add_argument('--out', default='processed_gcn_dataset.npz',
                        help='Output .npz path for processed dataset')
    parser.add_argument('--csv_out', default='combined_dataset_with_labels.csv',
                        help='CSV export of combined dataset for inspection')
    parser.add_argument('--subj_col', default='subject_id', help='Subject ID column name')
    parser.add_argument('--disease_col', default='disease', help='Disease column name (used to create labels)')
    parser.add_argument('--study_col', default='study', help='Study column name (optional)')
    parser.add_argument('--class_col', default='class', help='Class column name (optional)')
    args = parser.parse_args()

    file_map = load_map_csv(Path(args.map_csv))
    logger.info("Loaded map for %d diseases", len(file_map))

    combined, X_raw, Y, features, label_cols = build_combined(
        file_map=file_map,
        subj_col=args.subj_col,
        disease_col=args.disease_col,
        study_col=args.study_col,
        class_col=args.class_col
    )

    X_proc, means, stds = preprocess_log_zscore(X_raw)

    save_npz(args.out,
             X_raw=X_raw,
             X_proc=X_proc,
             Y=Y,
             feature_cols=np.array(features, dtype=object),
             label_cols=np.array(label_cols, dtype=object),
             means=means,
             stds=stds)

    combined.to_csv(args.csv_out, index=False)

    logger.info("Done! Shapes:")
    logger.info("X_proc: %s", X_proc.shape)
    logger.info("Y: %s", Y.shape)
    logger.info("CSV saved as %s", args.csv_out)
    logger.info("NPZ saved as %s", args.out)
