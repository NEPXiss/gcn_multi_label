# data_pipeline/dataset_with_class_column.py
"""
Extended dataset.py that preserves 'class' column for semi-supervised learning

Usage:
    python -m data_pipeline.dataset_with_class_column \
        --map_csv disease_map.csv \
        --out processed_gcn_dataset.npz \
        --csv_out combined.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from .utils import save_npz, get_logger

logger = get_logger("dataset")


def build_combined_with_class(
    file_map: dict,
    subj_col: str = 'subject_id',
    disease_col: str = 'disease',
    study_col: str = 'study',
    class_col: str = 'class'
):
    """
    Load per-disease CSVs, preserve 'class' column for semi-supervised learning
    
    Returns:
        combined (pd.DataFrame), X_raw (np.ndarray), Y (np.ndarray), 
        class_info (np.ndarray), study_info (np.ndarray),
        feature_cols (list), label_cols (list)
    """
    dfs = []
    all_features = set()

    # 1) Load each CSV
    for i, (name, path) in enumerate(file_map.items(), 1):
        print(f"[{i}/{len(file_map)}] Loading {name} from {path}...")
        df = pd.read_csv(path)
        
        # Create label column
        label_col = f'label_{name}'
        if disease_col in df.columns:
            df[label_col] = df[disease_col].astype(str).str.contains(
                name, case=False, na=False
            ).astype(int)
        else:
            df[label_col] = df.get(label_col, pd.Series(0, index=df.index)).astype(int)
        
        dfs.append(df)

        # Identify feature columns
        feat_cols = [c for c in df.columns
                     if c not in [subj_col, class_col, disease_col, study_col]
                     and not c.startswith('label_')]
        all_features.update(feat_cols)
        print(f"    {name}: {df.shape[0]} rows, {len(feat_cols)} features")

    all_features = sorted(list(all_features))
    label_cols = [f'label_{d}' for d in file_map.keys()]

    # 2) Align features and labels
    for i, df in enumerate(dfs):
        print(f"[{i+1}/{len(dfs)}] Aligning features & labels...")
        
        # Add missing features
        missing_feats = [f for f in all_features if f not in df.columns]
        if missing_feats:
            df = pd.concat([
                df, 
                pd.DataFrame(0.0, index=df.index, columns=missing_feats)
            ], axis=1)
        
        # Add missing labels
        for lbl in label_cols:
            if lbl not in df.columns:
                df[lbl] = 0
        
        # Reorder columns
        meta_cols = [c for c in [subj_col, class_col, disease_col, study_col] 
                     if c in df.columns]
        ordered = meta_cols + all_features + label_cols
        df = df[ordered]
        dfs[i] = df

    # 3) Concatenate
    print("Concatenating dataframes...")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {combined.shape}")

    # 4) Aggregate duplicates (within same class - don't mix train/test!)
    if subj_col in combined.columns and class_col in combined.columns:
        print("Aggregating duplicate subjects (preserving train/test split)...")
        
        def agg_func(series):
            if series.name in label_cols:
                return series.max()
            elif series.name in all_features:
                return series.mean()
            elif series.name == class_col:
                # Keep first class label (should be same for duplicates)
                nonnull = series.dropna()
                return nonnull.iloc[0] if len(nonnull) > 0 else np.nan
            elif series.name == study_col:
                # Keep first study (or most common)
                nonnull = series.dropna()
                return nonnull.iloc[0] if len(nonnull) > 0 else np.nan
            else:
                nonnull = series.dropna()
                return nonnull.iloc[0] if len(nonnull) > 0 else np.nan
        
        # Group by subject_id AND class to keep train/test separate
        combined = combined.groupby([subj_col, class_col], sort=False).agg(
            agg_func
        ).reset_index()
        print(f"After aggregation: {combined.shape}")

    # 5) Ensure labels are integers
    combined[label_cols] = combined[label_cols].fillna(0).astype(int)

    # 6) Extract matrices
    X_raw = combined[all_features].values.astype(float)
    Y = combined[label_cols].values.astype(int)
    
    # Extract class and study information
    if class_col in combined.columns:
        class_info = combined[class_col].fillna('unknown').values
    else:
        class_info = np.array(['train'] * len(combined))
    
    if study_col in combined.columns:
        study_info = combined[study_col].fillna('unknown').values
    else:
        study_info = np.array(['unknown'] * len(combined))

    return combined, X_raw, Y, class_info, study_info, all_features, label_cols


def preprocess_log_zscore(X: np.ndarray, eps: float = 1e-6):
    """
    Log10 transform and z-score each feature (column-wise).
    Returns: X_z, mean, std_safe
    """
    print("Preprocessing: log10 + z-score...")
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
    parser = argparse.ArgumentParser(
        description="Build combined dataset with class column for semi-supervised learning"
    )
    parser.add_argument('--map_csv', required=True,
                        help='CSV mapping: columns [disease,path]')
    parser.add_argument('--out', default='processed_gcn_dataset.npz',
                        help='Output .npz path for processed dataset')
    parser.add_argument('--csv_out', default='combined_dataset_with_labels.csv',
                        help='CSV export of combined dataset for inspection')
    parser.add_argument('--subj_col', default='subject_id', 
                        help='Subject ID column name')
    parser.add_argument('--disease_col', default='disease', 
                        help='Disease column name')
    parser.add_argument('--study_col', default='study', 
                        help='Study column name')
    parser.add_argument('--class_col', default='class', 
                        help='Class column name (train/test)')
    args = parser.parse_args()

    file_map = load_map_csv(Path(args.map_csv))
    logger.info("Loaded map for %d diseases", len(file_map))

    combined, X_raw, Y, class_info, study_info, features, label_cols = \
        build_combined_with_class(
            file_map=file_map,
            subj_col=args.subj_col,
            disease_col=args.disease_col,
            study_col=args.study_col,
            class_col=args.class_col
        )

    X_proc, means, stds = preprocess_log_zscore(X_raw)

    # Save with additional metadata
    save_npz(
        args.out,
        X_raw=X_raw,
        X_proc=X_proc,
        Y=Y,
        class_info=class_info,          # NEW: train/test labels
        study_info=study_info,          # NEW: domain/study labels
        feature_cols=np.array(features, dtype=object),
        label_cols=np.array(label_cols, dtype=object),
        means=means,
        stds=stds
    )

    combined.to_csv(args.csv_out, index=False)

    logger.info("Done! Shapes:")
    logger.info("X_proc: %s", X_proc.shape)
    logger.info("Y: %s", Y.shape)
    logger.info("class_info: %s (unique: %s)", 
                class_info.shape, np.unique(class_info))
    logger.info("study_info: %s (unique: %s)", 
                study_info.shape, np.unique(study_info))
    logger.info("CSV saved as %s", args.csv_out)
    logger.info("NPZ saved as %s", args.out)