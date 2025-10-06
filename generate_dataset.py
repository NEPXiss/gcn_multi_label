import os
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
file_map = {
    'CRC': 'path',
    'T2D': 'path',
    'OBT': 'path',
    'IBD': 'path',
    'Cirrhosis': 'path'
}

SUBJ_COL = 'subject_id'
CLASS_COL = 'class'
DISEASE_COL = 'disease'
STUDY_COL = 'study'

diseases = list(file_map.keys())

# 1) Load each CSV and create initial label column
dfs = []
all_features = set()

for disease, path in file_map.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    # create binary label for this disease
    df[f'label_{disease}'] = df[DISEASE_COL].str.contains(disease, case=False, na=False).astype(int)
    dfs.append(df)
    # collect feature columns
    feature_cols = [c for c in df.columns if c not in [SUBJ_COL, CLASS_COL, DISEASE_COL, STUDY_COL] and not c.startswith('label_')]
    all_features.update(feature_cols)

all_features = sorted(list(all_features))
label_cols = [f'label_{d}' for d in diseases]

# 2) Align features and labels in each df
for i, df in enumerate(dfs):
    # add missing feature columns
    missing_feats = [f for f in all_features if f not in df.columns]
    if missing_feats:
        df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing_feats)], axis=1)
    # add missing label columns
    for lbl in label_cols:
        if lbl not in df.columns:
            df[lbl] = 0
    # reorder columns: metadata -> features -> labels
    meta_cols = [c for c in [SUBJ_COL, CLASS_COL, DISEASE_COL, STUDY_COL] if c in df.columns]
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

if SUBJ_COL in combined.columns:
    combined = combined.groupby(SUBJ_COL, sort=False).agg(agg_func).reset_index()

# 5) Ensure labels are integers
combined[label_cols] = combined[label_cols].fillna(0).astype(int)

# 6) Extract feature matrix and label matrix
X_raw = combined[all_features].values.astype(float)
Y = combined[label_cols].values.astype(int)

# 7) Preprocess: log10 + z-score
def preprocess_log_zscore(X, eps=1e-6):
    X_log = np.log10(X + eps)
    mean = X_log.mean(axis=0)
    std = X_log.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_z = (X_log - mean) / std_safe
    return X_z, mean, std_safe

X_proc, means, stds = preprocess_log_zscore(X_raw)

# 8) Save npz for GCN and CSV for inspection
np.savez_compressed('processed_gcn_dataset.npz',
                    X_raw=X_raw,
                    X_proc=X_proc,
                    Y=Y,
                    feature_cols=np.array(all_features, dtype=object),
                    label_cols=np.array(label_cols, dtype=object))

combined.to_csv('combined_dataset_with_labels.csv', index=False)

print("Done! Shapes:")
print("X_proc:", X_proc.shape)
print("Y:", Y.shape)
print("CSV saved as combined_dataset_with_labels.csv")
print("NPZ saved as processed_gcn_dataset.npz")
