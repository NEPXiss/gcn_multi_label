## How to run:

Example

```
python model/gcn_model.py --epochs 50 --hidden_dim 64 --use-posweight --tune-thresholds --val-split 0.2
```

## Important Flags

```
--data : path of PyG Data object (default: processed_data/gcn_data.pt).

--epochs : Number of epoch in training (default: 100).

--lr : learning rate (default: 1e-3).

--hidden_dim : hidden_dim of GCN (default: 64).

--weight_decay : L2 weight decay for optimizer.

--dropout : dropout rate in model.

--batchnorm : if flaged, BatchNorm is enabled (default off).

--layernorm : if flaged, LayerNorm is enabled (default off).

--device : 'cpu' or 'cuda' (default: auto detect).

--save-model : path for storing state_dict of the model (default: processed_data/gcn_model.pt).

--output : path of prediction CSV (default: processed_data/prediction_output.csv).

--val-split : validation ratio (node-level random split); for example, 0.2 = 20% validation. if 0, then no validation (will not tune thresholds).

--seed : seed (reproducibility).

--early-stop : Number of epoch for early stop if no improvement -> validation micro-F1 (default: 15).

--tune-thresholds : if enabled, turn on thresholds for tune (training wrapper already tunes on val and returns best thresholds).

--use-posweight : Calculate pos_weight from labels and input to BCEWithLogitsLoss to compensate class imbalance.
```

## Expected Outcome:

```
processed_data/gcn_model.pt (default) — state_dict file of the model (you can load using model.load_state_dict(torch.load(path)))
```

- processed_data/prediction_output.csv — csv for sample_id, true_label_{j}, prob_label_{j}, pred_label_{j} for all labels (j = 0..D-1)

- Short message describing the metric (micro_F1, macro_F1) is printed on terminal