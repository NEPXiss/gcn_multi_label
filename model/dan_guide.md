## Example
```
python -m model.train_with_domain_adaptation --data processed_data/gcn_data.pt --da-method mmd --da-weight 0.1 --latent-dim 128 --epochs 100 --save-model processed_data/gcn_model_da.pt
```
```
python -m graph.graph_builder_with_da --input processed_data/processed_gcn_dataset.npz --encoder-model processed_data/gcn_model_da.pt --output processed_data/gcn_data_da_refined.pt --k 8
```
```
python -m graph.graph_builder_with_da --input processed_data/processed_gcn_dataset.npz --output processed_data/gcn_data_da.pt --latent-dim 128 --pretrain-epochs 50 --k 8
```
```
python -m model.train_with_domain_adaptation --data processed_data/gcn_data_da.pt --da-method mmd --da-weight 0.1 --latent-dim 128 --hidden-dim 64 --epochs 100 --lr 1e-3 --dropout 0.3 --val-split 0.2 --early-stop 15 --use-posweight --save-model processed_data/final_gcn_da.pt
```

# 1. Use original dataset pipeline
```
python -m data_pipeline.dataset --map_csv data_map.csv --out processed_data/processed_gcn_dataset.npz --csv_out processed_data/combined_dataset_with_labels.csv
```

# 2. Build simple graph (No DA encoder)
```
python -m graph.graph_builder --input processed_data/processed_gcn_dataset.npz --output processed_data/gcn_data.pt
```

# 3. Train with DA
```
python -m model.train_with_domain_adaptation --data processed_data/gcn_data.pt --da-method mmd --da-weight 0.1 --epochs 100 --save-model processed_data/model_da.pt
```

# 4. If the result is desirable, then rebuild the graph with learned encoder
```
python -m graph.graph_builder_with_da --input processed_data/processed_gcn_dataset.npz --encoder-model processed_data/model_da.pt --output processed_data/gcn_data_refined.pt
```

# 5. Train again with refined graph
```
python -m model.train_with_domain_adaptation --data processed_data/gcn_data_refined.pt --da-method mmd --da-weight 0.1 --epochs 100
```