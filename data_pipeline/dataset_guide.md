Suppose your project structure resembles:

```
project_root/
│
├─ raw_data/                  # <<< input csv
│   ├─ CRC.csv
│   ├─ T2D.csv
│   ├─ OBT.csv
│   ├─ IBD.csv
│   └─ Cirrhosis.csv
│
├─ processed_data/            # <<< output saved here
│
└─ data_pipeline/
    ├─ dataset.py
    ├─ utils.py
```

Create data_map.csv in root directory:

## data_map.csv ## (Add the following in data_map.csv)

```
disease,path
CRC,Input_files/CRC.csv
T2D,Input_files/T2D.csv
OBT,Input_files/Obt.csv
IBD,Input_files/IBD.csv
Cirrhosis,Input_files/cirrhosis.csv
```
###

## Run:

```
python -m data_pipeline.dataset --map_csv data_map.csv --out processed_data/processed_gcn_dataset.npz --csv_out processed_data/combined_dataset_with_labels.csv
```

## You'll get the following output in this structure :

```
processed_data/
├─ processed_gcn_dataset.npz
└─ combined_dataset_with_labels.csv
```