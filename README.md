# Multi-Disease GCN for Microbiome Classification (Inspired by GDmicro)

**⚠️ Disclaimer: This project is an academic work inspired by the original GDmicro research (GDmicro: Classifying host disease status with GCN and deep adaptation network based on the human gut microbiome data; Liao, H., Shang, J., & Sun, Y. (2023)). The purpose of this repository is educational and experimental, exploring multi-label GCN classification for microbiome data. The original research is licensed under the MIT License, and this work is done in good faith to follow its ideas.**

This project is an academic exploration conducted as part of my Bioinformatics course term project (2110581 - Chulalongkorn University, TH).

## Project Overview

This project is a from-scratch implementation inspired by the GDmicro framework. Its goal is to extend microbiome classification to handle multi-disease labels. Key features:

- Multi-label classification using a Graph Convolutional Network (GCN).

- Preprocessing of microbiome data (log10 transformation + z-score normalization).

- Construction of k-nearest neighbor graphs from sample features.

- Multi-label sigmoid output with binary cross-entropy loss for multi-disease prediction.

The implementation is original and follows the conceptual framework of GDmicro for academic exploration.

## Project Status

This repository represents work in progress. The project is inspired by GDmicro and is being developed as part of my Bioinformatics course. Further implementation and guidelines will be added.

## Environment Setup (Require Conda)
```conda env create -f clean_environment.yml --name gcn_multi```