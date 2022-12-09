# DeepQSE

## Introduction
Python implementation for our paper "Effective and Efficient Query-aware Snippet Extraction for Web Search" in EMNLP 2022.

## Class Selection

| Method | Model | Dataset | Collate | 
| -- | -- | -- | -- | 
| DeepQSE | TopTFModelLoad | TitleQueryDataset | TitleQueryCollate |
| Efficient-DeepQSE (coarse) | TopTFModelLoad | TitleQuerySentDataset | TitleQuerySentCollate |
| Efficient-DeepQSE (fine) | DiffQueryKVModel | QueryKVTitleDataset | QueryKVTitleCollate |
