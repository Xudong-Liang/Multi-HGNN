# Multi-HGNN: Multi-modal hypergraph neural networks for predicting missing reactions in metabolic networks
## Introduction
![image](https://github.com/Xudong-Liang/Multi-HGNN/blob/main/overview.png)

Multi-HGNN is a multi-modal hypergraph neural network that integrates multi-modal metabolic data to predict missing reactions. Multi-HGNN consists of three feature learning modules: biochemical feature learning module, which integrates metabolite biochemical features learned by a model that is pre-trained on large unlabeled small molecule datasets; the metabolic directed graph and hypergraph learning modules, which are used to learn the directionality and high-order interactions in metabolic reactions, respectively.

## Environment
We conduct our experiments with python3.8. Here are the requirements

- torch: 1.13.1
- dgl: 1.1.1+cu116
- dhg: 0.9.3
- numpy: 1.24.4
- scikit-learn: 1.3.0
- pandas: 1.5.3
- matplotlib: 3.7.2
- rdkit: 2023.9.6
- tqdm: 4.65.0


## Usage

- Download pre-trained node embedding files for all metabolites **`metabolite_emb.pkl`** ([download link](https://drive.google.com/file/d/1l3tNxv8pLRtyi-iZDGGnGAG7El6e-sTj/view?usp=sharing)) and place it in the `data` folder.
- python main.py

## Acknowledgement
DGL: https://www.dgl.ai/

DHG: https://github.com/iMoonLab/DeepHypergraph

KPGT: https://github.com/lihan97/KPGT
