from model import Model
import torch
import os
import csv
from train import train, valid
import pickle as pkl
from util import process_data, read_xml
import torch.optim as optim
import numpy as np
from dhg.random import set_seed
import pandas as pd
import torch
from tqdm import tqdm
import argparse
import time
from sklearn.model_selection import KFold
import glob


def parse_arguments():
    parser = argparse.ArgumentParser(description='train Multi-HGNN for missing reaction prediction.')
    parser.add_argument('--num_epochs', default=100, type=int, help='maximum training epochs')
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--in_dim', default=512, type=int, help='dim of input embedding')
    parser.add_argument('--h_dim', default=256, type=int, help='dim of hidden embedding')
    parser.add_argument('--out_dim', default=64, type=int, help='dim of output embedding')
    parser.add_argument('--cuda', default=0, type=int, help='gpu index')
    parser.add_argument('--k_fold', default=5, type=int, help='k-fold cross validation')
    parser.add_argument('--k', default=2, type=int, help='k')
    parser.add_argument('--num_dgnn', default=2, type=int, help='the number of layers in the directed graph neural network')
    parser.add_argument('--num_hgnn', default=1, type=int, help='the number of layers in the hypergraph neural network')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    args = parse_arguments()
    xml_files = glob.glob(os.path.join('BiGG Models', '*.xml'))
    # Loop through each BiGG model
    for xml_file in xml_files:
        model_name = os.path.splitext(os.path.basename(xml_file))[0]
        print(f'Current model ID: {model_name}')
        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        read_xml(xml_content)
        g, hg_pos, hg_neg, initial_features, reaction_count = process_data()
        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=0)
        m_emb = initial_features
        results = []
        Y_pos_r = torch.Tensor()
        Y_neg_r = torch.Tensor()
        for fold, (train_set, valid_set) in enumerate(kf.split(np.arange(reaction_count))):
            net = Model(args.in_dim, args.h_dim, args.out_dim, args.num_dgnn, args.num_hgnnp, use_bn=True)
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
            m_emb = m_emb.to(device)
            g = g.to(device)
            net = net.to(device)
            train_set = torch.tensor(train_set).long()
            valid_set = torch.tensor(valid_set).long()
            epoch_loss = []
            start_time = time.time()
            for epoch in tqdm(range(args.num_epochs), desc=f"Training Fold {fold+1}"):
                epoch_loss.append(train(net, m_emb, g, hg_pos, hg_neg, train_set, optimizer))
            elapsed_time = time.time() - start_time
            with torch.no_grad():
                accuracy, precision, recall, f1, mcc, auroc, auprc, X_emb, Y_pos, Y_neg, valid_pos_scores, valid_neg_scores = valid(net, m_emb, g, hg_pos, hg_neg, valid_set)
            results.append((accuracy, precision, recall, f1, mcc, auroc, auprc, elapsed_time))
        results_arr = np.array(results)
        # Calculate the mean value
        mean_acc = np.mean(results_arr[:, 0])
        mean_precision = np.mean(results_arr[:, 1])
        mean_recall = np.mean(results_arr[:, 2])
        mean_f1 = np.mean(results_arr[:, 3])
        mean_mcc = np.mean(results_arr[:, 4])
        mean_auroc = np.mean(results_arr[:, 5])
        mean_aupr = np.mean(results_arr[:, 6])
        mean_time = np.mean(results_arr[:, 7])
        # Calculate standard deviation
        std_acc = np.std(results_arr[:, 0])
        std_precision = np.std(results_arr[:, 1])
        std_recall = np.std(results_arr[:, 2])
        std_f1 = np.std(results_arr[:, 3])
        std_mcc = np.std(results_arr[:, 4])
        std_auroc = np.std(results_arr[:, 5])
        std_aupr = np.std(results_arr[:, 6])
        std_time = np.std(results_arr[:, 7])

        result_final = [(model_name, mean_acc, std_acc, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1,
                         mean_mcc, std_mcc, mean_auroc, std_auroc, mean_aupr, std_aupr, mean_time, std_time)]
        print(
            f"{args.k_fold}-fold average performance - ACC: {mean_acc:.3f} ± {std_acc:.3f}, Precision: {mean_precision:.3f} ± {std_precision:.3f}, Recall: {mean_recall:.3f} ± {std_recall:.3f}, F1: {mean_f1:.3f} ± {std_f1:.3f}, MCC: {mean_mcc:.3f} ± {std_mcc:.3f}, AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}, AUPR: {mean_aupr:.3f} ± {std_aupr:.3f}")