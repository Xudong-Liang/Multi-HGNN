import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    average_precision_score
import torch.nn as nn


def compute_loss(pos_scores, neg_scores):
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    labels = torch.cat((pos_labels, neg_labels), dim=0)
    scores = torch.cat((pos_scores, neg_scores), dim=0)

    loss = F.binary_cross_entropy(scores, labels)
    return loss


def train(net, m_emb, g, hg_pos, hg_neg, train_set, optimizer):
    net.train()
    optimizer.zero_grad()
    pos_scores, neg_scores, _, _, _ = net(m_emb, g, hg_pos, hg_neg)
    train_pos_scores = pos_scores[train_set]
    train_neg_scores = neg_scores[train_set]
    loss = compute_loss(train_pos_scores, train_neg_scores)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def valid(net, m_emb, g, hg_pos, hg_neg, valid_set):
    net.eval()
    pos_scores, neg_scores, m_emb, Y_pos, Y_neg = net(m_emb, g, hg_pos, hg_neg)
    valid_pos_scores = pos_scores[valid_set]
    valid_neg_scores = neg_scores[valid_set]
    scores = torch.cat((valid_pos_scores, valid_neg_scores), dim=0)
    labels = torch.cat((torch.ones_like(valid_pos_scores), torch.zeros_like(valid_neg_scores)), dim=0)
    predictions = (scores > 0.5).float()

    if torch.cuda.is_available():
        labels = labels.cpu()
        predictions = predictions.cpu()
        scores = scores.cpu()

    accuracy = accuracy_score(labels.numpy(), predictions.numpy())
    precision = precision_score(labels.numpy(), predictions.numpy())
    recall = recall_score(labels.numpy(), predictions.numpy())
    f1 = f1_score(labels.numpy(), predictions.numpy())
    mcc = matthews_corrcoef(labels.numpy(), predictions.numpy())
    auroc = roc_auc_score(labels.numpy(), scores.numpy())
    auprc = average_precision_score(labels.numpy(), scores.numpy())

    print(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}, AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
    return accuracy, precision, recall, f1, mcc, auroc, auprc, m_emb, Y_pos, Y_neg, valid_pos_scores, valid_neg_scores