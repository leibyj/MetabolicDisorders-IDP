import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

from models import MIL_classifier
from datasets.patch_datasets import FeatureBagDataset, collate_bag_batches

from sys import argv
import copy
import statistics

"""
Trains and evaluates classifiers
"""

def train_fold(split, fold, device, input_dim, data_path, label_path, out_path):
    print("_"*15, "FOLD: ", fold, "_"*15)

    tr_md = pd.read_csv(f"{label_path}/{split}/fold_{fold}_cv_files_TRAINING.csv", names=['ID', 'label'])
    tr_files = tr_md.ID.tolist()
    tr_labels = tr_md.label.tolist()

    tr_bag_data = FeatureBagDataset(data_path = data_path,
                         files = tr_files,
                         labels = tr_labels)

    tr_dl = DataLoader(tr_bag_data, batch_size=10, shuffle=True, collate_fn=collate_bag_batches)

    val_md = pd.read_csv(f"{label_path}/{split}/fold_{fold}_cv_files_VALIDATION.csv", names=['ID', 'label'])
    val_files =val_md.ID.tolist()
    val_labels = val_md.label.tolist()

    val_bag_data = FeatureBagDataset(data_path = data_path,
                         files = val_files,
                         labels = val_labels)

    val_dl = DataLoader(val_bag_data, batch_size=10, shuffle=False, collate_fn=collate_bag_batches)

    te_md = pd.read_csv(f"{label_path}/{split}/fold_{fold}_cv_files_TESTING.csv", names=['ID', 'label'])
    te_files = te_md.ID.tolist()
    te_labels = te_md.label.tolist()

    te_bag_data = FeatureBagDataset(data_path = data_path,
                         files = te_files,
                         labels = te_labels)

    te_dl = DataLoader(te_bag_data, batch_size=10, shuffle=False, collate_fn=collate_bag_batches)

    torch.manual_seed(1024)
    mod = MIL_classifier.MIL_model(dims=[input_dim, 512, 256], return_features=True).to(device)
    opt = torch.optim.AdamW(mod.parameters())
    criterion = nn.BCELoss().to(device)

    train_loss = []

    # validation metrics for early stopping
    best_val_auc = 0
    opt_epoch = 0
    opt_model = None 

    for j in range(100):
        mod.train()
        for i, (dat, lab, fn) in enumerate(tr_dl):
            opt.zero_grad()
            b_out = torch.empty(0).to(device)
            for d in dat:
                d = d.to(device)
                out, A, _ = mod(d)
                b_out = torch.cat((b_out, out))
            b_out = b_out.unsqueeze(1)
            lab = lab.to(device)
            loss = criterion(b_out, lab)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            # print(loss.item())
        # print(A)
        # print("Loss: ", loss.item())

        # Validation performance metrics
        all_labels = torch.empty(0)
        all_out = torch.empty(0)
        with torch.no_grad():
            mod.eval()
            for i, (dat, lab, fn) in enumerate(val_dl):
                b_out = torch.empty(0)
                for d in dat:
                    d = d.to(device)
                    out, _, _ = mod(d)
                    b_out = torch.cat((b_out, out.detach().cpu()))
                b_out = b_out.unsqueeze(1)
                lab = lab

                all_labels = torch.cat((all_labels, lab), 0)
                all_out = torch.cat((all_out, b_out), 0)
            val_loss = criterion(all_out, all_labels)
        print("EPOCH: ", j)
        auc = roc_auc_score(all_labels.numpy(), all_out.numpy())
        print(f"Validation AUC: {auc:.5f}")
        aupr = average_precision_score(all_labels.numpy(), all_out.numpy())
        print(f"Valdiation AUPRC: {aupr:.5f} \n")

        if auc > best_val_auc:
            best_val_auc = auc
            opt_epoch = j
            opt_model = copy.deepcopy(mod)
            print("New optimal epoch found. \n")
        

    # Test performance metrics with opt model
    all_labels = torch.empty(0)
    all_out = torch.empty(0)
    with torch.no_grad():
        opt_model.eval()
        for i, (dat, lab, fn) in enumerate(te_dl):
            b_out = torch.empty(0)
            for d in dat:
                d = d.to(device)
                out, _, _ = opt_model(d)
                b_out = torch.cat((b_out, out.detach().cpu()))
            b_out = b_out.unsqueeze(1)
            lab = lab

            all_labels = torch.cat((all_labels, lab), 0)
            all_out = torch.cat((all_out, b_out), 0)
        val_loss = criterion(all_out, all_labels)
    print("Opt EPOCH: ", opt_epoch)
    auc = roc_auc_score(all_labels.numpy(), all_out.numpy())
    print(f"Test AUC: {auc:.5f}")
    aupr = average_precision_score(all_labels.numpy(), all_out.numpy())
    print(f"Test AUPRC: {aupr:.5f} \n")
     # # save model...
    torch.save(opt_model, f"{out_path}/split_{split}_masked_early_stop_frozen_trained_feature_model_fold_{fold}.pt")

    return auc, aupr


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # path to extracted encoded feature dir
    data_path = str(argv[1])
    # path to label file dir
    label_path = str(argv[2])
    # path to write trained models
    out_path = str(argv[3])

    splits = [1,2,3,4,5,6,7,8,9,10]
    folds = [1,2,3,4,5]

    all_auc = []
    all_auprc = []
    for s in splits:
        print("New split")
        for f in folds:
            fold_auc, fold_auprc = train_fold(s, f, device, 1120, data_path, label_path, out_path)
            all_auc.append(fold_auc)
            all_auprc.append(fold_auprc)
            
        print(f"AUC mean: {statistics.mean(all_auc)}")
        print(f"AUC sd: {statistics.stdev(all_auc)}")

    print(data_path)
    print("AUC: ", all_auc)
    print("AUPRC: ", all_auprc)
