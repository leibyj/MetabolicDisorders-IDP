import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import pandas as pd

from sklearn.metrics import roc_curve

from models import MIL_classifier
from datasets.patch_datasets import FeatureBagDataset, collate_bag_batches

from sys import argv

def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort().iloc[0]]
    return roc_t['threshold']


def main(split, fold, device, data_path, label_path, model_path):

    te_md = pd.read_csv(f"{label_path}/{split}/fold_{fold}_cv_files_TESTING.csv", names=['ID', 'label'])
    te_files = te_md.ID.tolist()
    te_labels = te_md.label.tolist()

    te_bag_data = FeatureBagDataset(data_path = data_path,
                         files = te_files,
                         labels = te_labels)

    te_dl = DataLoader(te_bag_data, batch_size=100, shuffle=False, collate_fn=collate_bag_batches)

    mod = torch.load(f"{model_path}/split_{split}_masked_early_stop_frozen_trained_feature_model_fold_{fold}.pt", map_location=device)
    mod.eval()

    te_out = torch.empty(0)
    te_lab = torch.empty(0)
    fns = []
    for i, (dat, lab, fn) in enumerate(te_dl):
        print(lab.shape)
        te_lab = torch.cat((te_lab, lab), axis=0)
        print(te_lab.shape)
        for d in dat:
            d = d.to(device)
            out, _, _ = mod(d)
            out = out.unsqueeze(0).detach().cpu()
            te_out = torch.cat((te_out, out), axis=0)
        fns.extend(fn)
    print(find_optimal_cutoff(te_lab, te_out))

if __name__ == "__main__": 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # path to pretrained classifier models
    model_path = str(argv[1])
    # path to extracted encoded feature dir
    data_path = str(argv[2])
    # path to label file dir
    label_path = str(argv[3])
    folds = [1,2,3,4,5]
    for f in folds:
        main(2, f, device, data_path, label_path, model_path)