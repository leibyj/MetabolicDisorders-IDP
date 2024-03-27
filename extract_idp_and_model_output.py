import torch
from torch.utils.data import DataLoader

import pandas as pd
from datasets.patch_datasets import FeatureBagDataset, collate_bag_batches

from sys import argv

"""
Extracts the classifier features (IPD) and model output from pretrained classifiers
"""

    
def get_features_and_output(split, fold, device, data_path, label_path, model_path, out_path):
    tr_md = pd.read_csv(f"{label_path}/{split}/fold_{fold}_cv_files_TRAINING.csv", names=['ID', 'label'])
    tr_files = tr_md.ID.tolist()
    tr_labels = tr_md.label.tolist()

    tr_bag_data = FeatureBagDataset(data_path = data_path,
                         files = tr_files,
                         labels = tr_labels)

    tr_dl = DataLoader(tr_bag_data, batch_size=100, shuffle=True, collate_fn=collate_bag_batches)

    val_md = pd.read_csv(f"{label_path}/{split}/fold_{fold}_cv_files_VALIDATION.csv", names=['ID', 'label'])
    val_files =val_md.ID.tolist()
    val_labels = val_md.label.tolist()

    val_bag_data = FeatureBagDataset(data_path = data_path,
                         files = val_files,
                         labels = val_labels)

    val_dl = DataLoader(val_bag_data, batch_size=100, shuffle=False, collate_fn=collate_bag_batches)

    te_md = pd.read_csv(f"{label_path}/{split}/fold_{fold}_cv_files_TESTING.csv", names=['ID', 'label'])
    te_files = te_md.ID.tolist()
    te_labels = te_md.label.tolist()

    te_bag_data = FeatureBagDataset(data_path = data_path,
                         files = te_files,
                         labels = te_labels)

    te_dl = DataLoader(te_bag_data, batch_size=100, shuffle=False, collate_fn=collate_bag_batches)

    mod = torch.load(f"{model_path}/split_{split}_masked_early_stop_frozen_trained_feature_model_fold_{fold}.pt", map_location=device)
    mod.eval()

    tr_feats = torch.empty(0)
    tr_out = torch.empty(0)
    fns = []
    for i, (dat, lab, fn) in enumerate(tr_dl):
        for d in dat:
            d = d.to(device)
            out, _, feats = mod(d)
            feats = feats.unsqueeze(0).detach().cpu()
            out = out.unsqueeze(0).detach().cpu()
            tr_feats = torch.cat((tr_feats, feats), axis=0)
            tr_out = torch.cat((tr_out, out), axis=0)
        fns.extend(fn)

    tr_feats = pd.DataFrame(tr_feats, index=fns)
    tr_feats.to_csv(f"{out_path}/split_{split}_frozen_train_features_fold_{fold}.csv")
    tr_out = pd.DataFrame(tr_out, index=fns)
    tr_out.to_csv(f"{out_path}/split_{split}_frozen_train_out_fold_{fold}.csv")

    val_feats = torch.empty(0)
    val_out = torch.empty(0)
    fns = []
    for i, (dat, lab, fn) in enumerate(val_dl):
        for d in dat:
            d = d.to(device)
            out, _, feats = mod(d)
            feats = feats.unsqueeze(0).detach().cpu()
            out = out.unsqueeze(0).detach().cpu()
            val_feats = torch.cat((val_feats, feats), axis=0)
            val_out = torch.cat((val_out, out), axis=0)
        fns.extend(fn)

    val_feats = pd.DataFrame(val_feats, index=fns)
    val_feats.to_csv(f"{out_path}/split_{split}_frozen_val_features_fold_{fold}.csv")
    val_out = pd.DataFrame(val_out, index=fns)
    val_out.to_csv(f"{out_path}/split_{split}_frozen_val_out_fold_{fold}.csv")
    
    te_feats = torch.empty(0)
    te_out = torch.empty(0)
    fns = []
    for i, (dat, lab, fn) in enumerate(te_dl):
        for d in dat:
            d = d.to(device)
            out, _, feats = mod(d)
            feats = feats.unsqueeze(0).detach().cpu()
            out = out.unsqueeze(0).detach().cpu()
            te_feats = torch.cat((te_feats, feats), axis=0)
            te_out = torch.cat((te_out, out), axis=0)
        fns.extend(fn)

    te_feats = pd.DataFrame(te_feats, index=fns)
    te_feats.to_csv(f"{out_path}/split_{split}_frozen_test_features_fold_{fold}.csv")
    te_out = pd.DataFrame(te_out, index=fns)
    te_out.to_csv(f"{out_path}/split_{split}_frozen_test_out_fold_{fold}.csv")

if __name__ == "__main__": 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # path to pretrained classifier models
    model_path = str(argv[1])
    # path to extracted encoded feature dir
    data_path = str(argv[2])
    # path to label file dir
    label_path = str(argv[3])
    # path to write files
    out_path = str(argv[4])

    splits = [1,2,3,4,5,6,7,8,9,10]
    folds = [1,2,3,4,5]
    for s in splits:
        for f in folds:
            get_features_and_output(s, f, device, data_path, label_path, model_path, out_path)
