import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np


class FeatureBagDataset(Dataset):
    def __init__(self, data_path, files, labels):
        """
        Dataset used for concatenated features for a full volume (Concatenated features for each patch)
        --
        data_path: path to data dir
        files: list of filenames to include in dataset. (Files are tensors PxF P == # patches, F == size of feature map)
        labels: list of labels associated with file names
        """
        self.data_path = data_path
        self.data = [[file_name, lab] for file_name, lab in zip(files, labels)] # list of ["file name", label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, label = self.data[idx]
        label = torch.tensor(label)
        # load .pt file
        dat = torch.load(self.data_path + file_name)

        return dat, label, file_name # return data as tensor (P x F), label, file_name

def collate_bag_batches(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target).unsqueeze(1).float()
    fn = [item[2] for item in batch]

    return data, target, fn # returns data batch as list of length batch size tensors (P x F)
