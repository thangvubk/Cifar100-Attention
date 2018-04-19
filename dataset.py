from torch.utils.data import Dataset
from load_cifar100 import load_cifar100_dataset
import torch

class Cifar100_dataset(Dataset):
    def __init__(self, path, sample_range=None):
        self.inputs, self.labels = load_cifar100_dataset(path)
        if sample_range is not None:
            self.inputs = self.inputs[sample_range[0]: sample_range[1]]
            self.labels = self.labels[sample_range[0]: sample_range[1]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.inputs[idx].reshape(3, 32, 32)
        lbl = [self.labels[idx]]

        return torch.FloatTensor(inp), torch.LongTensor(lbl)

        

