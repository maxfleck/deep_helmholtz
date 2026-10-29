import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class ListedDataset(Dataset):
    def __init__(self,
            data_csv_paths: list,
            input_features: list,
            target_features: list,
            id_column: str = "ID",
            mode = "repeat"
        ):
        self.mode = mode
        self.data_csv_paths = data_csv_paths
        self.id_column = id_column
        self.input_features = input_features
        self.target_features = target_features

        lens = []
        self.Y_tensors = []
        self.X_tensors = []
        self.I_tensors = []
        for i, path in enumerate(data_csv_paths):
            data = pd.read_csv(path)
            lens.append(len(data))
            X = data[input_features[i]].values.astype(np.float32)
            Y = data[target_features[i]].values.astype(np.float32)
            I = data[id_column].values.astype(np.int32)

            X_tensor = torch.tensor(X, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)
            I_tensor = torch.tensor(I, dtype=torch.long)
            self.X_tensors.append(X_tensor)
            self.Y_tensors.append(Y_tensor)
            self.I_tensors.append(I_tensor)
        self.lens = lens
        self.ratios = [l / max(lens) for l in lens]
        self.len = max(lens)
        return 
        
    def __getitem__(self, index):
        """
        Returns a concatenated tensor of (ID, X, Y) for all datasets at the given index.
        The index is wrapped around for each dataset based on its length.

        a) Return a list of tuples: [(ID1, X1, Y1), (ID2, X2, Y2), ...]
        b) The return is concatenated along dimension 1. [changed later?]
        """
        # print("index ", index)
        returns = []
        for i, (X_tensor, Y_tensor) in enumerate(zip(self.X_tensors, self.Y_tensors)):
            l = self.lens[i]
            if self.mode == "repeat":
                idx = index % self.lens[i]
            else:
                if index >= l:
                    continue  # Skip this dataset for indices beyond its length
                idx = index
            I_tensor = self.I_tensors[i]
            returns.append((I_tensor[idx], X_tensor[idx], Y_tensor[idx]))
        return returns
    
    def __len__(self):
        return self.len
