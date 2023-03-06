"""
Dataloader and Dataset related abstractions.
"""

from collections import namedtuple

import numpy
import pandas
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

DeepCDRData = namedtuple("DeepCDRData", "drug, genomics, transcriptomics, epigenomics, y")


class DeepCDRDataset(Dataset):
    def __init__(self, df: pandas.DataFrame, smiles_dict: dict):
        """Initialize the DeepCDRDataset object.

        Args:
            df: A pandas DataFrame with every pairs of drugs-cell_lines.
            smiles_dict: A dictionary with the PyG graphs representation of SMILES.
        """
        self.df = df.reset_index(drop=True)
        self.len = df.shape[0]
        self.smiles_dict = smiles_dict

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            self.smiles_dict[self.df["smiles"][index]],
            self.df["label"][index],
            self.df["genomics_mutation"][index],
            self.df["genomics_expression"][index],
            self.df["genomics_methylation"][index],
        )


def collate_fn(batch):
    smiles, label, genomics_mutation, genomics_expression, genomics_methylation = zip(*batch)
    smiles = Batch.from_data_list(smiles)
    label = torch.tensor(label, dtype=torch.float32)
    genomics_mutation = torch.tensor(numpy.array(genomics_mutation, dtype=numpy.float32), dtype=torch.float32)
    genomics_expression = torch.tensor(numpy.array(genomics_expression, dtype=numpy.float32), dtype=torch.float32)
    genomics_methylation = torch.tensor(numpy.array(genomics_methylation, dtype=numpy.float32), dtype=torch.float32)

    return DeepCDRData(smiles, genomics_mutation, genomics_expression, genomics_methylation, label)


class DeepCDRDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
        )
