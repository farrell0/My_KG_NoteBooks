"""
In the dataloader.py file, add all the dataloader and dataset related abstractions.
"""

from collections import namedtuple

import numpy
import pandas
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

DeepCDRData = namedtuple("DeepCDRData", "drug, genomics, transcriptomics, epigenomics, y")


class DeepCDRDataset(Dataset):
    def __init__(self, dataframe: pandas.DataFrame, smiles_dict: dict):
        """Initialize the DeepCDRDataset object.

        Args:
            dataframe: A pandas DataFrame with every pairs of drugs-cell_lines.
            smiles_dict: A dictionary with the PyG graphs representation of SMILES.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.len = dataframe.shape[0]
        self.smiles_dict = smiles_dict

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            self.smiles_dict[self.dataframe["smiles"][index]],
            self.dataframe["label"][index],
            self.dataframe["genomics_mutation"][index],
            self.dataframe["genomics_expression"][index],
            self.dataframe["genomics_methylation"][index],
        )


def collate_fn(batch):
    smiles, label, genomics_mutation, genomics_expression, genomics_methylation = zip(*batch)
    smiles = Batch.from_data_list(smiles)
    label = torch.tensor(label, dtype=torch.float32)
    genomics_mutation = torch.tensor(numpy.array(genomics_mutation))
    genomics_expression = torch.tensor(numpy.array(genomics_expression))
    genomics_methylation = torch.tensor(numpy.array(genomics_methylation))

    return DeepCDRData(smiles, genomics_mutation, genomics_expression, genomics_methylation, label)
