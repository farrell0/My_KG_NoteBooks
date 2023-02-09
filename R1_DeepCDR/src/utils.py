"""
utils.py contains methods that can not be immediately categorized into any of the
four stages.
"""

import warnings

import matplotlib.pyplot as plt
import numpy
import scipy.sparse as sp

with warnings.catch_warnings():
    from deepchem.feat import ConvMolFeaturizer

import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

TCGA_label_set = [
    "ALL",
    "BLCA",
    "BRCA",
    "CESC",
    "DLBC",
    "LIHC",
    "LUAD",
    "ESCA",
    "GBM",
    "HNSC",
    "KIRC",
    "LAML",
    "LCML",
    "LGG",
    "LUSC",
    "MESO",
    "MM",
    "NB",
    "OV",
    "PAAD",
    "SCLC",
    "SKCM",
    "STAD",
    "THCA",
    "COAD/READ",
]


def normalized_adj(adj):
    adj = adj + numpy.eye(adj.shape[0])
    d = sp.diags(numpy.power(numpy.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def calculate_graph_feat(feat_mat, adj_list):
    max_atoms = 100
    assert feat_mat.shape[0] == len(adj_list)
    feat = numpy.zeros((max_atoms, feat_mat.shape[-1]), dtype="float32")
    adj_mat = numpy.zeros((max_atoms, max_atoms), dtype="float32")
    feat[: feat_mat.shape[0], :] = feat_mat
    for i, nodes in enumerate(adj_list):
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert numpy.allclose(adj_mat, adj_mat.T)
    adj_ = adj_mat[: len(adj_list), : len(adj_list)]
    adj_2 = adj_mat[len(adj_list) :, len(adj_list) :]
    norm_adj_ = normalized_adj(adj_)
    norm_adj_2 = normalized_adj(adj_2)

    adj_mat[: len(adj_list), : len(adj_list)] = norm_adj_
    adj_mat[len(adj_list) :, len(adj_list) :] = norm_adj_2

    return Data(x=torch.tensor(feat), edge_index=remove_self_loops(torch.tensor(adj_mat).to_sparse_coo().indices())[0])


def smiles_to_pyg(smiles):
    f = ConvMolFeaturizer()
    mol = f(smiles)[0]
    return calculate_graph_feat(mol.get_atom_features(), mol.get_adjacency_list())


def highlight_df(x):
    if x.name == "rmse":
        return ["font-weight: bold" if v == x.min() else "" for v in x]
    return ["font-weight: bold" if v == x.max() else "" for v in x]


def plot_prediction(y, ypred):
    plt.plot(y, ypred, "o", markersize=2)
    plt.title("DeepCDR predictions", loc="left")
    plt.xlabel("IC50 Observed")
    plt.ylabel("IC50 Predicted")
    plt.xlim([y.min(), y.max()])
    plt.ylim([y.min(), y.max()])
    plt.show()
