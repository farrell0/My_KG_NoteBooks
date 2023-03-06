"""
DeepCDR model construction.
"""
import torch
import torch_geometric
from config.hyperparams import DeepCDRConfig
from katana.ai.model import Mlp, PygGppGat, PygGppWrapper
from katana.ai.model.enums import PygPoolingMethod


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, data):
        return data.unsqueeze(self.dim)


class CellLineModel(torch.nn.Module):
    def __init__(self, genomics=True, epigenomics=True, transcriptomics=True):
        if not (genomics or epigenomics or transcriptomics):
            raise ValueError("At least one of the multi-omics data need to be True.")
        super().__init__()
        self.genomics = genomics
        self.epigenomics = epigenomics
        self.transcriptomics = transcriptomics

        if genomics:
            self.genomics_model = torch.nn.Sequential(
                Unsqueeze(1),
                torch.nn.Conv1d(1, 50, kernel_size=(700), stride=5, padding="valid"),
                torch.nn.Tanh(),
                torch.nn.MaxPool1d(kernel_size=(5)),
                torch.nn.Conv1d(50, 30, kernel_size=(5), stride=2, padding="valid"),
                torch.nn.Tanh(),
                torch.nn.MaxPool1d(kernel_size=(10)),
                torch.nn.Flatten(1, -1),
                torch.nn.Linear(2010, 100),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.1),
            )

        if epigenomics:
            self.epigenomics_model = torch.nn.Sequential(
                Mlp(
                    in_channels=808,
                    out_channels=100,
                    hidden_layers=[256],
                    dropout=0.1,
                    activation=None,
                    batchnorm=torch.nn.BatchNorm1d,
                ),
                torch.nn.ReLU(inplace=True),
            )

        if transcriptomics:
            self.transcriptomics_model = torch.nn.Sequential(
                Mlp(
                    in_channels=697,
                    out_channels=100,
                    hidden_layers=[256],
                    dropout=0.1,
                    activation=None,
                    batchnorm=torch.nn.BatchNorm1d,
                ),
                torch.nn.ReLU(inplace=True),
            )

    def forward(self, genomics, transcriptomics, epigenomics):
        out = torch.tensor([])
        if self.genomics:
            genomics = self.genomics_model(genomics)
            out = torch.cat((out, genomics), 1)

        if self.transcriptomics:
            transcriptomics = self.transcriptomics_model(transcriptomics)
            out = torch.cat((out, transcriptomics), 1)

        if self.epigenomics:
            epigenomics = self.epigenomics_model(epigenomics)
            out = torch.cat((out, epigenomics), 1)
        return out


class DrugModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling_method = PygPoolingMethod.MAX
        self.gcn = torch_geometric.nn.models.GCN(
            in_channels=74,
            hidden_channels=256,
            num_layers=3,
            out_channels=100,
            dropout=0.1,
            act=torch.nn.ReLU(inplace=True),
            norm=torch_geometric.nn.norm.BatchNorm(256),
        )

    def forward(self, data):
        out = self.gcn(data.x, data.edge_index)
        out = self.pooling_method(out, data.batch)
        return out


class DeepCDR(torch.nn.Module):
    """An DeepCDR model for Cancer Drug Response"""

    def __init__(self, config: DeepCDRConfig):
        super().__init__()
        self.genomics = config.genomics
        self.epigenomics = config.epigenomics
        self.transcriptomics = config.transcriptomics
        self.drug_model = DrugModel()
        self.cell_line_model = CellLineModel(self.genomics, self.epigenomics, self.transcriptomics)
        self.predictor_model = torch.nn.Sequential(
            torch.nn.Linear((self.genomics + self.epigenomics + self.transcriptomics + 1) * 100, 300),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1),
            Unsqueeze(1),
            torch.nn.Conv1d(1, 30, kernel_size=(150), stride=1, padding="valid"),
            torch.nn.MaxPool1d(kernel_size=(2)),
            torch.nn.Conv1d(30, 10, kernel_size=(5), stride=1, padding="valid"),
            torch.nn.MaxPool1d(kernel_size=(3)),
            torch.nn.Conv1d(10, 5, kernel_size=(5), stride=1, padding="valid"),
            torch.nn.MaxPool1d(kernel_size=(3)),
            torch.nn.Dropout(0.1),
            torch.nn.Flatten(1, -1),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(30, 1),
        )

    def forward(self, data):
        drug_embedding = self.drug_model(data.drug)
        cell_line_embedding = self.cell_line_model(data.genomics, data.transcriptomics, data.epigenomics)
        out = torch.cat((drug_embedding, cell_line_embedding), 1)
        out = self.predictor_model(out)
        return out.squeeze()
