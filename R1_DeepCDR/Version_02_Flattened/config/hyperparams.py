"""
Model and training related hyperparameters.
"""

import dataclasses
import os

from src import utils

# pylint: disable=too-many-instance-attributes


###########################################################################


@dataclasses.dataclass
class InputConfig:
    """Data class that includes all the parameters needed to setup the recipe."""

    # MMM
    # Number of partitions
    #  num_partitions: int = 4
    num_partitions: int = 3

    #     # Trained graph
    #     use_train_rdg: bool = False
    #     trained_rdg_path: str = "gs://hls-dataset-bucket/DeepCDR_trained"

    # MMM
    #
    #     # Generate graph
    #     # Nodes paths
    #     cell_lines_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/cell_lines.csv"
    #     drugs_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/drugs.csv"
    #     gdsc_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/gdsc.csv"
    #     genes_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/genes.csv"
    #     
    #     # Edges paths
    #     gdsc_cell_line_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/gdsc_cell_line_edges.csv"
    #     gdsc_drug_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/gdsc_drug_edges.csv"
    #     cell_line_gene_expression_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/cell_line_gene_expression_edges.csv"
    #     cell_line_gene_methylation_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/cell_line_gene_methylation_edges.csv"
    #     cell_line_gene_mutation_path: str = "gs://katana-demo-datasets/csv-datasets/DeepCDR/cell_line_gene_mutation_edges.csv"

    # Split parameters
    test_size: float = 0.05
    random_state: int = 42

    # Save Graph path
    save_graph_path = None

    def __repr__(self):
        return utils.dataclass_repr(self)


###########################################################################


@dataclasses.dataclass
class DeepCDRConfig:
    """Data class that includes all the parameters needed to set up the model."""

    # Set the following to False if you would not like to use genomics features.
    genomics: bool = True

    # Set the following to False if you would not like to epigenomics features.
    epigenomics: bool = True

    # Set the following to False if you would not like to use transcriptomics features.
    transcriptomics: bool = True

    def __repr__(self):
        return utils.dataclass_repr(self)


@dataclasses.dataclass
class TrainingConfig:
    """Data class that includes all the parameters specifying details of the training process."""

    # Batch size
    batch_size: int = 64

    # Number of epochs
    epochs: int = 80

    # Patience
    patience: int = 15

    # Name of the experiment
    experiment_name: str = "DeepCDR"

    # Set the following to a bucket path if you want to store tensorboard logs.
    tensorboard_path: str = None

    def __repr__(self):
        return utils.dataclass_repr(self)

    
###########################################################################
    

def load_input_config():
    """Generate input parameters."""
    return InputConfig()


def load_model_config():
    """Generate model parameters."""
    model_config = DeepCDRConfig()
    if os.environ.get("RED_SHORT_RECIPE"):
        model_config.epigenomics = False
        model_config.transcriptomics = False
    return model_config


def load_training_config():
    """Generate training parameters."""
    training_config = TrainingConfig()
    if os.environ.get("RED_SHORT_RECIPE"):
        training_config.epochs = 2
    return training_config
