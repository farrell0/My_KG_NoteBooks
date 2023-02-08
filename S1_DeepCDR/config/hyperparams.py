"""
Model and training related hyperparameters that the user can modify goes here.
"""

import dataclasses


@dataclasses.dataclass
class DeepCDRConfig:
    genomics: bool = True
    epigenomics: bool = True
    transcriptomics: bool = True


@dataclasses.dataclass
class TrainingHyperParams:
    epochs: int = 50
    training_samples: int = 89598
    valid_samples: int = 4716
    batch_size: int = 64
    torch_num_threads: int = 8  # Should be set to 50% of the available vCPUs for best performances
    patience: int = 15


def load_hyperparams():
    return DeepCDRConfig(), TrainingHyperParams()
