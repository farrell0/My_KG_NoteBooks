"""
Pipeline for DeepCDR recipe
"""

import logging

import cloudpickle
import numpy
import pandas
import torch

logging.getLogger("deepchem").setLevel(logging.CRITICAL)

import config.hyperparams
import src.dataloader
import src.model
import src.utils
from katana import remote
from katana_enterprise.ai import loss, train
from katana_enterprise.ai.hls.hls_preprocessing_graph import HlsPreprocessingGraph
from katana_enterprise.ai.torch import ReduceMethod
from katana_enterprise.ai.train import DistTrainer, Trainer
from katana_enterprise.distributed import Experiment, ExperimentManager, Graph, single_host
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

cloudpickle.register_pickle_by_value(config.hyperparams)
cloudpickle.register_pickle_by_value(src.dataloader)
cloudpickle.register_pickle_by_value(src.model)
cloudpickle.register_pickle_by_value(src.utils)


from config.hyperparams import DeepCDRConfig, TrainingHyperParams
from scipy import stats
from src import preprocessing
from src.dataloader import DeepCDRDataset, collate_fn
from src.model import DeepCDR
from src.utils import highlight_df, plot_prediction, smiles_to_pyg


class RecipePipeline:
    def __init__(self, graph: remote.Graph) -> None:
        """Initialization of the pipeline

        Args:
            graph: The input graph. Must be a katana remote Graph.
        """
        self.graph = graph

    def feature_generator(self) -> None:
        """Preprocessing and generation of features in the graph"""
        df_cell_lines, df_target = preprocessing.get_drug_cell_line_pairs(self.graph)
        self.graph.run(lambda g: g.upsert_graph_property("cell_lines", df_cell_lines))
        self.graph.run(lambda g: g.upsert_graph_property("df_target", df_target))

    def train(self, model_params: DeepCDRConfig, training_params: TrainingHyperParams) -> None:
        """Training of DeepCDR model

        Args:
            model_params: The model parameters for the pipeline.
            training_params: The parameters used for training models.
        """
        # A method to train the AI model and save the model as a
        # graph property using context manager

        pickled_model_params = cloudpickle.dumps(model_params)
        pickled_training_params = cloudpickle.dumps(training_params)

        def remote_training(graph: Graph, pickeld_model_params, pickeld_training_params):
            # unpickle the parameters in the graph workers
            model_params = cloudpickle.loads(pickeld_model_params)
            training_params = cloudpickle.loads(pickeld_training_params)

            torch.set_num_threads(training_params.torch_num_threads)

            df_cell_lines = graph.get_graph_property("cell_lines")
            df_target = graph.get_graph_property("df_target")

            # smiles featurization on the fly
            smiles_dict = {}
            for smiles in df_target["smiles"].unique():
                smiles_dict[smiles] = smiles_to_pyg(smiles)

            # We randomly split the datainto 95% training set and 5% test set on the cell line or drug level
            df_train, df_test = train_test_split(
                df_target, test_size=0.05, random_state=42, stratify=df_target["tcga_code"], shuffle=True
            )

            df_train = df_train[: training_params.training_samples]
            df_test = df_test[: training_params.valid_samples]

            df_train = numpy.array_split(df_train, graph.num_partitions)[graph.partition_id]
            df_valid = numpy.array_split(df_test, graph.num_partitions)[graph.partition_id]

            df_train = pandas.merge(df_cell_lines, df_train, left_index=True, right_on="CCLE_ID")
            df_valid = pandas.merge(df_cell_lines, df_valid, left_index=True, right_on="CCLE_ID")

            deepcdr_train = DeepCDRDataset(dataframe=df_train, smiles_dict=smiles_dict)
            deepcdr_valid = DeepCDRDataset(dataframe=df_valid, smiles_dict=smiles_dict)

            train_data_loader = DataLoader(
                deepcdr_train, batch_size=training_params.batch_size, shuffle=True, collate_fn=collate_fn
            )
            val_data_loader = DataLoader(
                deepcdr_valid, batch_size=training_params.batch_size, shuffle=False, collate_fn=collate_fn
            )

            model = DeepCDR(model_params)
            optimizer = torch.optim.Adam(model.parameters())
            tracker = train.DistTracker(on_epoch_end=train.tracker.pretty_print, src_rank=0)

            # Initialize the distributed trainer with given configurations
            trainer = DistTrainer(
                model=model,
                train_loss_fn=loss.L2Loss(),
                validation_metric_fn=lambda x, y: stats.pearsonr(x, y)[0],
                train_loader=train_data_loader,
                validation_loader=val_data_loader,
                validation_reduce_method=ReduceMethod.MEAN,
                optimizer=optimizer,
                epochs=training_params.epochs,
                maximization=True,
                patience=training_params.patience,
                tracker=tracker,
            )

            # Model training
            trained_model, best_validation_metric = trainer.train()
            experiment_mgr = ExperimentManager(graph)
            experiment = Experiment({"training_config": training_params}, trained_model)
            experiment_mgr.upsert("DeepCDR", experiment)

            print(best_validation_metric)
            assert best_validation_metric["validation_metric"] > 0.89

        self.graph.run(lambda g: remote_training(g, pickled_model_params, pickled_training_params))

    def test(self, training_params: TrainingHyperParams) -> None:
        """Testing of DeepCDR model

        Args:
            training_params: The parameters used for training models.
        """
        pickled_training_params = cloudpickle.dumps(training_params)

        def remote_testing(graph, pickeld_training_params):

            training_params = cloudpickle.loads(pickeld_training_params)

            torch.set_num_threads(training_params.torch_num_threads)

            df_cell_lines = graph.get_graph_property("cell_lines")
            df_target = graph.get_graph_property("df_target")

            # smiles featurization on the fly
            smiles_dict = {}
            for smiles in df_target["smiles"].unique():
                smiles_dict[smiles] = smiles_to_pyg(smiles)

            # We randomly split the datainto 80% training set and 20% test set on the cell line or drug level
            _, df_test = train_test_split(
                df_target, test_size=0.05, random_state=42, stratify=df_target["tcga_code"], shuffle=True
            )

            df_test = df_test[: training_params.valid_samples]
            df_test = pandas.merge(df_cell_lines, df_test, left_index=True, right_on="CCLE_ID")
            deepcdr_test = DeepCDRDataset(dataframe=df_test, smiles_dict=smiles_dict)

            test_data_loader = DataLoader(
                deepcdr_test, batch_size=training_params.batch_size, shuffle=False, collate_fn=collate_fn
            )

            experiment_mgr = ExperimentManager(graph)
            experiment = experiment_mgr.get("DeepCDR", include_model=True)
            model = experiment.model

            validation_metric_fn = {
                "pearson": lambda x, y: stats.pearsonr(x, y)[0],
                "rmse": lambda x, y: mean_squared_error(x, y, squared=False),
                "spearman": lambda x, y: stats.spearmanr(x, y).correlation,
            }

            trainer = Trainer(
                model=model,
                train_loss_fn=loss.L2Loss(),
                validation_metric_fn=validation_metric_fn,
                train_loader=None,
                validation_loader=None,
                optimizer=torch.optim.Adam(model.parameters()),
                epochs=0,
                primary_validation_metric="pearson",
            )

            # Trainer test for more metrics
            return single_host(trainer.test(test_data_loader)[0])

        res = self.graph.run(lambda g: remote_testing(g, pickled_training_params))
        deepcdr_baseline = {"pearson": 0.923, "rmse": 1.056, "spearman": 0.903}
        res = pandas.DataFrame([deepcdr_baseline, res], index=["DeepCDR", "DeepCDR_Katana"])
        return res.style.apply(highlight_df)

    def plot(self, training_params: TrainingHyperParams) -> None:
        """Testing of DeepCDR model

        Args:
            training_params: The parameters used for training models.
        """
        pickled_training_params = cloudpickle.dumps(training_params)

        def remote_plot(graph, pickeld_training_params):

            training_params = cloudpickle.loads(pickeld_training_params)

            torch.set_num_threads(training_params.torch_num_threads)

            df_cell_lines = graph.get_graph_property("cell_lines")
            df_target = graph.get_graph_property("df_target")

            # smiles featurization on the fly
            smiles_dict = {}
            for smiles in df_target["smiles"].unique():
                smiles_dict[smiles] = smiles_to_pyg(smiles)

            # We randomly split the datainto 80% training set and 20% test set on the cell line or drug level
            _, df_test = train_test_split(
                df_target, test_size=0.05, random_state=42, stratify=df_target["tcga_code"], shuffle=True
            )

            df_test = df_test[: training_params.valid_samples]
            df_test = pandas.merge(df_cell_lines, df_test, left_index=True, right_on="CCLE_ID")
            deepcdr_test = DeepCDRDataset(dataframe=df_test, smiles_dict=smiles_dict)

            test_data_loader = DataLoader(
                deepcdr_test, batch_size=training_params.batch_size, shuffle=False, collate_fn=collate_fn
            )

            experiment_mgr = ExperimentManager(graph)
            experiment = experiment_mgr.get("DeepCDR", include_model=True)
            model = experiment.model

            validation_metric_fn = {
                "pearson": lambda x, y: stats.pearsonr(x, y)[0],
                "rmse": lambda x, y: mean_squared_error(x, y, squared=False),
                "spearman": lambda x, y: stats.spearmanr(x, y).correlation,
            }

            trainer = Trainer(
                model=model,
                train_loss_fn=loss.L2Loss(),
                validation_metric_fn=validation_metric_fn,
                train_loader=None,
                validation_loader=None,
                optimizer=torch.optim.Adam(model.parameters()),
                epochs=0,
                primary_validation_metric="pearson",
            )

            _, ypred = trainer.test(test_data_loader)
            y = df_test["label"]

            # Trainer test for more metrics
            return single_host([y, ypred])

        y, ypred = self.graph.run(lambda g: remote_plot(g, pickled_training_params))
        plot_prediction(y, ypred)

    def infer(self, training_params: TrainingHyperParams, drug: str, cell_line: str) -> None:
        """Run trained model to save node embeddings

        Args:
            training_params: The parameters used for training models.
            drug: SMILES of a Drug.
            cell_line: CCLE ID of a cell_line.
        """
        pickled_training_params = cloudpickle.dumps(training_params)

        def remote_infer(graph: Graph, pickled_training_params, drug, cell_line):

            training_params = cloudpickle.loads(pickled_training_params)

            torch.set_num_threads(training_params.torch_num_threads)
            df_cell_lines = graph.get_graph_property("cell_lines")

            # smiles featurization on the fly
            smiles_dict = {}
            smiles_dict[drug] = smiles_to_pyg(drug)

            df_test = pandas.DataFrame({"smiles": drug, "CCLE_ID": cell_line, "label": 0}, index=[0])
            df_test = pandas.merge(df_cell_lines, df_test, left_index=True, right_on="CCLE_ID")
            if df_test.shape[0] == 0:
                print("You need to use a cell_line defined in the graph: ", df_cell_lines.index.values.tolist())
            deepcdr_test = DeepCDRDataset(dataframe=df_test, smiles_dict=smiles_dict)
            test_data_loader = DataLoader(deepcdr_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
            experiment_mgr = ExperimentManager(graph)
            experiment = experiment_mgr.get("DeepCDR", include_model=True)
            model = experiment.model
            model.eval()
            with torch.no_grad():
                pred = model(next(iter(test_data_loader))).item()
                return single_host(pred)

        return self.graph.run(lambda g: remote_infer(g, pickled_training_params, drug, cell_line))

    def infer_embeddings(self, model_params: DeepCDRConfig) -> None:
        """Run trained model to save node embeddings

        Args:
            model_params: The model parameters for the pipeline.
        """
        pickled_model_params = cloudpickle.dumps(model_params)

        def remote_infer_embeddings(graph: Graph, pickeld_model_params):
            model_params = cloudpickle.loads(pickeld_model_params)
            df_cell_lines = graph.get_graph_property("cell_lines")
            experiment_mgr = ExperimentManager(graph)
            experiment = experiment_mgr.get("DeepCDR", include_model=True)
            model = experiment.model

            feat_obj = HlsPreprocessingGraph(graph)

            def drug_feature(smiles):
                drug = smiles_to_pyg(smiles)
                feat = model.drug_model(drug).squeeze()
                return feat.numpy().astype("float32")

            def genomics_feature(ccle_id):
                if not ccle_id in df_cell_lines.index:
                    feat = numpy.zeros(100).astype("float32")
                    return feat
                ccle = torch.tensor(df_cell_lines["genomics_mutation"][ccle_id], dtype=torch.float32)
                ccle = ccle.reshape(1, -1)
                feat = model.cell_line_model.genomics_model(ccle).squeeze()
                feat = feat.numpy().astype("float32")
                return feat

            def transcriptomics_feature(ccle_id):
                if not ccle_id in df_cell_lines.index:
                    feat = numpy.zeros(100).astype("float32")
                    return feat
                ccle = torch.tensor(df_cell_lines["genomics_expression"][ccle_id], dtype=torch.float32)
                ccle = ccle.reshape(1, -1)
                feat = model.cell_line_model.transcriptomics_model(ccle).squeeze()
                return feat.numpy().astype("float32")

            def epigenomics_feature(ccle_id):
                if not ccle_id in df_cell_lines.index:
                    feat = numpy.zeros(100).astype("float32")
                    return feat
                ccle = torch.tensor(df_cell_lines["genomics_methylation"][ccle_id], dtype=torch.float32)
                ccle = ccle.reshape(1, -1)
                feat = model.cell_line_model.epigenomics_model(ccle).squeeze()
                feat = feat.numpy().astype("float32")
                return feat

            model.eval()
            with torch.no_grad():
                feat_obj.upsert_featurizer_feature(
                    in_feature_name="smiles",
                    out_feature_name="drug_embeddings",
                    featurizer=drug_feature,
                    node_types=["DRUG"],
                )
                if model_params.genomics:
                    feat_obj.upsert_featurizer_feature(
                        in_feature_name="CCLE_ID",
                        out_feature_name="genomics_embeddings",
                        featurizer=genomics_feature,
                        node_types=["CELL_LINE"],
                    )
                if model_params.transcriptomics:
                    feat_obj.upsert_featurizer_feature(
                        in_feature_name="CCLE_ID",
                        out_feature_name="transcriptomics_embeddings",
                        featurizer=transcriptomics_feature,
                        node_types=["CELL_LINE"],
                    )
                if model_params.epigenomics:
                    feat_obj.upsert_featurizer_feature(
                        in_feature_name="CCLE_ID",
                        out_feature_name="epigenomics_embeddings",
                        featurizer=epigenomics_feature,
                        node_types=["CELL_LINE"],
                    )

        self.graph.run(lambda g: remote_infer_embeddings(g, pickled_model_params))
