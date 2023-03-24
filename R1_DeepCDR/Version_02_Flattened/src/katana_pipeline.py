"""
Pipeline for DeepCDR recipe.
"""

import cloudpickle
import ipywidgets as widgets
import pandas
import torch
from katana import remote
from katana.ai import loss, train
from katana.ai.preprocessing import FunctionTransform
from katana.ai.remote import preprocessing as remote_preprocessing
from katana.ai.torch import ReduceMethod
from katana.ai.train import DistTrainer, PredictionEngine
from katana.distributed import Experiment, ExperimentManager, Graph, single_host
from katana.remote.experiments import RemoteExperimentManager
from rdkit import Chem
from scipy import stats
from sklearn.metrics import mean_squared_error
from src import dataloader, model, preprocessing, split, utils
from src.dataloader import DeepCDRDataloader, DeepCDRDataset
from src.model import DeepCDR
from src.split import SplitType

from config import hyperparams  # isort: skip
from config.hyperparams import DeepCDRConfig, TrainingConfig  # isort: skip

cloudpickle.register_pickle_by_value(hyperparams)
cloudpickle.register_pickle_by_value(dataloader)
cloudpickle.register_pickle_by_value(model)
cloudpickle.register_pickle_by_value(utils)
cloudpickle.register_pickle_by_value(split)


##############################################################################3


class RecipePipeline:
    
    def __init__(self, graph: remote.Graph) -> None:
        """Initialization of the pipeline
        Args:
            graph: The input graph
        """
        
        # MMM
        #
        print("This ran MMM 01")
        
        self.graph = graph

    @utils.disable_warnings
    def stats(self) -> None:
        """Print statistics of the graph"""
        num_nodes = self.graph.num_nodes()
        num_edges = self.graph.num_edges()
        drugs = utils.get_node_number(self.graph, "DRUG")
        cell_lines = utils.get_node_number(self.graph, "CELL_LINE")
        pairs = self.graph.query(
            """
            MATCH (a:DRUG)<-[:FOR_DRUG]-(:GDSC)-[:HAS_CELL_LINE]->(:CELL_LINE)
            RETURN COUNT(a) as n
            """
        ).head()["n"][0]
        print(f"Number of nodes in the graph: {num_nodes}")
        print(f"Number of edges in the graph: {num_edges}")
        print(f"Number of DRUG nodes in the graph: {drugs}")
        print(f"Number of CELL_LINE nodes in the graph: {cell_lines}")
        print(f"Number of pairs between a drug and a cell line: {pairs}")

        
    ##############################################################3
    
    
    @utils.disable_warnings
    def feature_generator(self) -> None:
        """Preprocessing and generation of features in the graph"""
#       features = {
#           "HAS_EXPRESSION_OBSERVATION": "genomics_expression",
#           "HAS_METHYLATION_OBSERVATION": "genomics_methylation",
#           "HAS_MUTATION_OBSERVATION": "genomics_mutation",
#       }
#       for u, v in features.items():
#           print(f"Collecting {v} features...")
#           preprocessing.set_cell_line_property(self.graph, u, v)

        print("Deleting Cell lines with NULL features...")
        preprocessing.remove_null_cells(self.graph)
#       print("Deleting single nodes")
#       preprocessing.delete_single_node(self.graph)
        
        
    ##############################################################3
        

    @utils.disable_warnings
    def split_generator(self, input_hp) -> None:
        """Add split edges in the graph

        Args:
            input_hp: The graph input parameters.
        """

        print("Deleting old split")
        split.delete_split(self.graph)
        print("Generating split")
        split_df = pandas.concat(
            self.graph.query(
                """
            MATCH (a:DRUG)<-[:FOR_DRUG]-(:GDSC)-[u:HAS_CELL_LINE]->(c:CELL_LINE)
            RETURN u.label as label, c.tcga_code as tcga_code,
            a.id as drug_id, c.id as cell_line_id
            """
            )
        )
        split_df = split_df.sort_values(by=["drug_id", "cell_line_id", "label"]).reset_index(drop=True)
        df_split = split.generate_split(split_df, input_hp.random_state, input_hp.test_size)
        print("Upsert split on the graph")
        split.upsert_split(self.graph, df_split)

    @utils.disable_warnings
    def train(self, model_hp: DeepCDRConfig, training_hp: TrainingConfig) -> None:
        """Training of DeepCDR model

        Args:
            model_hp: The model parameters for the pipeline.
            training_hp: The parameters used for training models.
        """

        def remote_training(graph: Graph, model_hp, training_hp):
            df_train = utils.pairs_query_str(graph, SplitType.TRAIN)
            df_valid = utils.pairs_query_str(graph, SplitType.VAL)

            print("train data shape:", df_train.shape)
            print("valid data shape:", df_valid.shape)
            print()

            deepcdr_train = DeepCDRDataset(df_train, utils.smiles_dict(df_train["smiles"].unique()))
            deepcdr_valid = DeepCDRDataset(df_valid, utils.smiles_dict(df_valid["smiles"].unique()))

            train_data_loader = DeepCDRDataloader(deepcdr_train, batch_size=training_hp.batch_size, shuffle=True)
            val_data_loader = DeepCDRDataloader(deepcdr_valid, batch_size=training_hp.batch_size, shuffle=False)

            model = DeepCDR(model_hp)
            optimizer = torch.optim.Adam(model.parameters())
            tracker = train.DistTracker(on_epoch_end=train.tracker.pretty_print)
            if training_hp.tensorboard_path is not None:
                tracker = train.DistTensorboardTracker(training_hp.tensorboard_path)

            # Initialize the distributed trainer with given configurations
            trainer = DistTrainer(
                model=model,
                train_loss_fn=loss.L2Loss(),
                validation_metric_fn=lambda x, y: mean_squared_error(x, y, squared=False),
                train_loader=train_data_loader,
                validation_loader=val_data_loader,
                validation_reduce_method=ReduceMethod.MEAN,
                optimizer=optimizer,
                epochs=training_hp.epochs,
                maximization=False,
                patience=training_hp.patience,
                tracker=tracker,
                find_unused_parameters=False,
            )

            # Model training
            trained_model, best_validation_metric = trainer.train()
            experiment_mgr = ExperimentManager(graph)
            experiment = Experiment({"model_state": trained_model.state_dict()}, trained_model)
            experiment_mgr.upsert(training_hp.experiment_name, experiment)
            return single_host(best_validation_metric["validation_metric"])

        return self.graph.run(lambda g: remote_training(g, model_hp, training_hp))

    @utils.disable_warnings
    def test(self, training_hp: TrainingConfig) -> None:
        """Testing of DeepCDR model

        Args:
            training_hp: The parameters used for training models.
        """

        def remote_testing(graph: Graph, training_hp):
            df_test = utils.pairs_query_str(graph, SplitType.TEST)

            if graph.partition_id != 0:
                return None

            deepcdr_test = DeepCDRDataset(df_test, utils.smiles_dict(df_test["smiles"].unique()))
            data_loader_test = DeepCDRDataloader(deepcdr_test, batch_size=training_hp.batch_size, shuffle=False)

            experiment_mgr = ExperimentManager(graph)
            experiment = experiment_mgr.get(training_hp.experiment_name, include_model=True)

            metric_fn = {
                "pearson": lambda x, y: stats.pearsonr(x, y.squeeze())[0],
                "rmse": lambda x, y: mean_squared_error(x, y.squeeze(), squared=False),
                "spearman": lambda x, y: stats.spearmanr(x, y.squeeze()).correlation,
            }

            engine = PredictionEngine(experiment.model)
            labels, predictions = engine.get_labels_and_predictions(data_loader_test)
            metrics = engine.calculate_metrics(labels, predictions, metric_fn)

            return metrics, predictions, labels

        metrics, predictions, labels = self.graph.run(lambda g: remote_testing(g, training_hp))
        return utils.compare_baselines(metrics), predictions, labels

    @utils.disable_warnings
    def plot(self, labels, ypred) -> None:
        """Plotting of DeepCDR model

        Args:
            ytrue: Ground truth labels.
            ypred: Predictions of the model.
        """
        utils.plot_prediction(labels, ypred)

    @utils.disable_warnings
    def infer(self, training_hp: TrainingConfig, drug=None, cell_line=None) -> None:
        """Run trained model with custom drug cell_line pair.

        Args:
            training_hp: The parameters used for training models.
            drug: SMILES of a Drug.
            cell_line: ID of a Cell Line node.
        """

        def remote_infer(graph: Graph, training_hp, drug, cell_line):
            df_test = graph.query(
                f"""
                MATCH (c:CELL_LINE)
                WHERE c.id = "{cell_line}"
                RETURN "{drug}" as smiles, "{cell_line}" as id,
                0 as label, c.genomics_expression as genomics_expression,
                c.genomics_mutation as genomics_mutation, c.genomics_methylation as genomics_methylation
                ORDER BY id
                """
            ).to_pandas()
            if df_test.shape[0] == 0:
                return None

            deepcdr_test = DeepCDRDataset(df_test, utils.smiles_dict([drug]))
            test_data_loader = DeepCDRDataloader(deepcdr_test, shuffle=False)

            experiment_mgr = ExperimentManager(graph)
            experiment = experiment_mgr.get(training_hp.experiment_name, include_model=True)

            engine = PredictionEngine(experiment.model)
            pred = engine.get_predictions(test_data_loader)
            return pred[0][0].item()

        if not cell_line or not drug:
            smiles = self.graph.query(
                """
                MATCH(a:DRUG)
                RETURN a.smiles as smiles
                ORDER BY smiles
                LIMIT 20
                """
            ).head(20)["smiles"]

            cell_lines = (
                self.graph.query(
                    """
                MATCH(a:CELL_LINE)
                RETURN a.id as id, a.tcga_code as tcga_code
                ORDER BY id
                LIMIT 20
                """
                )
                .head(20)
                .agg("|".join, axis=1)
            )

            @widgets.interact(smiles=smiles, cell_line=cell_lines)
            def run(smiles, cell_line):
                cell_line = cell_line.split("|")[0]
                ic50 = self.graph.run(lambda g: remote_infer(g, training_hp, smiles, cell_line))
                print("IC50 predicted: ", ic50)
                return Chem.MolFromSmiles(smiles)

        else:
            ic50 = self.graph.run(lambda g: remote_infer(g, training_hp, drug, cell_line))
            print("IC50 predicted: ", ic50)
            return Chem.MolFromSmiles(drug)

    @utils.disable_warnings
    def infer_embeddings(self, model_hp: DeepCDRConfig) -> None:
        """Run trained model to save node embeddings

        Args:
            model_hp: The model parameters for the pipeline.
        """
        experiment_mgr = RemoteExperimentManager(self.graph)
        experiment = experiment_mgr.get("DeepCDR", include_model=False)
        model = DeepCDR(model_hp)
        model.load_state_dict(experiment.metadata["model_state"])
        model.eval()

        @torch.no_grad()
        def cell_line_feature(ccle_feat, model):
            ccle = torch.tensor(ccle_feat, dtype=torch.float32).reshape(1, -1)
            return model(ccle).squeeze().numpy()

        @torch.no_grad()
        def drug_feature(smiles):
            drug = utils.smiles_to_pyg(smiles)
            return model.drug_model(drug).squeeze().numpy()

        if model.genomics:
            print("Saving genomics embeddings on the graph")
            remote_preprocessing.generate_single_property_node_embedding(
                self.graph,
                "genomics_mutation",
                encoder=FunctionTransform(cell_line_feature, {"model": model.cell_line_model.genomics_model}),
                output_feature_name="genomics_embeddings",
                node_types=["CELL_LINE"],
            )

        if model.transcriptomics:
            print("Saving transcriptomics embeddings on the graph")
            remote_preprocessing.generate_single_property_node_embedding(
                self.graph,
                "genomics_expression",
                encoder=FunctionTransform(cell_line_feature, {"model": model.cell_line_model.transcriptomics_model}),
                output_feature_name="transcriptomics_embeddings",
                node_types=["CELL_LINE"],
            )

        if model.epigenomics:
            print("Saving epigenomics embeddings on the graph")
            remote_preprocessing.generate_single_property_node_embedding(
                self.graph,
                "genomics_methylation",
                encoder=FunctionTransform(cell_line_feature, {"model": model.cell_line_model.epigenomics_model}),
                output_feature_name="epigenomics_embeddings",
                node_types=["CELL_LINE"],
            )

        print("Saving Drugs embeddings on the graph")
        remote_preprocessing.generate_single_property_node_embedding(
            self.graph,
            "smiles",
            encoder=FunctionTransform(drug_feature),
            output_feature_name="drug_embeddings",
            node_types=["DRUG"],
        )
