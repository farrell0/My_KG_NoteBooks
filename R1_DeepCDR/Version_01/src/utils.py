"""
Utils methods for the recipe.
"""

import dataclasses

import matplotlib.pyplot as plt
import pandas
from IPython.utils.capture import capture_output
from katana.ai.hls.featurizer import CanonicalAtomFeaturizer
from katana.ai.hls.smiles_to_graph import smiles2PyG
from src.split import SplitType

TCGA_label_set = {
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
}


def compare_baselines(result, index="DeepCDR_Katana"):
    result["Use GNN"] = "\u2714"

    ridge_baseline = {"pearson": 0.780, "rmse": 2.368, "spearman": 0.731, "Use GNN": "\u274C"}
    rf_baseline = {"pearson": 0.809, "rmse": 2.270, "spearman": 0.767, "Use GNN": "\u274C"}
    moli_baseline = {"pearson": 0.813, "rmse": 2.282, "spearman": 0.782, "Use GNN": "\u274C"}
    cdr_baseline = {"pearson": 0.871, "rmse": 1.982, "spearman": 0.852, "Use GNN": "\u274C"}
    tcnn_baseline = {"pearson": 0.885, "rmse": 1.782, "spearman": 0.862, "Use GNN": "\u274C"}
    indexes = [
        "Ridge Regression",
        "Random Forest",
        "MOLI",
        "CDRscan",
        "tCNNs",
        index,
    ]
    baselines = [
        ridge_baseline,
        rf_baseline,
        moli_baseline,
        cdr_baseline,
        tcnn_baseline,
        result,
    ]

    df = pandas.DataFrame(baselines, index=indexes)[["pearson", "spearman", "rmse"]]
    df = df.rename(columns={"pearson": "pearson (\u2B06)", "rmse": "rmse (\u2B07)", "spearman": "spearman (\u2B06)"})
    return df.style.apply(highlight_df)


def get_node_number(graph, node_type):
    return graph.query(
        f"""
        MATCH (a:{node_type})
        RETURN COUNT(a) as n
        """
    ).head()["n"][0]


def smiles_to_pyg(smiles):
    return smiles2PyG(smiles, CanonicalAtomFeaturizer())


def disable_warnings(func):
    def wrapper(*args, **kwargs):
        with capture_output(False, True, False):
            return func(*args, **kwargs)

    return wrapper


def dataclass_repr(dataclass):
    s = ",\n    ".join(f"{field.name} = {getattr(dataclass, field.name)}" for field in dataclasses.fields(dataclass))
    s = f"{type(dataclass).__name__}:\n    {s}"
    return s


def highlight_df(x):
    if "rmse" in x.name:
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


def smiles_dict(smiles_list):
    smiles_dict = {}
    for smiles in smiles_list:
        smiles_dict[smiles] = smiles_to_pyg(smiles)
    return smiles_dict


def pairs_query_str(graph, split):
    if split == SplitType.TEST:
        df_test = graph.query(
            f"""
            MATCH (a:DRUG)-[u:SPLIT]->(c:CELL_LINE)
            WHERE u.split = {SplitType.VAL}
            RETURN a.smiles as smiles, c.id as id, u.label as label,
            c.genomics_expression as genomics_expression,
            c.genomics_mutation as genomics_mutation,
            c.genomics_methylation as genomics_methylation
            ORDER BY smiles, id, label
            """,
            balance_output=False,
        ).to_pandas()
        return df_test

    df_pairs = graph.query(
        f"""
        MATCH (a:DRUG)-[u:SPLIT]->(c:CELL_LINE)
        WHERE u.split = {split}
        RETURN a.smiles as smiles, u.label as label,
        c.id as id
        """,
        balance_output=True,
    ).to_pandas()

    df_features = graph.query(
        """
        MATCH (c:CELL_LINE)
        RETURN c.id as id,
        c.genomics_expression as genomics_expression,
        c.genomics_mutation as genomics_mutation,
        c.genomics_methylation as genomics_methylation
        ORDER BY id
        """
    ).to_pandas()
    
    from katana.distributed import network
    df_features = network.broadcast(0, df_features)
    return pandas.merge(df_pairs, df_features, left_on="id", right_on="id").drop(["id"], axis=1)
