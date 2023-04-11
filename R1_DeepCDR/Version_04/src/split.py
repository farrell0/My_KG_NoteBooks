from enum import IntEnum

import pandas
from katana.remote.import_data import DataFrameImporter
from sklearn.model_selection import train_test_split


class SplitType(IntEnum):
    """SplitType encodes training, validation, and test data separation"""

    TRAIN = 0
    VAL = 1
    TEST = 2


def delete_split(graph):
    graph.query(
        """
        MATCH (:DRUG)-[u:SPLIT]->(:CELL_LINE)
        DELETE u
        """
    )


def generate_split(df, random_state, test_size):
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["tcga_code"], shuffle=True,
    )
    df_train["split"], df_test["split"] = SplitType.TRAIN, SplitType.VAL
    df_split = pandas.concat([df_train, df_test], ignore_index=True)
    df_split = df_split.astype({"drug_id": "string", "cell_line_id": "string"})
    return df_split.drop(columns=["tcga_code"])


def upsert_split(graph, split):
    with DataFrameImporter(graph) as df_importer:
        df_importer.node_id_property_name("id")
        df_importer.edges_dataframe(
            split,
            source_id_space="DRUG",
            destination_id_space="CELL_LINE",
            source_column="drug_id",
            destination_column="cell_line_id",
            type="SPLIT",
        )
        df_importer.insert()
