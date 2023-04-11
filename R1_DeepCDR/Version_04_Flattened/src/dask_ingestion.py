"""
Dask ingestion function to generate the graph.
"""

import dask.dataframe as dd
from katana.remote.import_data import DataFrameImporter
from src import utils


def nodes_dataframe(input_config):
    """Generate the nodes dataframes"""
    node_dd = {}

    node_type = "CELL_LINE"
    node_dd[node_type] = dd.read_csv(input_config.cell_lines_path, sep=",", dtype={"id": "string"})
    node_dd[node_type] = node_dd[node_type][node_dd[node_type]["tcga_code"].isin(utils.TCGA_label_set)]

    node_type = "GDSC"
    node_dd[node_type] = dd.read_csv(input_config.gdsc_path, sep=",", dtype={"id": "string"})

    node_type = "GENE"
    node_dd[node_type] = dd.read_csv(input_config.genes_path, sep=",", dtype={"id": "string"})

    node_type = "DRUG"
    node_dd[node_type] = dd.read_csv(input_config.drugs_path, sep=",", dtype={"id": "string"})
    return node_dd


def edges_dataframe(input_config):
    """Generate the edges dataframes"""
    edge_dd = {}
    edges = {
        input_config.gdsc_cell_line_path: ("HAS_CELL_LINE", "GDSC", "CELL_LINE"),
        input_config.gdsc_drug_path: ("FOR_DRUG", "GDSC", "DRUG"),
        input_config.cell_line_gene_expression_path: ("HAS_EXPRESSION_OBSERVATION", "CELL_LINE", "GENE"),
        input_config.cell_line_gene_methylation_path: ("HAS_METHYLATION_OBSERVATION", "CELL_LINE", "GENE"),
        input_config.cell_line_gene_mutation_path: ("HAS_MUTATION_OBSERVATION", "CELL_LINE", "GENE"),
    }

    for filename, edge_type in edges.items():
        edge_dd[edge_type] = dd.read_csv(filename, sep=",", dtype={"START_ID": "string", "END_ID": "string"})
        edge_dd[edge_type] = edge_dd[edge_type].dropna()
    return edge_dd


def upsert_dataframe(graph, node_dd, edge_dd):
    """Upsert nodes and edges into the graph"""
    print("Importing graph from dataframe files into graph...")
    with DataFrameImporter(graph) as df_importer:
        for node_type, dd in node_dd.items():
            df_importer.nodes_dataframe(dd, id_column="id", id_space=node_type)
        for tup, dd in edge_dd.items():
            source_col = "START_ID"
            destination_col = "END_ID"
            df_importer.edges_dataframe(
                dd,
                source_id_space=tup[1],
                destination_id_space=tup[2],
                source_column=source_col,
                destination_column=destination_col,
                type=tup[0],
            )
        df_importer.upsert()


def generate_deepcdr_graph(graph, input_dir_path):
    """Generate DeepCDR graph from csv files"""
    print("***Loading nodes dataframe***")
    nodes_dd = nodes_dataframe(input_dir_path)
    print("***Loading edges dataframe***")
    edges_dd = edges_dataframe(input_dir_path)
    print("***Upserting nodes and edges into the graph***")
    upsert_dataframe(graph, nodes_dd, edges_dd)
