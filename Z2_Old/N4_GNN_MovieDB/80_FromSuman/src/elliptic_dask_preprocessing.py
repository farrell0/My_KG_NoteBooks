
# Elliptic Preprocessing Dataset
import os                                                                        
import time                                                                      
import json
import uuid
import pandas as pd
import numpy
import argparse
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')
from google.cloud import bigquery
import datetime

from katana import remote
from katana.remote import import_data
from katana.remote.import_data import Operation

import dask.dataframe as dd

import yaml



with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def feature_export(nodes, local_cols):
    """
    Exports the original/raw features from the elliptic dataset to BQ for 
    building the baseline model
    """

    table_id = config['project']['project_id']+"."+config['databases']['dataset']+"."+config['databases']['account_features']
    # set schema
    schema = [bigquery.SchemaField("account_id", "string")]
    schema.extend([bigquery.SchemaField(f"local_feat_{i}", "float") for i in range(2,95)])
    schema.extend([bigquery.SchemaField("event_timestamp", "timestamp")])
    cols = ["account_id"]
    cols.extend(local_cols)
    cols.extend(["event_timestamp"])
    nodes["account_id"] = nodes["account_id"].astype(str) 
    nodes["event_timestamp"] = pd.Timestamp(datetime.datetime.now().isoformat()).replace(hour=0, minute=0, second=0, microsecond=0)
    # load into bq
    client = bigquery.Client()
    # delete table if it exists
    try:
        client.delete_table(table_id)
        print(f"Table deleted: {table_id}")
    except:
        print(f"Table created: {table_id}")
    pass
    job_config = bigquery.LoadJobConfig(schema=schema)
    job = client.load_table_from_dataframe(
        nodes[cols].compute(), table_id, job_config=job_config
    ) 
    job.result()
    table = client.get_table(table_id)  # Make an API request.
    print(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
        )
    )
    print("Updated elliptic_feast/feature_repo/elliptic_features.py with baseline feature view")
    

def elliptic_preprocessing():
    """
    Importing and preprocessing of the dataset.

    Returns a graph.
    """

    # Import datasets here
    tx_classes = "gs://katana-demo-datasets/fsi/solution_raw_data/elliptic/elliptic_txs_classes.csv"
    tx_edges = "gs://katana-demo-datasets/fsi/solution_raw_data/elliptic/elliptic_txs_edgelist.csv"
    tx_features = "gs://katana-demo-datasets/fsi/solution_raw_data/elliptic/elliptic_txs_features.csv"

    # Creating variables to store datasets
    feat_col_names = ["account_id", "timestamp"]
    local_feats_name = [f"local_feat_{i}" for i in range(2,95)]
    agg_feats_name = [f"agg_feat_{i}" for i in range(95,167)]
    feat_col_names.extend(local_feats_name)
    feat_col_names.extend(agg_feats_name)

    feat_types = {
        "class": "string",
        "timestamp": "string", 
        "target": "float",
        "node_type": "string"
    }
    local_cols = {}
    for i in range(2,95):
        local_cols[f"local_feat_{i}"] = "float"
    agg_cols = {}
    for i in range(95,167):
        agg_cols[f"agg_feat_{i}"] = "float"
    feat_types.update(local_cols)
    feat_types.update(agg_cols)

    # Data stored in dask dataframes here
    classes = dd.read_csv(tx_classes)
    edges = dd.read_csv(tx_edges)
    features = dd.read_csv(tx_features, header=None, names=feat_col_names)

    # Clean and customize dataset here
    classes['target'] = classes['class'].map({'unknown': 0.0, '1': 1.0, '2': 0.0})
    classes['node_type'] = classes['class'].map({'unknown': 'Unclassified_Acct', '1': 'Classified_Acct', '2': 'Classified_Acct'})       

    classes = classes.rename(columns={"txId": "account_id"})  
    edges = edges.rename(columns={"txId1": "account_id_src", "txId2": "account_id_dst"})
    nodes = features.merge(classes)

    # export features to BQ and register in feature store
    feature_export(nodes, local_cols)

    # Graph creation with 4 partitions
    graph = remote.Client(disable_version_check=False).create_graph(
        num_partitions=4
    )

    reverse_edges=True
    with import_data.DataFrameImporter(graph) as df_importer:   
        
        df_importer.nodes_dataframe(nodes,
                                id_column="account_id",
                                id_space="account", 
                                property_columns=feat_types,
                                label_column="node_type")
        
        df_importer.edges_dataframe(edges,
                                source_id_space="account",
                                destination_id_space="account",
                                source_column="account_id_src",
                                destination_column="account_id_dst",
                                type="transaction")
        if reverse_edges:
            df_importer.edges_dataframe(edges,
                            source_id_space="account",
                            destination_id_space="account",
                            source_column="account_id_dst",
                            destination_column="account_id_src",
                            type="rev_transaction")
        #df_importer.insert()
    return graph