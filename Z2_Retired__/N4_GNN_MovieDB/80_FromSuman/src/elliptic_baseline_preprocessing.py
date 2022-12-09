from google.cloud import aiplatform
from google.cloud import bigquery
import datetime

import yaml
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    table_header = config['project']['project_id']+"."+config['databases']['dataset']+"."

def  create_table(pipeline_uri, type):
    
    #Initialize BigQuery Client
    client = bigquery.Client()
    
    mapping_table = table_header+config['databases']['account_mapping']+pipeline_uri
    features = table_header+config['databases']['account_features']
    embeddings = table_header+config['databases']['embeddings']+pipeline_uri
    
    sql_baseline = f"""
SELECT af.*, am.split, am.target
FROM 
    {mapping_table} am
JOIN {features} af
on am.account_id = af.account_id
"""
    
    sql_enhanced = f"""
SELECT af.*, am.split, am.target, em.* 
FROM 
    {mapping_table} am
JOIN {features} af
on am.account_id = af.account_id
join {embeddings} em 
on am.account_id = em.account_id
"""
    
    #create table for baseline model
    if type == "baseline":
        table_id = table_header+config['databases']['ml_baseline']+pipeline_uri
        display_name = "elliptic_baseline_training_dataset"
        bq_source = "bq://katana-clusters-beta.fsi_elliptic.auto_ml_baseline_training"
        sql = sql_baseline
        cols = ["split", "account_id", "event_timestamp"]
    else:
        table_id = table_header+config['databases']['ml_enhanced']+pipeline_uri
        display_name = "elliptic_enhanced_training_dataset"
        bq_source = "bq://katana-clusters-beta.fsi_elliptic.auto_ml_enhanced_gnn_training"
        sql = sql_enhanced
        cols=["split", "account_id", "event_timestamp", "event_timestamp_1"]

    df = client.query(sql).to_dataframe()
    
    # set train / test split column
    df['split_str'] = df['split'].map({0: "TRAIN", 1:"VALIDATE", 2: "TEST"})
    # drop columns
    df = df.drop(columns= cols)

    # set schema
    schema = [bigquery.SchemaField("target", "integer")]
    schema.extend([bigquery.SchemaField(f"local_feat_{i}", "float") for i in range(2,95)])
    schema.extend([bigquery.SchemaField("split_str", "string")])
    
    # load into bq
    # delete table if it exists
    try:
        client.delete_table(table_id)
        print(f"Table deleted: {table_id}")
    except:
        print(f"Table created: {table_id}")
        pass
    job_config = bigquery.LoadJobConfig(schema=schema)
    job = client.load_table_from_dataframe(
    df, table_id, job_config=job_config
    ) 
    job.result()
    table = client.get_table(table_id) # Make an API request.
    print(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
        )
    )
    dataset = aiplatform.TabularDataset.create(
    display_name = display_name,
    bq_source = bq_source,
    )

    dataset.wait()

    print(f'\tDataset: "{dataset.display_name}"')
    print(f'\tname: "{dataset.resource_name}"')

    return dataset
   