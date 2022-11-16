from calendar import c
import pandas as pd
import pytz
import uuid
import argparse
import time
import sys, os
import datetime
import numpy as np
from google.cloud import bigquery
from torch_geometric.nn import SAGEConv


import katana
from katana import remote
import katana.distributed
import katana.remote.analytics
from katana.ai import data, loss, train
from katana_enterprise.distributed import Graph
from katana_enterprise.distributed import MPI

import yaml
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    table_header = config['project']['project_id']+"."+config['databases']['dataset']+"."

# needs to be refined more
class GNNEmbeddingPipeline:

    def __init__(self, display_name, optimization_prediction_type, embed_dim = 16, supervised=True):
        self.display_name = display_name
        self.optimization_prediction_type = optimization_prediction_type
        self.embed_dim = embed_dim
        self.supervised = supervised
        self.pipeline_id = uuid.uuid4().hex

    # TODO: Hyperparameter tuning based on budget_milli_node_hours
    def train(self, graph, target_property_name, graph_analytics_features, budget_milli_node_hours,
        model_display_name, disable_early_stopping, sync, split_ratio = None,):

        args = argparse.Namespace(
            feat_name="h_init",
            label_name="target",
            label_dtype=np.float32,
            split_name="split",
            distributed_execution=True,
            pos_weight=8,
            in_dim=165,
            hidden_dim=256,
            embed_dim=self.embed_dim,
            train_fan_in="100,100,100,100",
            test_fan_in="100,100,100,100",
            num_layers=4,
            out_dim=1,
            minibatch_size=1024,
            max_minibatches=20,
            lr=0.001,
            dropout=0.2,
            num_epochs=5,
            tensorboard_dir=f"gs://katana-internal1/tensorboard/elliptic_demo_e2e_{self.pipeline_id}",
            embedding_dir="gs://katana-internal1/embeddings/elliptic_demo_e2e",
            optimization_prediction_type = "node_classification"
        )

        for i in graph_analytics_features:
            if i == "pagerank":
                property_name = "pr"
                katana.remote.analytics.pagerank(graph, property_name)
            elif i == "betweenness_centrality":
                property_name="bc"
                katana.remote.analytics.betweenness_centrality(graph, property_name)

        def run_feature_init(g, split_ratio): 

            import katana.distributed
            from katana.distributed import NodeView, KeyedColumnNode
            from katana import remote
            from katana_enterprise.distributed import Graph
            from katana_enterprise.ai.preprocessing.preprocessing_graph import PreprocessingGraph
            from katana_enterprise.ai.preprocessing.split_generator import RandomSplitter
            import numpy as np
            import pyarrow as pa

            if split_ratio == "" or split_ratio is None:
                split_ratio = [0.8, 0.15, 0.05]

            def get_node_property_list(g, property_list=[]):

                lsout = []
                for property_name in property_list:
                    x = g.nodes.get_property(property_name).to_numpy().reshape(-1, 1)
                    lsout.append(x)
                feat_array = np.hstack(lsout)
                return feat_array

            local_feats_name = [f"local_feat_{i}" for i in range(2,95)]
            agg_feats_name = [f"agg_feat_{i}" for i in range(95,167)]
            prep = PreprocessingGraph(g)
            
            # extract features
            local_feats = get_node_property_list(g, property_list=local_feats_name)
            agg_feats = get_node_property_list(g, property_list=agg_feats_name)
            feat_vec = np.concatenate([local_feats, agg_feats], axis=-1)
            
            # save new features vector to graph
            prep.upsert_node_feature(feature_name="h_init", feature_data=feat_vec)
            
            # create train/test split mask
            splitter_fnc = RandomSplitter(split_ratio, random_state=42)
            split_arr = prep.generate_split_property(target_property_name="target", split_encoder=splitter_fnc)
            prep.upsert_node_feature(feature_name="split", feature_data=split_arr)

        # TODO: Add other prediction types with validation metrics and loss functions
        def run_gnn(graph, args):

            from calendar import c
            import torch
            import numpy
            import katana
            from katana_enterprise.distributed.pytorch import init_workers
            from katana_enterprise.ai.data import PygNppSubgraphSampler, SampledSubgraphConfig 
            from katana_enterprise.ai.data import NppDataLoader
            from torch.nn.parallel import DistributedDataParallel as torch_DDP
            from torch.utils.tensorboard import SummaryWriter
            import sys, os
            sys.path.append(os.path.join("/home/anuhyabs_katanagraph_com/solutions/fsi/demos/elliptic"))

            from katana.ai import data, loss, train
            from katana_enterprise.ai.torch import ReduceMethod
            from katana_enterprise.ai.train import DistTrainer
            from sklearn.metrics import roc_auc_score, f1_score
            from katana_enterprise.distributed import MPI
            from katana.distributed import Experiment, ExperimentManager
            
            # model definition
            class DistSAGE(torch.nn.Module):
                    
                def __init__(self, in_dim, hidden_dim,embed_dim, out_dim, num_layers,
                            dropout):
                    super(DistSAGE, self).__init__()

                    self.convs = torch.nn.ModuleList()
                    self.convs.append(SAGEConv(in_dim, hidden_dim))
                    self.bns = torch.nn.ModuleList()
                    self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
                    for _ in range(num_layers - 3):
                        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
                    self.convs.append(SAGEConv(hidden_dim, embed_dim))
                    self.bns.append(torch.nn.BatchNorm1d(embed_dim))
                    self.convs.append(SAGEConv(embed_dim, out_dim))
                    self.activation = torch.nn.functional.relu
                    self.dropout = torch.nn.Dropout(dropout)

                def reset_parameters(self):
                    for conv in self.convs:
                        conv.reset_parameters()
                    for bn in self.bns:
                        bn.reset_parameters()

                def encode(self, data):
                    x, edges = data.x, data.adjs

                    for i, conv in enumerate(self.convs):
                        if i != len(self.convs) - 1:
                            x_target = x[: data.dest_count[i]]
                            x = conv((x, x_target), edges[i])
                            x = self.bns[i](x)
                            x = self.activation(x)
                            x = self.dropout(x)  
                            embed = x
                    return embed

                def forward(self, data):
                    embed = self.encode(data)
                    edges = data.adjs
                    i = len(self.convs) - 1
                    x_target = embed[: data.dest_count[i]]
                    x = self.convs[-1]((embed, x_target), edges[i])
                    return x

            experiment_mgr = ExperimentManager(graph)
 
            os.environ['MODIN_ENGINE']='python'
            katana.set_active_threads(32)
            exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            main_start = time.time()
            if args.distributed_execution:
                init_workers()
            
            if args.optimization_prediction_type == "node_classification":
                loss_fn = loss.BCEWithLogitsLoss()
                validation_metric_fn = roc_auc_score
                validation_reduce_method = ReduceMethod.MEAN
            
            # initialize the multiminibatch sampler
            train_sampler = PygNppSubgraphSampler(
                graph, 
                SampledSubgraphConfig(
                layer_fan=[int(fan_in) for fan_in in args.train_fan_in.split(',')], 
                    max_minibatches=args.max_minibatches, 
                    property_batch_size=args.max_minibatches,
                    feat_prop_name=args.feat_name,
                    label_prop_name=args.label_name,
                    multilayer_export=True
                )
            )
            
            val_sampler = PygNppSubgraphSampler(
                graph, 
                SampledSubgraphConfig(
                layer_fan=[int(fan_in) for fan_in in args.train_fan_in.split(',')], 
                    max_minibatches=args.max_minibatches, 
                    property_batch_size=args.max_minibatches,
                    feat_prop_name=args.feat_name,
                    label_prop_name=args.label_name,
                    multilayer_export=True
                )
            )

            # test sampler used for evaluation; it samples 100s per hop to simulate getting
            test_sampler = PygNppSubgraphSampler(
                graph, 
                SampledSubgraphConfig(
                layer_fan=[int(fan_in) for fan_in in args.train_fan_in.split(',')], 
                    max_minibatches=args.max_minibatches, 
                    property_batch_size=args.max_minibatches,
                    feat_prop_name=args.feat_name,
                    label_prop_name=args.label_name,
                    multilayer_export=True
                )
            )
            
            train_dataloader = NppDataLoader(
                train_sampler, 
                local_batch_size=args.minibatch_size, 
                split_prop_dict={args.split_name: 0},   
                shuffle=True, 
                drop_last=True,
                balance_seeds=True)
            
            val_dataloader = NppDataLoader(
                val_sampler, 
                local_batch_size=args.minibatch_size, 
                split_prop_dict={args.split_name: 1},   
                shuffle=True, 
                drop_last=True,
                balance_seeds=True)
            
            test_dataloader = NppDataLoader(
                test_sampler, 
                local_batch_size=args.minibatch_size, 
                split_prop_dict={args.split_name: 2},  
                balance_seeds=True)
            
            model = DistSAGE(
                in_dim=args.in_dim, 
                hidden_dim=args.hidden_dim, 
                embed_dim=args.embed_dim,
                out_dim=args.out_dim, 
                num_layers=args.num_layers,
                dropout=args.dropout
            ).to(exec_device)
            
            if args.distributed_execution:
                model = torch_DDP(model)
                
            tracker = train.DistTracker(callback_fn=None, src_rank=0)
            tracker_tensor = train.DistTensorboardTracker(args.tensorboard_dir)
            # trainer configuration
            trainer = DistTrainer(
                model,
                loss_fn,
                validation_metric_fn,
                validation_reduce_method,
                train_loader=train_dataloader,
                validation_loader=val_dataloader,
                optimizer= torch.optim.Adam(model.parameters(), lr=args.lr),
                epochs=args.num_epochs,
                maximization=False,
                tracker=tracker_tensor,
            )
            # Model training
            trained_model, bce_loss = trainer.train()  
            print(f"Tensorboard log directory: {args.tensorboard_dir}") 

            # save model
            experiment = Experiment(bce_loss, trained_model.state_dict())
            experiment_mgr.upsert("ellipticExperiment", experiment)
            
            val_score, val_ypred = trainer.test(val_dataloader)
            print(f"Validation data test_score: {val_score}")

            test_score, test_ypred = trainer.test(test_dataloader)
            print(f"Test data test_score: {test_score}")

        graph.run(lambda g: run_feature_init(g, split_ratio))
        graph.run(lambda g: run_gnn(g, args))

    def infer_embeddings(self, graph, project_id):

        import torch
        import numpy as np, datetime
        import katana
        from katana_enterprise.distributed.pytorch import init_workers
        from katana_enterprise.ai.data import PygNppSubgraphSampler, SampledSubgraphConfig 
        from katana_enterprise.ai.data import NppDataLoader
        from katana_enterprise.distributed import MPI
        from torch.nn.parallel import DistributedDataParallel as torch_DDP
        import sys, os
        from katana.ai import data, loss, train
        from katana.distributed import Experiment, ExperimentManager
        from collections import OrderedDict

        args = argparse.Namespace(
                feat_name="h_init",
                label_name="target",
                label_dtype=np.float32,
                split_name="split",
                distributed_execution=True,
                pos_weight=8,
                in_dim=165,
                hidden_dim=256,
                embed_dim=self.embed_dim,
                train_fan_in="100,100,100,100",
                test_fan_in="100,100,100,100",
                num_layers=4,
                out_dim=1,
                minibatch_size=1024,
                max_minibatches=20,
                lr=0.001,
                dropout=0.2,
                num_epochs=5,
                embedding_dir="gs://katana-internal1/embeddings/elliptic_demo_e2e",
                pipeline_id = self.pipeline_id
            )

        table_id = table_header+config['databases']['embeddings']+args.pipeline_id
        schema = [bigquery.SchemaField("account_id", "string")]
        schema.extend([bigquery.SchemaField(f"embed_{i}", "float") for i in range(args.embed_dim)])
        schema.extend([bigquery.SchemaField("event_timestamp", "timestamp")])
        job_config = bigquery.LoadJobConfig(schema=schema)

        def run_inf(graph, args, table_id, job_config):
            # TODO: De-duplicate the class definition from the training loop
            class DistSAGE(torch.nn.Module):
                from torch_geometric.nn import SAGEConv
                    
                def __init__(self, in_dim, hidden_dim,embed_dim, out_dim, num_layers,
                            dropout):
                    super(DistSAGE, self).__init__()

                    self.convs = torch.nn.ModuleList()
                    self.convs.append(SAGEConv(in_dim, hidden_dim))
                    self.bns = torch.nn.ModuleList()
                    self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
                    for _ in range(num_layers - 3):
                        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
                    self.convs.append(SAGEConv(hidden_dim, embed_dim))
                    self.bns.append(torch.nn.BatchNorm1d(embed_dim))
                    self.convs.append(SAGEConv(embed_dim, out_dim))
                    self.activation = torch.nn.functional.relu
                    self.dropout = torch.nn.Dropout(dropout)

                def reset_parameters(self):
                    for conv in self.convs:
                        conv.reset_parameters()
                    for bn in self.bns:
                        bn.reset_parameters()

                def encode(self, data):
                    x, edges = data.x, data.adjs

                    for i, conv in enumerate(self.convs):
                        if i != len(self.convs) - 1:
                            x_target = x[: data.dest_count[i]]
                            x = conv((x, x_target), edges[i])
                            x = self.bns[i](x)
                            x = self.activation(x)
                            x = self.dropout(x)  
                            embed = x
                    return embed

                def forward(self, data):
                    embed = self.encode(data)
                    edges = data.adjs
                    i = len(self.convs) - 1
                    x_target = embed[: data.dest_count[i]]
                    x = self.convs[-1]((embed, x_target), edges[i])
                    return x
            
            experiment_mgr = ExperimentManager(graph)
            os.environ['MODIN_ENGINE']='python'
            katana.set_active_threads(32)
            exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            global_ids = graph.nodes.masters()
            # initialize the multiminibatch sampler
            sampler = PygNppSubgraphSampler(
                graph, 
                SampledSubgraphConfig(
                layer_fan=[int(fan_in) for fan_in in args.train_fan_in.split(',')], 
                    max_minibatches=args.max_minibatches, 
                    property_batch_size=args.max_minibatches,
                    feat_prop_name=args.feat_name,
                    label_prop_name=args.label_name,
                    multilayer_export=True
                )
            )

            dataloader = NppDataLoader(
                sampler, 
                local_batch_size=args.minibatch_size, 
                nodes=global_ids, 
                balance_seeds=True
                )

            model = DistSAGE(
                in_dim=args.in_dim, 
                hidden_dim=args.hidden_dim, 
                embed_dim=args.embed_dim,
                out_dim=args.out_dim, 
                num_layers=args.num_layers,
                dropout=args.dropout
            )

            exp = experiment_mgr.get("ellipticExperiment", include_model=True)
            checkpoint = exp.model
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v
            model.load_state_dict(new_state_dict)
            # Performing inference
            ids = graph.nodes.get_property("id")
            #ids = graph.nodes.get_property("account_id")
            inference_ids = [ids[node] for node in global_ids]

            model.eval()
            all_embed = []
            for batch in iter(dataloader):
                embed = model.encode(batch)
                all_embed.append(embed)
            all_embed = torch.vstack(all_embed).detach().numpy()

            embedding_dict = {}
            for i in range(len(inference_ids)):
                embedding_dict[inference_ids[i]] = all_embed[i]
                
            # save embedding
            col_names = ["account_id"]
            embed_names = [f"embed_{i}" for i in range(all_embed.shape[1])]
            col_names.extend(embed_names)
            out_df = pd.DataFrame(embedding_dict).transpose()
            out_df.insert(0, 'index', out_df.index)
            out_df.columns = col_names
            out_df["event_timestamp"] = pd.Timestamp(datetime.datetime.now().isoformat()).replace(hour=0, minute=0, second=0, microsecond=0)
            out_df["account_id"] = out_df["account_id"]#.astype(int)

            # write embeddings to bq
            client = bigquery.Client()
            # delete table if it exists
            try:
                client.delete_table(table_id)
                print(f"Table deleted: {table_id}")
            except:
                print(f"Table created: {table_id}")
            pass
            job = client.load_table_from_dataframe(
                out_df, table_id, job_config=job_config
            ) 
            job.result()
            table = client.get_table(table_id)  # Make an API request.
            print(
                "Loaded {} rows and {} columns to {}".format(
                    table.num_rows, len(table.schema), table_id
                )
            )     

        def feature_export(graph):
            table_id = table_header+config['databases']['account_mapping']+args.pipeline_id
            schema = [
                bigquery.SchemaField("account_gid", "integer"),
                bigquery.SchemaField("account_id", "string"),
                bigquery.SchemaField("node_type", "string"),
                bigquery.SchemaField("target", "integer"),
                bigquery.SchemaField("split", "integer"),
                bigquery.SchemaField("event_timestamp", "timestamp")
            ]    
            job_config = bigquery.LoadJobConfig(schema=schema)
            def embed_mapping(g):
                rank = MPI.COMM_WORLD.Get_rank()
                df = g.query("""
                MATCH (n)
                RETURN id(n) as account_gid, n.id as account_id, n.node_type as node_type, n.target as target, n.split as split
                """)
                df = df.to_pandas()
                df["event_timestamp"] = pd.Timestamp(datetime.datetime.now().isoformat()).replace(hour=0, minute=0, second=0, microsecond=0)
                # write embeddings to bq
                client = bigquery.Client()
                # delete table if it exists
                try:
                    client.delete_table(table_id)
                    print(f"Table deleted: {table_id}")
                except:
                    print(f"Table created: {table_id}")
                pass
                job = client.load_table_from_dataframe(
                    df, table_id, job_config=job_config
                ) 
                job.result()
                table = client.get_table(table_id)  # Make an API request.
                print(
                    "Loaded {} rows and {} columns to {}".format(
                        table.num_rows, len(table.schema), table_id
                    )
                )

            graph.run(lambda g: embed_mapping(g))    

        graph.run(lambda g: run_inf(g, args, table_id, job_config))
        feature_export(graph)  
        
        return self.pipeline_id   
