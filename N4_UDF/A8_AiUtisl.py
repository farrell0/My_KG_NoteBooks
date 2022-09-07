
"""
Katana Graph AI Utility Functions
- Preprocessing
- Training

"""

import time
import math
import argparse
from functools import partial
import katana
from katana import remote
import torch
import torch.nn.functional as F
import katana_enterprise
from katana.remote import import_data
import katana.distributed
from katana.local_native import TxnContext
from katana import remote
from katana.remote import import_data
from katana_enterprise.distributed import Graph
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pyarrow as pa
import os
import time, os
import argparse
import copy
# katana remote
# We need to import some distributed torch stuff to handle model synchronization and all-reduces



os.environ['MODIN_ENGINE']='python'

np.random.seed(42)

### PREPROCESSING

def get_node_property_list(g, property_list=[]):
    lsout = []
    for property_name in property_list:
        x = g.get_node_property(property_name) \
            .to_numpy() \
            .reshape(-1, 1)
        lsout.append(x)
    feat_array = np.hstack(lsout)
    return feat_array
#from gnn_preprocessing_fns import encode_features, train_test_split_mask, get_binary_feature, save_features_to_graph
def normalize_node_property(g, property_name="DTI", property_type="categorical"):
    if property_type=="numeric":
        # if numeric, use standard scaler
        x = g.get_node_property(property_name) \
                .to_numpy() \
                .reshape(-1, 1)
        standard_scaler = preprocessing.StandardScaler()
        x = standard_scaler.fit_transform(x)
        x = np.nan_to_num(x)
        return x
    else: 
        # if categorical, one hot encode
        arry = np.array(g.get_node_property(property_name).to_pandas().unique(), dtype=str)
        arry.sort()
        x = g.get_node_property(property_name).to_pandas()
        x = pd.get_dummies(x, columns=list(arry))
        x = np.nan_to_num(x.values)
        arry_out = arry[arry != 'None'].reshape(1, -1)
        arry_out = np.char.add(np.repeat(property_name+":", arry_out.shape[1]).reshape(1,-1), arry_out)
        arry_out = np.repeat(arry_out, x.shape[0], axis=0)
        return x, arry_out

def collect_node_properties(g, node_type_dict):
    values = []
    value_name = []
    if "categorical_features" in node_type_dict.keys():
        for i in node_type_dict["categorical_features"]:
            x, arry = normalize_node_property(g, i, "categorical")
            values.append(x)
            value_name.append(arry)
    if "numeric_features" in node_type_dict.keys():
        for i in node_type_dict["numeric_features"]:
            x = normalize_node_property(g, i, "numeric")
            values.append(x)
            value_name.append(np.repeat(i, x.shape[0]).reshape(-1,1))

    values = np.hstack(values)
    value_name = np.hstack(value_name)

    # convert to one dim
    value_name = np.array([' | '.join(row) for row in value_name]).reshape(-1,1)
    return values, value_name

def save_features_to_graph(original_graph, features_numpy, feature_name="h_init"):

    # save feature vector of a node
    features_numpy = np.ascontiguousarray(features_numpy)
    pa_type = pa.binary(features_numpy.dtype.itemsize * features_numpy.shape[1])
    arrow_buffer = pa.py_buffer(features_numpy.data)
    buffers = [None, arrow_buffer]
    # creates pyarrow wrapper over the numpy array
    pyarrow_array = pa.Array.from_buffers(pa_type, original_graph.num_nodes(), buffers=buffers)
    # to table
    table = pa.Table.from_arrays([pyarrow_array], [feature_name])
    # save to in-memory graph
    original_graph.upsert_node_property(table)
    return original_graph

def np_xavier_init(shape=(0,0)):
    n = shape[0]
    dim = shape[1]
    np.random.seed(0)
    scale = 1/max(1., (n+dim)/dim)
    limit = math.sqrt(3.0 * scale)
    weights = np.random.uniform(-limit, limit, size=(n,dim))
    return weights




# MMM

def encode_features(g, feature_list, feature_name="h_init", feature_dim=24, zero_pad_features=True):

    for node_type_dict in feature_list:
        if node_type_dict["feature_init_method"] == "node_properties":

                parray, value_name = collect_node_properties(g, node_type_dict)

                if parray.shape[1] > feature_dim:
                    raise RuntimeError(f"Feature size from node type {node_type_dict['node_type']} ({parray.shape[1]}) is larger than feature dim defined ({feature_dim})")
                if zero_pad_features: 
                    parray = np.pad(parray, ((0,0),(0,feature_dim-parray.shape[1])), 'constant').astype('float32')
                g = save_features_to_graph(g, parray, feature_name=f"{node_type_dict['node_type']}_{feature_name}")

    # add one xavier init feature for all nodes that can be used if no feature available

    n = g.num_nodes()
    x = np_xavier_init(shape=(n,feature_dim))
    x = np.array(x, dtype=np.float32)
    g = save_features_to_graph(g, x, feature_name=f"{feature_name}_xavier")

    # save metadata on features
    tbl = pd.DataFrame(value_name)    
    tbl.columns = ["h_init_metadata"]
    pa_tbl = pa.Table.from_pandas(tbl)
    g.upsert_node_property(pa_tbl)

    return g

# MMM




def get_binary_feature(g, feature_name="h_init", feat_dtype=np.float32):
    converted = np.frombuffer(g.get_node_property(feature_name).chunk(0).buffers()[1], feat_dtype)
    hdim = converted.shape[0] / g.num_nodes()
    converted.shape = (g.num_nodes(), int(hdim))
    return converted

def train_test_split_mask(g, train_test_validation_split=[0.5, 0.3, 0.2]):
    data = np.random.choice(  
        a=[0, 1, 2],  
        size=g.num_nodes(),  
        p=train_test_validation_split  
    ) 
    pyarrow_array = pa.array(data)
    table = pa.Table.from_arrays([pyarrow_array], ["train_test_val_mask"])
    # save to in-memory graph
    g.upsert_node_property(table)
    return g

def visualize_embeddings(g, writer, feature_name="h_init", target_name="target", filter_node_type=None, sample_size=3000):

    node_embedding = get_binary_feature(g, feature_name=feature_name)
    node_labels = g.get_node_property(target_name).to_numpy()

    if filter_node_type is not None: 
        node_filter = g.get_node_property("node_type")
        split = g.get_node_property("train_test_val_mask")
        node_set = [
            g.local_to_global_id(i)
            for i, s in enumerate(node_filter)
            if split[i].as_py() == 2 and s.as_py() == filter_node_type and i < g.num_master_nodes()
        ]
    writer.add_embedding(node_embedding[node_set][0:sample_size], 
        metadata=node_labels[node_set][0:sample_size].tolist(), 
        label_img=None, 
        global_step=1, 
        tag=feature_name, 
        metadata_header=None)

def analyze_embeddings(g, writer, feature_list, feature_name="h_init"):

    # from katana.distributed import Graph
    # from torch.utils.tensorboard import SummaryWriter
    # from katana_enterprise import distributed
    # writer = SummaryWriter("/tmp/tensorboard/fnma-embed-test-1")
    # distributed.initialize()
    # g = Graph("gs://katana-internal1/graph/BsHJ5Roapibfi1ZWC6jpr8maAFWanuKUxUsrZRWjNk1S")
    # node_type_dict = feature_list[1]

    for node_type_dict in feature_list:
        if node_type_dict["feature_init_method"] == "node_properties":
            node_type = node_type_dict['node_type']
            node_embedding = get_binary_feature(g, feature_name=f"{node_type_dict['node_type']}_{feature_name}")
            node_labels = g.get_node_property("node_label").to_numpy()

            node_embedding = node_embedding[node_labels==node_type]

            # create label for each embedding point, based on which features are included in the embedding
            pstr = ""
            for prop in node_type_dict["numeric_features"]:
                pstr += f" \" {prop}: \" + COALESCE(toString(n.{prop}),\"\") + \" | \" + "
            for prop in node_type_dict["categorical_features"]:
                pstr += f" \" {prop}: \" + COALESCE(toString(n.{prop}),\"\") + \" | \" + "            
            
            out = g.query(f"""
            
                    MATCH (n:{node_type}) 
                    RETURN (
                        " | " +
                        {pstr}
                        ""
                    ) as node_info
                    
                    """)["node_info"].to_numpy()

            writer.add_embedding(node_embedding[0:500], 
                metadata=out[0:500].tolist(), 
                label_img=None, 
                global_step=0, 
                tag=node_type, 
                metadata_header=None)

## TRAINING / TESTING



def get_split(graph, split_id, split_name="split", node_label=None, node_label_prop="node_label"):
    """
    Construct test multi minibatches to use with sampling based on the split_id and the split prop.
    This should ultimately be handled by the library itself rather than by the user.
    """
    import torch
    split = graph.get_node_property(split_name)
    if node_label is None:
        node_set = [
            graph.local_to_global_id(i)
            for i, s in enumerate(split)
            if s.as_py() == split_id and i < graph.num_master_nodes()
        ]
    else:
        node_type_list = graph.get_node_property(node_label_prop)
        node_set = [
            graph.local_to_global_id(i)
            for i, s in enumerate(split)
            if s == split_id and node_type_list[i] == node_label and i < graph.num_master_nodes()
        ]

    return node_set

def evaluate_model(model, test_dataloader, remote_execution=False):
    """
    Model evaluation on test nodes given a test dataloader
    """
    import torch
    import torch.distributed as torch_dist
    #exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    cm = torch.zeros(4)
    all_embed = []
    all_y = []
    
    for batch in iter(test_dataloader):
        #batch.to(exec_device)
        prediction, embed = model(batch)#.argmax(dim=1)
        # writer.add_text('Model Eval', f'Embed Size: {embed.shape}', 0)
        prediction = torch.sigmoid(prediction)[batch.seed_nodes]
        prediction = torch.where(prediction > .5, 1, 0)
        y = batch.y[batch.seed_nodes]
        metrics = torch.from_numpy(
            confusion_matrix(y, prediction.squeeze(1)
            ).ravel()
        )
        cm += metrics
        all_embed.append(embed[batch.seed_nodes])
        all_y.append(y)

    
    all_embed = torch.concat(all_embed, dim=0)
    all_y = torch.concat(all_y, dim=0)
    
    # writer.add_text('Model Eval', f'All embed shape: {all_embed.shape}', 0)
    if remote_execution:
        torch_dist.all_reduce(cm, op=torch_dist.ReduceOp.SUM)
    tn, fp, fn, tp = cm
    total = tn + fp + fn + tp
    correct = tn + tp 
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = correct / total
    f1_score = 2 * (precision * recall) / (precision + recall)

    #model.train()

    return acc, f1_score, precision, recall, all_embed, all_y

def train_model(model, loss_function, optimizer, writer, train_dataloader, test_dataloader, args):
    """
    Trains the GNN model given the model, the sampler (which is a dataloader), and list of
    training batches in lists
    """
    import torch
    import sys, os
    sys.path.append(os.path.join(args.katana_ai_dir))
    from katana_ai import evaluate_model
    exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_topo_time = 0
    total_pull_time = 0
    total_forward_time = 0
    total_backward_time = 0
    total_opt_time = 0

    end_to_end_start = time.time()

    saved_batch = None
    writer.add_text('Model Training', f'Begin epochs', 0)
    best_score = 0
    for epoch in range(args.num_epochs):
        multibatch_count = 0
        batch_count = 0
        correct = torch.zeros(1).to(exec_device)
        total = torch.zeros(1).to(exec_device)

        epoch_pull_time = 0
        epoch_forward_time = 0
        epoch_backward_time = 0
        epoch_opt_time = 0
        all_out = []
        all_y = []
        feat_pull_start = time.time()
        
		# both topo pulling and feat pulling are done here; not possible to time them from out here
        for batch in train_dataloader:
            feat_pull_time = time.time() - feat_pull_start
            epoch_pull_time += feat_pull_time

            optimizer.zero_grad(set_to_none=True)

            forward_start = time.time()
            out, _ = model(batch)
            out = out[batch.seed_nodes]
            y = batch.y[batch.seed_nodes].reshape(-1,1)

            saved_batch = batch
            forward_time = time.time() - forward_start
            epoch_forward_time += forward_time
            #print_flush("Forward time for epoch", epoch, "batch", batch_count, "is", forward_time)

            backward_start = time.time()
            loss = loss_function(out, y)
            loss.backward()
            backward_time = time.time() - backward_start
            epoch_backward_time += backward_time

            all_out.append(out)
            all_y.append(y)
            opt_start = time.time()
            optimizer.step()
            opt_time = time.time() - opt_start
            epoch_opt_time += opt_time

            batch_count += 1
            feat_pull_start = time.time()

        # calculate total loss
        all_y = torch.concat(all_y, dim=0)
        all_out = torch.concat(all_out, dim=0)
        total_loss = loss_function(all_out, all_y)

        total_pull_time += epoch_pull_time
        total_forward_time += epoch_forward_time
        total_backward_time += epoch_backward_time
        total_opt_time += epoch_opt_time
        
        # tensorboard logging
        writer.add_scalar("Loss/loss", total_loss.item(), epoch)
        writer.add_scalar("Performance/feature_pull", epoch_pull_time, epoch) 
        writer.add_scalar("Performance/forward_time", epoch_forward_time, epoch)  
        writer.add_scalar("Performance/backward_time", epoch_backward_time, epoch) 
        writer.add_scalar("Performance/optimization_time", epoch_opt_time, epoch) 

        test_acc, test_f1_score, test_precision, test_recall, test_embed, test_y = evaluate_model(model, test_dataloader)
        train_acc, train_f1_score, train_precision, train_recall, _, _ = evaluate_model(model, train_dataloader)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("F1/test", test_f1_score, epoch)  
        writer.add_scalar("F1/train", train_f1_score, epoch)  
        writer.add_scalar("Precision/test", test_precision, epoch)  
        writer.add_scalar("Precision/train", train_precision, epoch)  
        writer.add_scalar("Recall/test", test_recall, epoch)  
        writer.add_scalar("Recall/train", train_recall, epoch) 

        # save best model
        # if test_f1_score > best_score:
        #     best_score = test_f1_score
        #     if epoch > 4:
        torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name))    

        # add embed
        # writer.add_embedding(test_embed.detach().numpy()[0:3000], 
        #         metadata=test_y.detach().numpy()[0:3000].tolist(), 
        #         label_img=None, 
        #         global_step=epoch, 
        #         tag="trained_embed", 
        #         metadata_header=None)     

    end_to_end_end = time.time()
    writer.add_text('Model Training', f'End-to-end time: {round(end_to_end_end - end_to_end_start, 2):,}', 1)
    writer.add_text('Model Training', f'Topo + feat pull/export time: {round(total_pull_time, 2):,}', 1)
    writer.add_text('Model Training', f'Total forward time: {round(total_forward_time, 2):,}', 1)
    writer.add_text('Model Training', f'Total backward time: {round(total_backward_time, 2):,}', 1)
    writer.add_text('Model Training', f'Total optimization time: {round(total_opt_time, 2):,}', 1)

# explainability
def explain_model(model, loss_function, optimizer, dataloader, args):

    exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch in iter(dataloader):
        batch.to(exec_device)
        batch.x.requires_grad = True
        optimizer.zero_grad(set_to_none=True)
        out, _ = model(batch)
        out = out[batch.seed_nodes]
        y = batch.y[batch.seed_nodes].reshape(-1,1)
        loss = loss_function(out, y)
        loss.backward()
        return batch.x[batch.seed_nodes][0].detach().numpy(), batch.x.grad[batch.seed_nodes][0].detach().numpy(), y[0].detach().numpy(), batch
     




