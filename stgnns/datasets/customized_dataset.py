import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import random

from torch_geometric.data import Dataset, Data
from torch_geometric.utils import add_remaining_self_loops

from utils.utils import get_logger

from datasets.data_preprocessing import filter_edges
from datasets.data_splitting import split_nodes_spatio, split_nodes_temporal, split_nodes_hybrid





logger = get_logger('FsimNNDataset')

class CustomGraphDataset(Dataset):
    def __init__(self, root, train_val_dirs, test_dirs, phase, split_mode="spatio", max_node_feat_len=60, max_edge_feat_len=10, transform=None, pre_transform=None):
        """
        A PyTorch Geometric Dataset that loads graphs for training, validation, and testing.

        Parameters:
        - root: str, directory where processed data will be stored
        - train_val_dir: str, directory containing raw training & validation data
        - test_dir: str, directory containing raw test data
        - phase: str, one of ["train", "val", "test"]
        - transform: callable, optional transformation function applied per data instance
        - pre_transform: callable, optional transformation function applied before saving

        """
        self.root = root
        self.train_val_dirs = train_val_dirs
        self.test_dirs = test_dirs
        self.split_mode = split_mode
        self.phase = phase
        self.max_node_feat_len = max_node_feat_len 
        self.max_edge_feat_len = max_edge_feat_len 
        self.node_id_map = {}
        self.split_file = os.path.join(self.processed_dir, "splits.pkl") # processed_dir: root/processed
        self.config_file = os.path.join(self.processed_dir, "config.pkl")
        self.train_val_data_file = os.path.join(self.processed_dir, "train_val_processed_data.pt")
        self.test_data_file = os.path.join(self.processed_dir, "test_processed_data.pt")

        super().__init__(root, transform, pre_transform)
        
        # Load or process data
        self._ensure_data_is_ready()
            
        # Load correct phase dataset
        self._load_phase_data()
        
        
        
    @property
    def raw_file_names(self):
        """Defines expected raw files in train/val and test directories."""
        return ["nodes.csv", "edges.csv", "graphs.csv"]

    @property
    def processed_file_names(self):
        """Defines expected processed file names."""
        # PyG will check if root/processed/splits.pkl, config.pkl, etc. exist, if they do, it won't call process() automatically.
        return ["splits.pkl", "config.pkl", "train_val_processed_data.pt", "test_processed_data.pt"]
    

    
    def _ensure_data_is_ready(self):
        config = self._load_config()

        if self._needs_processing(config):
            logger.info('Need to reprocess the dataset ...')
            self.process()
        else:
            self.train_val_data_list = torch.load(self.train_val_data_file, weights_only=False)
            self.test_data_list = torch.load(self.test_data_file, weights_only=False)

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "rb") as f:
                return pickle.load(f)
        return None

    def _needs_processing(self, config):
        if ( not os.path.exists(self.train_val_data_file) ) or ( not os.path.exists(self.test_data_file) ):
            return True
        if config is None:
            return True
        if config.get("max_node_feat_len") != self.max_node_feat_len or config.get("max_edge_feat_len") != self.max_edge_feat_len or config.get("split_mode") != self.split_mode:
            logger.info(f'The configuration of max_node_feat_len, max_edge_feat_len or split_mode is changed')
            return True  # Feature length settings changed, needs reprocessing
        return False  # No change, no need to reprocess 
   

    def process(self):
        logger.info('Processing Start ...')
        train_val_data_list = []
        test_data_list = []
        for dir_path in self.train_val_dirs:
            raw_dir = os.path.join(dir_path, "raw")
            train_val_data_list.extend(self._process_graphs(raw_dir))
            
        if self.train_val_dirs != self.test_dirs:
            for dir_path in self.test_dirs:
                raw_dir = os.path.join(dir_path, "raw")
                test_data_list.extend(self._process_graphs(raw_dir))
        else:
            test_data_list = train_val_data_list

        self.train_val_data_list = train_val_data_list
        self.test_data_list = test_data_list

        # Create or reuse splits for training and validation datasets
        # test dataset is directly generated based on test_dirs
        splits = self._load_or_create_splits()
        
        # Apply splits to processed data
        self._apply_splits(splits)

        # Save processed data and updated config
        torch.save(self.train_val_data_list, self.train_val_data_file)
        torch.save(self.test_data_list, self.test_data_file)
        
        # Save feature length settings
        with open(self.config_file, "wb") as f:
            pickle.dump({"split_mode": self.split_mode, "max_node_feat_len": self.max_node_feat_len, "max_edge_feat_len": self.max_edge_feat_len}, f)

        logger.info('Processing complete.')



    def _process_graphs(self, raw_dir):

        circuit = raw_dir.split('/')[2]
        logger.info(f'Process data in {circuit} ...')

        """Loads CSV files and creates PyG Data objects."""
        nodes_df = pd.read_csv(os.path.join(raw_dir, "nodes.csv"))
        graphs_df = pd.read_csv(os.path.join(raw_dir, "graphs.csv"))

        orig_edge_file = os.path.join(raw_dir, "edges.csv")
        edge_file = os.path.join(raw_dir, "processed", "filtered_edges.csv")
        filter_edges(orig_edge_file, edge_file, self.max_edge_feat_len)
        edges_df = pd.read_csv(edge_file)

        data_list = []
        for graph_id in graphs_df["graph_id"].unique():

            # Filter data for the current graph
            graph_nodes = nodes_df[nodes_df["graph_id"] == graph_id]
            graph_edges = edges_df[edges_df["graph_id"] == graph_id]

            # Extract unique node IDs and create a mapping
            node_ids = graph_nodes["node_id"].unique()
            node_num = len(node_ids)
            node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
            self.node_id_map[circuit] = node_id_map


            # Convert node features and truncate them
            node_features = np.array([list(map(float, feat.split(',')))[:self.max_node_feat_len] for feat in graph_nodes["feat"]])
            
            # Convert labels
            node_labels = graph_nodes["label"].values

            # Convert edge index
            src_nodes = graph_edges["src_id"].map(node_id_map).dropna().astype(int).tolist()
            dst_nodes = graph_edges["dst_id"].map(node_id_map).dropna().astype(int).tolist()
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

            # Convert edge features
            edge_features = np.array([list(map(float, feat.split(','))) for feat in graph_edges["feat"]])

            # Create PyG Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                node_ids=torch.tensor(node_ids, dtype=torch.long),
                edge_index=edge_index,
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                y=torch.tensor(node_labels, dtype=torch.long),
                graph_id=graph_id,
                train_mask = torch.zeros(node_num, dtype=torch.bool),
                val_mask = torch.zeros(node_num, dtype=torch.bool),
                test_mask = torch.zeros(node_num, dtype=torch.bool),
                circuit = circuit,
                source_path=raw_dir
            )

            # Add self-loops to the graph
            data.edge_index, data.edge_attr = add_remaining_self_loops(data.edge_index, edge_attr=data.edge_attr, num_nodes=data.num_nodes, fill_value=0.)

            data_list.append(data)

        return data_list

    def _load_or_create_splits(self):

        config = self._load_config()
        if config is not None:
            if config.get("split_mode") != self.split_mode:
                os.remove(self.split_file)
                logger.info(f'The split mode is changed to {self.split_mode}, delete the old split_file: {self.split_file} and generate a new one.')

        if os.path.exists(self.split_file):
            with open(self.split_file, "rb") as f:
                return pickle.load(f)
        else:
            splits = self._create_splits()
            with open(self.split_file, "wb") as f:
                pickle.dump(splits, f)
            return splits

    def _create_splits(self):
        # when train/val data and test data are from the same directory, we
        # will split data in this directory into train/val/test data. Otherwise, we only need to split
        # data in train_val_dir into train/val data.
        if self.train_val_dirs == self.test_dirs:
            split_test = True
        else:
            split_test = False
        
        data_list = self.train_val_data_list
        if self.split_mode == 'spatio':
            splits = split_nodes_spatio(data_list, use_test=split_test)
        elif self.split_mode == 'temporal':
            splits = split_nodes_temporal(data_list, use_test=split_test)
        elif self.split_mode == 'hybrid':
            splits = split_nodes_hybrid(data_list, use_test=split_test)
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")
            sys.exit()

        if split_test == False:
            splits['test'] = {}
            for data in self.test_data_list:
                splits['test'][data.graph_id] = data.node_ids[:]
        
        return splits


    def _apply_splits(self, splits):
        self._apply_splits_phase("train", self.train_val_data_list, splits)
        self._apply_splits_phase("val", self.train_val_data_list, splits)
        self._apply_splits_phase("test", self.test_data_list, splits)

    def _apply_splits_phase(self, phase, data_list, splits):
        for data in data_list:
            if data.graph_id.item() in splits[phase]:
                node_indices = np.array([self.node_id_map[data.circuit][node_id.item()] for node_id in splits[phase][data.graph_id]])
                if phase == "train":
                    data.train_mask[node_indices] = True
                elif phase == "val":
                    data.val_mask[node_indices] = True
                elif phase == "test":
                    data.test_mask[node_indices] = True
                else:
                    raise ValueError(f"Unknown phase: {phase}")


    def _load_phase_data(self):
        if self.phase == "train":
            self.dataset = [d for d in self.train_val_data_list if d.train_mask.sum() > 0]
        elif self.phase == "val":
            self.dataset = [d for d in self.train_val_data_list if d.val_mask.sum() > 0]
        elif self.phase == "test":
            self.dataset = [d for d in self.test_data_list if d.test_mask.sum() > 0]
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def len(self):
        """Returns the number of graphs in the dataset."""
        return len(self.dataset)

    def get(self, idx):
        """Returns a single graph from the dataset."""
        return self.dataset[idx]
    
    
    

    
def DisplayDatasetInfo(dataset, phase_name):
    num_graphs = len(dataset)
    
    node_counts = []
    graph_ids =[]
    for data in dataset.dataset:
        if phase_name == 'train':
            node_cnt = data.train_mask.sum().item()
        if phase_name == 'val':
            node_cnt = data.val_mask.sum().item()
        if phase_name == 'test':
            node_cnt = data.test_mask.sum().item()
        node_counts.append(node_cnt)
        graph_ids.append(data.graph_id.item())

    logger.info(f"{phase_name} Dataset:")
    logger.info(f"    Number of Graphs: {num_graphs}")
    logger.info(f"    Graph IDs are: {graph_ids}")
    logger.info(f"    Number of Nodes in Each Graph: {node_counts}")
    logger.info(f"    Total Number of Nodes: {sum(node_counts)}")
    logger.info("-" * 50)

