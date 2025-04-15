import random
from typing import List, Dict
import torch

def split_nodes_spatio(data_list: List[Dict], train_ratio=0.6, val_ratio=0.2, use_test=True):
    if not use_test:
        # Normalize ratios so train + val = 1.0
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total

    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    for data in data_list:
        graph_id = data.graph_id
        # Here we can't use random.shuffle(nodes) because random.shuffle() only works on Python lists, not on PyTorch tensors. 
        # It may silently corrupt the data or do nothing meaningful.
        nodes = data.node_ids.clone()
        indices = torch.randperm(nodes.size(0))
        nodes = nodes[indices]
        n = len(nodes)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        # Each call to tensor() creates a new tensor object, and even if the values are the same, 
        # they are not the same key in Python dictionaries unless they are the same object or 
        # Python sees them as equal and hash-compatible. Therefore, here we convert graph_id.
        graph_id = int(graph_id)
        split['train'][graph_id] = nodes[:train_end]
        split['val'][graph_id] = nodes[train_end:val_end]

        if use_test:
            split['test'][graph_id] = nodes[val_end:]

    return split


def split_nodes_temporal(data_list: List[Dict], train_ratio=0.6, val_ratio=0.2, use_test=True):
    if not use_test:
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total

    # creates a shallow copy of data_list, so when random.shuffle(graph_list) is called, 
    # it does not change the order of data_list
    graph_list = data_list[:]
    random.shuffle(graph_list)
    total_graphs = len(graph_list)

    train_end = int(train_ratio * total_graphs)
    val_end = int((train_ratio + val_ratio) * total_graphs)

    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    for i, data in enumerate(graph_list):
        graph_id = data.graph_id
        nodes = data.node_ids

        graph_id = int(graph_id)
        if i < train_end:
            split['train'][graph_id] = nodes
        elif i < val_end:
            split['val'][graph_id] = nodes
        elif use_test:
            split['test'][graph_id] = nodes

    return split


def split_nodes_hybrid(data_list: List[Dict], train_ratio=0.6, val_ratio=0.2, use_test=True):
    if not use_test:
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total

    split = {'train': {}, 'val': {}}
    if use_test:
        split['test'] = {}

    graph_list = data_list[:]
    random.shuffle(graph_list)

    for data in graph_list:
        graph_id = data.graph_id
        # Here we can't use random.shuffle(nodes) because random.shuffle() only works on Python lists, not on PyTorch tensors. 
        # It may silently corrupt the data or do nothing meaningful.
        nodes = data.node_ids.clone()
        indices = torch.randperm(nodes.size(0))
        nodes = nodes[indices]

        # Decide mode per graph
        mode = random.choice(['temporal', 'spatio'])

        if mode == 'temporal':
            choices = ['train', 'val']
            weights = [train_ratio, val_ratio]
            if use_test:
                choices.append('test')
                weights.append(1 - train_ratio - val_ratio)
            group = random.choices(choices, weights=weights)[0]
            split[group][graph_id] = nodes

        else:  # spatio mode
            n = len(nodes)
            train_end = int(train_ratio * n)
            val_end = int((train_ratio + val_ratio) * n)
            
            graph_id = int(graph_id)
            split['train'][graph_id] = nodes[:train_end]
            split['val'][graph_id] = nodes[train_end:val_end]

            if use_test:
                split['test'][graph_id] = nodes[val_end:]

    return split
