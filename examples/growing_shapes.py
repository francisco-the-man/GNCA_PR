import torch
import torch.nn.functional as F
import numpy as np
import os
from pygsp.graphs import Bunny

# Import your local package
from gnca import GNCAModel, SamplePool

def get_bunny_data():
    """Downloads and prepares the Stanford Bunny graph via PyGSP."""
    print("Loading Stanford Bunny from PyGSP...")
    graph = Bunny()
    
    # 1. Get Coordinates
    pos = torch.tensor(graph.coords, dtype=torch.float)
    
    # 2. Get Edge Index
    rows, cols = graph.W.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
    return pos, edge_index

def train():
    # Detect device (with MPS support for Mac, CUDA for Colab)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Hyperparameters
    LR = 0.001
    BATCH_SIZE = 8       # Increased back to 8 (Safe now with the fix)
    POOL_SIZE = 1024
    N_EPOCHS = 2000

    # 1. Setup Data
    target_pos, edge_index = get_bunny_data()
    target_pos = target_pos.to(device)
    edge_index = edge_index.to(device)
    
    # Calculate dims
    num_nodes = target_pos.size(0)
    
    # --- CRITICAL FIX: Pre-compute Batched Edge Index ---
    # We repeat the edge_index for each item in the batch, offsetting node indices.
    # This treats the batch as one giant graph with disjoint components.
    edge_indices_list = []
    for i in range(BATCH_SIZE):
        # Shift the node indices by (i * num_nodes)
        edge_indices_list.append(edge_index + i * num_nodes)
    
    # Combine into one big edge list: Shape [2, Batch * Edges]
    batched_edge_index = torch.cat(edge_indices_list, dim=1).to(device)
    # ----------------------------------------------------

    # Normalization
    target_mean = target_pos.mean(dim=0)
    target_scale = target_pos.abs().max()
    target_normalized = (target_pos - target_mean) / target_scale
    
    # Initial Seed (Tiny dot at center)
    seed_state = target_normalized * 0.01 

    # 2. Setup Model
    model = GNCAModel(input_channels=3, output_channels=3, output_activation='tanh').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 3. Setup Pool
    pool = SamplePool(POOL_SIZE, seed_state)
    
    print(f"Starting training on {device}...")
    
    for epoch in range(N_EPOCHS):
        # Sample batch
        batch_idx, current_states = pool.sample(BATCH_SIZE)
        current_states = current_states.to(device)
        
        # Replace first item with seed to prevent catastrophic forgetting
        current_states