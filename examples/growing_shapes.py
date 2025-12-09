import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pygsp.graphs import Bunny
import numpy as np
import os


from gnca import GNCAModel, SamplePool

def get_bunny_data():
    """Downloads and prepares the Stanford Bunny graph via PyGSP."""
    print("Loading Stanford Bunny from PyGSP...")
    # 1. Load Mesh
    graph = Bunny()
    
    # 2. Create PyG Data object
    # The target state is the 3D coordinates
    pos = torch.tensor(graph.coords, dtype=torch.float)
    
    # Create edge_index from the adjacency matrix
    rows, cols = graph.W.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
    return pos, edge_index

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters from Appendix A.5
    LR = 0.001
    BATCH_SIZE = 8
    POOL_SIZE = 1024
    N_EPOCHS = 2000 # Paper uses convergence patience, simplified here
    
    # 1. Setup Data
    target_pos, edge_index = get_bunny_data()
    target_pos = target_pos.to(device)
    edge_index = edge_index.to(device)
    
    # Normalization: Paper suggests normalizing target 
    # We center and scale to fit roughly in [-1, 1]
    target_mean = target_pos.mean(dim=0)
    target_scale = target_pos.abs().max()
    target_normalized = (target_pos - target_mean) / target_scale
    
    # Initial State (Seed): "Normalisation of the target"
    # Often in NCA, we might start from a small seed (e.g., zeros + center point), 
    # but the paper says "initial state a normalisation of the target" divided by norm.
    # Let's use a "noisy" version of the target or a contracted version as the seed 
    # to give it something to "grow" to.
    # SIMPLIFICATION: The paper implies reconstructing coordinates. 
    # We will define the "Seed" as the target scaled down to near-zero (a small dot).
    seed_state = target_normalized * 0.01 

    # 2. Setup Model
    # Input/Output channels = 3 (x, y, z coordinates)
    model = GNCAModel(input_channels=3, output_channels=3, output_activation='tanh').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 3. Setup Sample Pool (The Cache)
    # "Cache has a size of 1024 states and is initialised entirely with S" (cite: 361)
    # Here S refers to the seed state.
    pool = SamplePool(POOL_SIZE, seed_state)
    
    print("Starting training...")
    
    for epoch in range(N_EPOCHS):
        # Sample batch from pool (cite: 359)
        batch_idx, current_states = pool.sample(BATCH_SIZE)
        current_states = current_states.to(device)
        
        # "One element of the cache is replaced with S (seed) to avoid catastrophic forgetting" (cite: 360)
        current_states[0] = seed_state.clone()
        
        # Randomize timesteps t in [10, 20] (cite: 365)
        steps = torch.randint(10, 21, (1,)).item()
        
        # Forward pass (Time Evolution)
        x = current_states
        for _ in range(steps):
            # Evolve: x_new = x_old + delta (usually handled inside model or here)
            # The model output is often treated as a velocity 'delta' in continuous tasks
            delta = model(x, edge_index)
            x = x + delta
            
            # Clip if using tanh constraints (implied by bounds)
            x = torch.clamp(x, -1.0, 1.0)
            
        # Loss: MSE between evolved state and Target (cite: 357)
        # Note: We compare against the batch of Targets (broadcasted)
        loss = F.mse_loss(x, target_normalized.unsqueeze(0).expand(BATCH_SIZE, -1, -1))
        
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping "to stabilise training" (cite: 551)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update Pool with new states (cite: 360)
        pool.commit(batch_idx, x)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/bunny_gnca.pt')
    print("Model saved to checkpoints/bunny_gnca.pt")

if __name__ == '__main__':
    train()