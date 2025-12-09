import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pygsp.graphs import Bunny
import numpy as np

from gnca import GNCAModel

def get_bunny_edges():
    graph = Bunny()
    rows, cols = graph.W.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    pos = torch.tensor(graph.coords, dtype=torch.float)
    return pos, edge_index

def visualize():
    # Setup
    device = torch.device('cpu') # Visualization is fast enough on CPU
    pos, edge_index = get_bunny_edges()
    
    # Normalize target exactly as in training
    target_mean = pos.mean(dim=0)
    target_scale = pos.abs().max()
    
    # Initial seed (collapsed state)
    current_state = ((pos - target_mean) / target_scale) * 0.01
    
    # Load Model
    model = GNCAModel(input_channels=3, output_channels=3, output_activation='tanh')
    model.load_state_dict(torch.load('checkpoints/bunny_gnca.pt', map_location=device))
    model.eval()
    
    # Setup Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("GNCA Bunny Convergence")
    
    # Storage for animation frames
    frames = []
    
    # Simulation Loop
    steps = 60
    with torch.no_grad():
        for i in range(steps):
            # Visualization update
            ax.clear()
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_title(f"Step {i}")
            
            # Plot current point cloud
            np_state = current_state.numpy()
            scat = ax.scatter(np_state[:, 0], np_state[:, 1], np_state[:, 2], 
                              c='purple', s=5, alpha=0.6)
            
            # Capture frame for GIF
            # Note: Matplotlib animation requires a function update, 
            # but simpler approach for scripts is usually to save images or use FuncAnimation.
            # We will use FuncAnimation logic below instead of this loop structure 
            # if we want to display live, but here is the logic for the update.
            
            # Evolution Step
            delta = model(current_state, edge_index)
            current_state = current_state + delta
            current_state = torch.clamp(current_state, -1.0, 1.0)

    # To actually save the GIF, we use FuncAnimation
    def update(frame):
        nonlocal current_state
        ax.clear()
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_title(f"GNCA Growth - Step {frame}")
        
        np_state = current_state.numpy()
        ax.scatter(np_state[:, 0], np_state[:, 1], np_state[:, 2], 
                   c=np_state[:, 2], cmap='viridis', s=10) # Color by Z-height
        
        # Evolve
        delta = model(current_state, edge_index)
        current_state = current_state + delta
        current_state = torch.clamp(current_state, -1.0, 1.0)
        
        return ax,

    # Reset state for animation
    current_state = ((pos - target_mean) / target_scale) * 0.01
    
    ani = animation.FuncAnimation(fig, update, frames=60, interval=100)
    ani.save('bunny_growth.gif', writer='pillow', fps=10)
    print("Saved bunny_growth.gif")

if __name__ == '__main__':
    visualize()