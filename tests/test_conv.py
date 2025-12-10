import pytest
import torch
from torch_geometric.nn import MessagePassing

# Import your layer
# Make sure you have installed your package with `pip install -e .` 
# or add `import sys; sys.path.append("..")` if running locally without install.
from gnca.conv import GNCAConv

def test_gnca_conv_basic():
    """
    Verifies the layer works on a single graph with no edge attributes.
    """
    in_channels, out_channels = 16, 32
    # Create a small graph: 4 nodes, 6 edges
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    num_nodes = 4
    x = torch.randn((num_nodes, in_channels))

    # Initialize layer
    conv = GNCAConv(in_channels, out_channels)
    
    # Check string representation
    assert str(conv) == 'GNCAConv(16, 32, edge_dim=0)'

    # Forward pass
    out = conv(x, edge_index)
    
    # Check output shape: [N, in_channels + out_channels]
    # Because the paper concatenates input with message
    assert out.size() == (num_nodes, in_channels + out_channels)


def test_gnca_conv_with_edge_features():
    """
    Verifies the layer handles edge attributes correctly when edge_dim > 0.
    """
    in_channels, out_channels = 16, 32
    edge_dim = 4
    
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    num_nodes = 2
    num_edges = edge_index.size(1)
    
    x = torch.randn((num_nodes, in_channels))
    edge_attr = torch.randn((num_edges, edge_dim))

    conv = GNCAConv(in_channels, out_channels, edge_dim=edge_dim)
    
    assert str(conv) == f'GNCAConv(16, 32, edge_dim={edge_dim})'

    out = conv(x, edge_index, edge_attr=edge_attr)
    assert out.size() == (num_nodes, in_channels + out_channels)


def test_gnca_conv_jit_script():
    """
    Verifies the layer is compatible with TorchScript (JIT).
    This is often required for high-performance production environments.
    """
    in_channels, out_channels = 8, 16
    conv = GNCAConv(in_channels, out_channels)
    
    # Attempt to script the model
    jit_conv = torch.jit.script(conv)
    
    x = torch.randn((4, in_channels))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    out = jit_conv(x, edge_index)
    assert out.size() == (4, in_channels + out_channels)


def test_gnca_conv_static_graph():
    """
    Ensures correct behavior when used inside a wrapper model 
    (mimicking the GNCAModel architecture).
    """
    in_channels, out_channels = 10, 10
    conv = GNCAConv(in_channels, out_channels)
    
    x = torch.randn((5, in_channels))
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

    # Simulate 5 steps of cellular automata evolution
    # The output of step T becomes input of step T+1
    # Note: GNCAConv output is (in + out), so we would usually need a projection 
    # back to `in_channels` size here, but this tests raw stability.
    
    for _ in range(5):
        # We slice to keep dimensions matching for this simple test
        # In the real model, an MLP handles this size change.
        out = conv(x, edge_index)
        x = out[:, :in_channels] # Mocking the update
        
    assert x.size() == (5, in_channels)