import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing

class GNCAConv(MessagePassing):
    r"""The Graph Neural Cellular Automata convolution layer from the
    `"Learning Graph Cellular Automata" <https://arxiv.org/abs/2110.14237>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \, \| \, \sum_{j \in \mathcal{N}(i)}
        \mathrm{ReLU}(\mathbf{W} \mathbf{x}_j + \mathbf{b})

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample (message dimension).
        aggr (string, optional): The aggregation scheme to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"add"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 aggr: str = 'add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=True)
        
        self.act = ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # 1. Propagate messages. 
        out = self.propagate(edge_index, x=x)

        # 2. Concatenate with self (Eq 4: h_i || sum(...))
        return torch.cat([x, out], dim=-1)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        
        # Apply the linear transformation (W * h_j + b)
        msg = self.lin(x_j)
        
        # Apply the activation (ReLU) BEFORE aggregation
        return self.act(msg)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'