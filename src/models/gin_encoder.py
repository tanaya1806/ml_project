"""
GIN Encoder for Milestone 3.
Converts temporal subgraph structure into fixed-dimensional representations.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from typing import Optional

from src.graphs.subgraph_extraction import CandidateSubgraph


class WeightedGINConv(MessagePassing):
    """
    A variant of Graph Isomorphism Network (GIN) convolution that natively 
    supports signed scalar edge weights via MessagePassing.
    
    Formula:
        h_v^(k) = MLP_k( (1 + eps) * h_v^(k-1) + \\sum_{u \\in N(v)} w_uv h_u )
    """
    def __init__(self, nn_module: nn.Module, eps: float = 0., train_eps: bool = False):
        super().__init__(aggr='add')
        self.nn = nn_module
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity of shape [2, num_edges]
            edge_weight: Edge weights of shape [num_edges]
        """
        # Propagate messages. We pass edge_weight so it is available in message()
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        # Apply the (1 + eps) * x + aggregated_messages
        out = (1 + self.eps) * x + out
        
        # Pass through the MLP
        return self.nn(out)

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Constructs messages.
        Args:
            x_j: Features of source nodes of shape [num_edges, in_channels]
            edge_weight: Edge weights of shape [num_edges]
        """
        if edge_weight is not None:
            # Multiply source node features by scalar edge weight
            return x_j * edge_weight.unsqueeze(-1)
        return x_j


class GINEncoder(nn.Module):
    """
    GIN-based spatial representation for brain subgraphs.
    Input: [B subgraphs] -> Output: [B, 128] representations
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, embedding_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        
        # Layer 1
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(WeightedGINConv(mlp1, train_eps=True))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(WeightedGINConv(mlp, train_eps=True))
            
        self.relu = nn.ReLU()
        self.proj = nn.Linear(hidden_channels, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Edges [2, E]
            edge_weight: Signed weights [E]
            batch: Batch vector [N] mapping nodes to graphs
        Returns:
            z: Graph embeddings [B, embedding_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            # ReLU applies after all but the last layer usually in GIN, 
            # but user specifically asked for ReLU after each GIN layer
            h = self.relu(h)
                
        # Global mean pooling
        h_pool = global_mean_pool(h, batch)
        
        # Linear projection
        out = self.proj(h_pool)
        return out


def subgraph_to_pyg(subgraph: CandidateSubgraph) -> Data:
    """
    Converts a CandidateSubgraph from M2 into a PyTorch Geometric Data object.
    Uses x_i = 1.0 for all nodes to focus on structural information.
    """
    # Create constant node features [num_nodes, 1]
    x = torch.ones((subgraph.num_nodes, 1), dtype=torch.float32)
    
    # Edges and weights
    edge_index = torch.tensor(subgraph.edge_index, dtype=torch.long)
    edge_weight = torch.tensor(subgraph.edge_weight, dtype=torch.float32)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    
    # Store metadata
    data.subject_id = subgraph.subject_id
    data.seed_roi = subgraph.seed_roi
    data.window_idx = subgraph.window_idx
    data.k_hop = subgraph.k_hop
    data.roi_indices = subgraph.roi_indices
    data.seed_local_idx = subgraph.seed_local_idx
    
    return data
