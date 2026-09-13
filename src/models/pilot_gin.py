"""
Pilot Experiment for Milestone 3 (GIN Encoder).
Runs a small sample of controls and ASD subjects through the GIN encoder 
to demonstrate structural embedding functionality.
"""

import os
import glob
import time
import torch
import numpy as np
from torch_geometric.data import Batch

from src.graphs.subgraph_extraction import extract_all_seed_subgraphs_for_window
from src.models.gin_encoder import GINEncoder, subgraph_to_pyg

GRAPH_DIR = r"D:\Graph anomaly detection\data\processed\graphs"

def run_pilot():
    print("Starting M3 GIN Pilot Experiment...")
    
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.npz")))
    if len(files) == 0:
        print("No processed graph files found!")
        return

    # Use first 3 control, first 3 ASD (or whatever is available up to 6)
    files = files[:6]
    print(f"Selected {len(files)} subjects for pilot.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize GIN Encoder
    model = GINEncoder(in_channels=1, hidden_channels=64, embedding_dim=128, num_layers=2)
    model.to(device)
    model.eval()

    total_graphs_processed = 0
    total_time = 0.0
    nan_count = 0
    inf_count = 0
    batch_size = 32

    # Collect subgraphs
    data_list = []
    
    for file in files:
        data = np.load(file)
        subject_id = str(data["subject_id"])
        num_windows = int(data["num_windows"])
        edge_indices = data["edge_indices"]
        edge_weights = data["edge_weights"]

        # Extract only from first 2 windows to keep pilot fast
        for w in range(min(2, num_windows)):
            subgraphs = extract_all_seed_subgraphs_for_window(
                edge_index=edge_indices[w],
                edge_weight=edge_weights[w],
                window_idx=w,
                subject_id=subject_id,
                k=1,
                num_total_nodes=360
            )
            for sg in subgraphs:
                pyg_data = subgraph_to_pyg(sg)
                data_list.append(pyg_data)

    print(f"Total subgraphs extracted: {len(data_list)}")
    
    # Process in batches
    start_time = time.time()
    
    for i in range(0, len(data_list), batch_size):
        batch_data = data_list[i:i+batch_size]
        batch = Batch.from_data_list(batch_data).to(device)
        
        with torch.no_grad():
            embeddings = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        
        total_graphs_processed += len(batch_data)
        
        if torch.isnan(embeddings).any():
            nan_count += 1
        if torch.isinf(embeddings).any():
            inf_count += 1

    end_time = time.time()
    total_time = end_time - start_time
    avg_inference_time = (total_time / total_graphs_processed) * 1000 if total_graphs_processed > 0 else 0

    print("-" * 40)
    print("Pilot Results:")
    print(f"Graphs processed: {total_graphs_processed}")
    print(f"Batch size: {batch_size}")
    print(f"Total inference time: {total_time:.4f}s")
    print(f"Average inference time per graph: {avg_inference_time:.4f} ms")
    print(f"Output shape (last batch): {embeddings.shape}")
    print(f"NaN count (batches with NaN): {nan_count}")
    print(f"Inf count (batches with Inf): {inf_count}")
    print("-" * 40)


if __name__ == "__main__":
    run_pilot()
