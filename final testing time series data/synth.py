"""Synthetic ROI time-series generator (NOT patient data). A factor model gives a normal-like 360-ROI scan; an 'injected anomaly' scan
adds a shared signal to one seed ROI and its 1-hop neighbours for a stretch of time, which changes that sub-structure's connectivity."""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from src.data.sliding_window import compute_sliding_window_fc
from src.data.graph_construction import construct_temporal_graphs
from src.data.subgraph_cache import select_seed_rois
N_ROI = 360; N_NET = 12


def ar1(rng, n, T, phi):
    x = np.zeros((n, T)); e = rng.standard_normal((n, T))
    for t in range(1, T): x[:, t] = phi * x[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]
    x[:, 0] = e[:, 0]; return x


def normal_scan(rng, T=150):
    net = np.arange(N_ROI) % N_NET; rng.shuffle(net)
    shared = ar1(rng, N_NET, T, 0.6)
    load = rng.uniform(0.5, 0.8, N_ROI)
    x = load[:, None] * shared[net] + np.sqrt(1 - load[:, None] ** 2) * ar1(rng, N_ROI, T, 0.4)
    scale = rng.uniform(30, 100, N_ROI)
    return (x * scale[:, None]).astype(np.float64), scale


def seed_neighbours(ts, seed):
    fcs = compute_sliding_window_fc(ts.T.astype(np.float32), window_size=30, stride=10, remove_diagonal=True)
    g = construct_temporal_graphs(fcs, sparsity_ratio=0.20, keep_signed_weights=True)[0]
    ei = np.asarray(g.edge_index); nb = np.unique(np.concatenate([ei[1][ei[0] == seed], ei[0][ei[1] == seed]]))
    return np.unique(np.concatenate([[seed], nb]))


def inject(rng, ts, scale, amp, n_seeds, seg=40):
    """add a common AR(1) signal (amplitude `amp` x each ROI's own scale) to the seed ROI + its 1-hop neighbours during one segment"""
    T = ts.shape[1]; out = ts.copy(); seeds = select_seed_rois(20, 42)
    for s in rng.choice(seeds, n_seeds, replace=False):
        grp = seed_neighbours(ts, int(s)); t0 = int(rng.integers(35, T - seg - 5))
        sig = ar1(rng, 1, seg, 0.5)[0]
        out[grp, t0:t0 + seg] += amp * scale[grp, None] * sig[None, :]
    return out
