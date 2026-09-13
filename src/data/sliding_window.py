"""
Sliding-Window Functional Connectivity (FC) Computation.

Extracts overlapping temporal windows from ROI BOLD time-series
and calculates Pearson correlation matrices per window.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class WindowFC:
    window_idx: int
    start_t: int
    end_t: int
    fc_matrix: np.ndarray  # Shape: (360, 360), dtype: np.float32


def compute_pearson_fc(window_data: np.ndarray, remove_diagonal: bool = True) -> np.ndarray:
    """
    Computes Pearson correlation matrix for a given window (L x N_ROIs).
    
    Args:
        window_data: np.ndarray of shape (L, N_ROIs)
        remove_diagonal: Whether to set diagonal elements to 0.0
        
    Returns:
        fc: np.ndarray of shape (N_ROIs, N_ROIs), dtype np.float32
    """
    # window_data shape: (L, N_ROIs)
    L, N = window_data.shape
    if L < 3:
        raise ValueError(f"Window length {L} is too short to compute meaningful correlation.")

    # Center window data
    mean = np.mean(window_data, axis=0, keepdims=True)
    centered = window_data - mean
    
    # Standard deviation
    std = np.std(centered, axis=0, keepdims=True, ddof=1)
    
    # Avoid division by zero for any constant/quiescent ROIs
    valid_std_mask = (std > 1e-8)
    std_safe = np.where(valid_std_mask, std, 1.0)
    
    normalized = centered / std_safe
    
    # Pearson correlation matrix: (N, N)
    fc = np.dot(normalized.T, normalized) / (L - 1)
    
    # Zero out any ROIs with zero variance
    invalid_rois = ~valid_std_mask.squeeze()
    if np.any(invalid_rois):
        fc[invalid_rois, :] = 0.0
        fc[:, invalid_rois] = 0.0

    # Clean NaNs and Infs if any remain
    fc = np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Clip numerical precision drift to [-1.0, 1.0]
    fc = np.clip(fc, -1.0, 1.0)

    # Enforce strict symmetry to eliminate tiny floating-point arithmetic errors
    fc = 0.5 * (fc + fc.T)

    # Remove self-connectivity on diagonal
    if remove_diagonal:
        np.fill_diagonal(fc, 0.0)

    return fc.astype(np.float32)


def compute_sliding_window_fc(
    time_series: np.ndarray,
    window_size: int = 30,
    stride: int = 10,
    remove_diagonal: bool = True
) -> List[WindowFC]:
    """
    Generates sliding-window functional connectivity matrices for a subject time-series.
    
    Args:
        time_series: np.ndarray of shape (T, N_ROIs)
        window_size: Number of timepoints per window (default: 30)
        stride: Step size in timepoints between windows (default: 10)
        remove_diagonal: Whether to zero the diagonal (default: True)
        
    Returns:
        List of WindowFC objects containing window index, time boundaries, and 360x360 FC matrix.
    """
    T, N = time_series.shape
    if T < window_size:
        raise ValueError(
            f"Subject time series length (T={T}) is shorter than window_size ({window_size})."
        )

    num_windows = (T - window_size) // stride + 1
    windows = []

    for w_idx in range(num_windows):
        start_t = w_idx * stride
        end_t = start_t + window_size
        window_slice = time_series[start_t:end_t, :]
        
        fc_mat = compute_pearson_fc(window_slice, remove_diagonal=remove_diagonal)
        windows.append(
            WindowFC(
                window_idx=w_idx,
                start_t=start_t,
                end_t=end_t,
                fc_matrix=fc_mat
            )
        )

    return windows
