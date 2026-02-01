"""
GPU utilities for accelerated distance computation using CuPy.

Provides a unified interface that falls back to NumPy when CuPy is unavailable.
"""

import numpy as np

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def get_array_module(use_gpu=True):
    """
    Get the appropriate array module (cupy or numpy).

    Parameters:
        use_gpu (bool): Whether to use GPU if available.

    Returns:
        module: cupy if available and requested, else numpy.
    """
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def to_gpu(arr):
    """Transfer numpy array to GPU."""
    if CUPY_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """Transfer array to CPU (works for both numpy and cupy arrays)."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def is_gpu_array(arr):
    """Check if array is on GPU."""
    if CUPY_AVAILABLE:
        return isinstance(arr, cp.ndarray)
    return False


class GPUAccelerator:
    """
    GPU-accelerated operations for TVWall computations.

    Maintains GPU-resident data to minimize host-device transfers.
    """

    def __init__(self, use_gpu=True):
        """
        Initialize the GPU accelerator.

        Parameters:
            use_gpu (bool): Whether to use GPU acceleration if available.
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

        # Cached GPU arrays
        self._frames_gpu = None
        self._perm_x_gpu = None
        self._perm_y_gpu = None
        self._all_series_gpu = None

    @property
    def is_gpu_enabled(self):
        """Check if GPU acceleration is active."""
        return self.use_gpu

    def cache_frames(self, frames):
        """
        Cache video frames on GPU.

        Parameters:
            frames (np.ndarray): Video frames of shape (num_frames, height, width, 3).
        """
        if self.use_gpu:
            self._frames_gpu = cp.asarray(frames)
        else:
            self._frames_gpu = frames

    def cache_permutation(self, perm_x, perm_y):
        """
        Cache permutation arrays on GPU.

        Parameters:
            perm_x (np.ndarray): X permutation array.
            perm_y (np.ndarray): Y permutation array.
        """
        if self.use_gpu:
            self._perm_x_gpu = cp.asarray(perm_x)
            self._perm_y_gpu = cp.asarray(perm_y)
        else:
            self._perm_x_gpu = perm_x
            self._perm_y_gpu = perm_y

    def invalidate_series_cache(self):
        """Invalidate the cached all_series (call after swaps)."""
        self._all_series_gpu = None

    def get_all_series(self, frames=None, perm_x=None, perm_y=None):
        """
        GPU-accelerated version of get_all_series().

        Parameters:
            frames: Video frames (uses cached if None).
            perm_x: X permutation (uses cached if None).
            perm_y: Y permutation (uses cached if None).

        Returns:
            Array of shape (height, width, 3, num_frames) on GPU or CPU.
        """
        xp = self.xp

        # Use cached arrays if available
        if frames is None:
            frames = self._frames_gpu
        elif self.use_gpu and not is_gpu_array(frames):
            frames = cp.asarray(frames)

        if perm_x is None:
            perm_x = self._perm_x_gpu
        elif self.use_gpu and not is_gpu_array(perm_x):
            perm_x = cp.asarray(perm_x)

        if perm_y is None:
            perm_y = self._perm_y_gpu
        elif self.use_gpu and not is_gpu_array(perm_y):
            perm_y = cp.asarray(perm_y)

        # Vectorized indexing and transpose
        series = frames[:, perm_y, perm_x, :].transpose(1, 2, 3, 0)

        if self.use_gpu:
            series = series.astype(cp.float32)
        else:
            series = series.astype(np.float32)

        return series

    def compute_euclidean_distances(self, center_series, neighbor_series):
        """
        Compute Euclidean distances from center to each neighbor.

        Parameters:
            center_series: Shape (3, num_frames) - the center TV's color series.
            neighbor_series: Shape (n_neighbors, 3, num_frames) - neighbor series.

        Returns:
            Array of shape (n_neighbors,) with distances.
        """
        xp = self.xp
        diff = neighbor_series - center_series
        distances = xp.sqrt(xp.sum(diff ** 2, axis=(1, 2)))
        return distances

    def compute_squared_distances(self, center_series, neighbor_series):
        """Compute squared Euclidean distances."""
        xp = self.xp
        diff = neighbor_series - center_series
        distances = xp.sum(diff ** 2, axis=(1, 2))
        return distances

    def compute_manhattan_distances(self, center_series, neighbor_series):
        """Compute Manhattan distances."""
        xp = self.xp
        diff = neighbor_series - center_series
        distances = xp.sum(xp.abs(diff), axis=(1, 2))
        return distances

    def compute_position_dissonance(self, x, y, all_series, neighbors,
                                     distance_metric='euclidean'):
        """
        GPU-accelerated position dissonance computation.

        Parameters:
            x, y: Position coordinates.
            all_series: Full series array of shape (height, width, 3, num_frames).
            neighbors: List of (nx, ny) neighbor positions.
            distance_metric: 'euclidean', 'squared', or 'manhattan'.

        Returns:
            float: Mean distance to neighbors (dissonance).
        """
        if not neighbors:
            return 0.0

        xp = self.xp

        # Get center series
        center_series = all_series[y, x]  # (3, num_frames)

        # Stack neighbor series
        # For GPU, we want to do this efficiently
        if self.use_gpu:
            neighbor_indices_y = cp.array([ny for _, ny in neighbors], dtype=cp.int32)
            neighbor_indices_x = cp.array([nx for nx, _ in neighbors], dtype=cp.int32)
            neighbor_series = all_series[neighbor_indices_y, neighbor_indices_x]
        else:
            neighbor_series = np.array([all_series[ny, nx] for nx, ny in neighbors])

        # Compute distances based on metric
        if distance_metric == 'euclidean':
            distances = self.compute_euclidean_distances(center_series, neighbor_series)
        elif distance_metric == 'squared':
            distances = self.compute_squared_distances(center_series, neighbor_series)
        elif distance_metric == 'manhattan':
            distances = self.compute_manhattan_distances(center_series, neighbor_series)
        else:
            raise ValueError(f"GPU acceleration not supported for metric: {distance_metric}")

        return float(xp.mean(distances))

    def compute_batch_dissonance_gpu(self, positions, all_series, get_neighbors_func,
                                      kernel_size=3, distance_metric='euclidean'):
        """
        GPU-accelerated batch dissonance computation.

        Computes dissonance for multiple positions in parallel on GPU.

        Parameters:
            positions: List of (x, y) positions.
            all_series: Full series array (height, width, 3, num_frames).
            get_neighbors_func: Function to get neighbors for a position.
            kernel_size: Neighborhood kernel size.
            distance_metric: Distance metric to use.

        Returns:
            dict: Mapping from (x, y) -> dissonance value.
        """
        if not positions:
            return {}

        xp = self.xp

        # Ensure all_series is on GPU
        if self.use_gpu and not is_gpu_array(all_series):
            all_series = cp.asarray(all_series)

        results = {}

        # For now, compute each position (still faster due to GPU distance computation)
        # TODO: Further optimize by batching all positions together
        for pos in positions:
            x, y = pos
            neighbors = get_neighbors_func(x, y, kernel_size)
            results[pos] = self.compute_position_dissonance(
                x, y, all_series, neighbors, distance_metric
            )

        return results

    def compute_pairwise_euclidean(self, series_stack):
        """
        Compute pairwise Euclidean distances between all series.

        Parameters:
            series_stack: Array of shape (n_series, 3, num_frames).

        Returns:
            Distance matrix of shape (n_series, n_series).
        """
        xp = self.xp
        n = series_stack.shape[0]

        # Flatten to (n_series, 3 * num_frames)
        flat = series_stack.reshape(n, -1)

        # Compute pairwise squared distances using the expansion:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        sq_norms = xp.sum(flat ** 2, axis=1)
        dot_products = flat @ flat.T
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * dot_products

        # Clamp negative values (numerical precision) and sqrt
        sq_dists = xp.maximum(sq_dists, 0)
        distances = xp.sqrt(sq_dists)

        return distances

    def compute_pairwise_squared(self, series_stack):
        """Compute pairwise squared Euclidean distances."""
        xp = self.xp
        n = series_stack.shape[0]
        flat = series_stack.reshape(n, -1)
        sq_norms = xp.sum(flat ** 2, axis=1)
        dot_products = flat @ flat.T
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * dot_products
        return xp.maximum(sq_dists, 0)

    def compute_pairwise_manhattan(self, series_stack):
        """
        Compute pairwise Manhattan distances.

        Note: This is O(n^2 * d) and less efficient than Euclidean on GPU.
        """
        xp = self.xp
        n = series_stack.shape[0]
        flat = series_stack.reshape(n, -1)

        # Use broadcasting: shape (n, 1, d) - (1, n, d) -> (n, n, d)
        # This can be memory-intensive for large n
        if n > 1000:
            # Fall back to loop for large n to avoid memory issues
            distances = xp.zeros((n, n), dtype=xp.float32)
            for i in range(n):
                diff = xp.abs(flat[i] - flat)
                distances[i] = xp.sum(diff, axis=1)
        else:
            diff = xp.abs(flat[:, None, :] - flat[None, :, :])
            distances = xp.sum(diff, axis=2)

        return distances

    def evaluate_swap_batch(self, swap_candidates, all_series, get_neighbors_func,
                            kernel_size=3, distance_metric='euclidean'):
        """
        Evaluate multiple swap candidates in parallel.

        Parameters:
            swap_candidates: List of ((x1, y1), (x2, y2)) swap pairs.
            all_series: Current series array.
            get_neighbors_func: Function to get neighbors.
            kernel_size: Neighborhood kernel size.
            distance_metric: Distance metric.

        Returns:
            List of (improvement, swap_pair) tuples, sorted by improvement descending.
        """
        if not swap_candidates:
            return []

        xp = self.xp

        # Ensure series is on GPU
        if self.use_gpu and not is_gpu_array(all_series):
            all_series = cp.asarray(all_series)

        results = []

        for pos1, pos2 in swap_candidates:
            x1, y1 = pos1
            x2, y2 = pos2

            # Get neighbors for both positions
            neighbors1 = get_neighbors_func(x1, y1, kernel_size)
            neighbors2 = get_neighbors_func(x2, y2, kernel_size)

            # Compute current dissonance
            diss1_before = self.compute_position_dissonance(
                x1, y1, all_series, neighbors1, distance_metric
            )
            diss2_before = self.compute_position_dissonance(
                x2, y2, all_series, neighbors2, distance_metric
            )

            # Simulate swap by creating swapped view
            # Make a shallow copy and swap the relevant entries
            if self.use_gpu:
                series_copy = all_series.copy()
            else:
                series_copy = all_series.copy()

            series_copy[y1, x1], series_copy[y2, x2] = (
                series_copy[y2, x2].copy(), series_copy[y1, x1].copy()
            )

            # Compute new dissonance
            diss1_after = self.compute_position_dissonance(
                x1, y1, series_copy, neighbors1, distance_metric
            )
            diss2_after = self.compute_position_dissonance(
                x2, y2, series_copy, neighbors2, distance_metric
            )

            improvement = (diss1_before + diss2_before) - (diss1_after + diss2_after)
            results.append((improvement, (pos1, pos2)))

        # Sort by improvement (highest first)
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def sync(self):
        """Synchronize GPU operations (wait for completion)."""
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()

    def free_memory(self):
        """Free cached GPU memory."""
        self._frames_gpu = None
        self._perm_x_gpu = None
        self._perm_y_gpu = None
        self._all_series_gpu = None
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()


# Convenience function to check GPU availability
def check_gpu():
    """
    Check GPU availability and print status.

    Returns:
        bool: True if CuPy and GPU are available.
    """
    if not CUPY_AVAILABLE:
        print("CuPy not installed. GPU acceleration unavailable.")
        print("Install with: pip install cupy-cuda12x  (adjust for your CUDA version)")
        return False

    try:
        # Test GPU access
        x = cp.array([1, 2, 3])
        _ = x + x
        cp.cuda.Stream.null.synchronize()

        # Get GPU info
        device = cp.cuda.Device()
        props = device.attributes
        mem_info = device.mem_info

        print(f"GPU acceleration available!")
        print(f"  Device: {device.id}")
        print(f"  Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
        return True

    except Exception as e:
        print(f"GPU test failed: {e}")
        return False
