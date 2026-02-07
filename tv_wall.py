#!/usr/bin/env python
# coding: utf-8

"""
TVWall - A class representing a wall of CRT TVs displaying a video.

Each TV displays one pixel position's color over time. TVs can be swapped
from their original positions to simulate scrambling/unscrambling.
"""

import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# GPU acceleration support
try:
    from gpu_utils import GPUAccelerator, CUPY_AVAILABLE, to_cpu
except ImportError:
    GPUAccelerator = None
    CUPY_AVAILABLE = False
    def to_cpu(x): return x


def compute_color_entropy(frame):
    """
    Compute Shannon entropy of a color image based on color histogram.

    Uses a coarse 8x8x8 color quantization (512 bins) to compute the entropy
    of the color distribution. This captures how varied the colors are in the frame.

    Parameters:
        frame (np.ndarray): RGB image of shape (height, width, 3) with uint8 values.

    Returns:
        float: Shannon entropy in bits. Range is 0 (single color) to ~9 (maximum diversity).
               Typical values: <2 = very uniform, 2-4 = low variety, 4-6 = moderate, >6 = high variety.
    """
    # Quantize to 8x8x8 = 512 color bins (divide by 32 to get 0-7 range)
    quantized = (frame // 32).astype(np.uint8)

    # Convert to single index: r*64 + g*8 + b
    color_indices = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]

    # Count occurrences of each color bin
    counts = np.bincount(color_indices.ravel(), minlength=512)

    # Convert to probabilities (exclude zero counts)
    probs = counts[counts > 0] / counts.sum()

    # Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))

    return entropy


class TVWall:
    """
    Represents a wall of CRT TVs where each TV displays a single pixel's
    color sequence from a video.

    Attributes:
        width (int): Width of the video in pixels (number of TV columns).
        height (int): Height of the video in pixels (number of TV rows).
        _perm_y (np.ndarray): Index array mapping current positions to original y coords.
        _perm_x (np.ndarray): Index array mapping current positions to original x coords.
    """

    def __init__(self, video_path, num_frames=None, start_frame=0, stride=1, crop_percent=100,
                 use_gpu=True, min_entropy=0.0):
        """
        Initialize a TVWall from a video file.

        Parameters:
            video_path (str): Path to the video file.
            num_frames (int, optional): Number of frames to extract. If None, extracts all frames.
            start_frame (int): Starting frame number (default: 0).
            stride (int): Frame skip interval (default: 1 for every frame).
            crop_percent (float): Percentage of video to keep, cropped to center (1-100).
                                  Maintains original aspect ratio. Default: 100 (no crop).
            use_gpu (bool): Whether to use GPU acceleration if available (default: True).
            min_entropy (float): Minimum Shannon entropy threshold for frames (default: 0.0).
                                 Frames with entropy below this are skipped. Useful values:
                                 0 = no filtering, 2-3 = skip very uniform frames,
                                 4+ = only keep frames with moderate color variety.
        """
        self.video_path = video_path
        self.start_frame = start_frame
        self.stride = stride
        self.crop_percent = max(1, min(100, crop_percent))
        self.min_entropy = min_entropy

        self._load_video(num_frames)
        self._apply_crop()
        self._init_permutation()

        # Initialize GPU accelerator
        self._use_gpu = use_gpu and CUPY_AVAILABLE and GPUAccelerator is not None
        if self._use_gpu:
            self._gpu = GPUAccelerator(use_gpu=True)
            self._gpu.cache_frames(self._frames)
            self._gpu.cache_permutation(self._perm_x, self._perm_y)
        else:
            self._gpu = None

    def _load_video(self, num_frames):
        """Load frames from the video file, filtering by entropy if min_entropy > 0."""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Error: Couldn't open the video file: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine how many frames to extract
        if num_frames is None:
            # Calculate max frames possible with given stride
            num_frames = (total_frames - self.start_frame) // self.stride

        # Validate frame range
        frames_span = num_frames * self.stride
        if self.start_frame + frames_span > total_frames:
            self.start_frame = max(0, total_frames - frames_span)

        frames = []
        skipped_count = 0
        frame_entropies = []  # Track entropies for reporting

        # If filtering by entropy, we may need to scan more frames to get num_frames accepted
        max_frame_idx = total_frames
        current_frame_idx = self.start_frame

        desc = "Loading video frames" if self.min_entropy == 0 else f"Loading frames (entropy >= {self.min_entropy:.1f})"

        with tqdm(total=num_frames, desc=desc) as pbar:
            while len(frames) < num_frames and current_frame_idx < max_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Check entropy if filtering is enabled
                if self.min_entropy > 0:
                    entropy = compute_color_entropy(frame)
                    frame_entropies.append(entropy)

                    if entropy < self.min_entropy:
                        skipped_count += 1
                        current_frame_idx += self.stride
                        continue

                frames.append(frame)
                pbar.update(1)
                current_frame_idx += self.stride

        cap.release()

        if skipped_count > 0:
            total_considered = len(frames) + skipped_count
            print(f"Entropy filter: kept {len(frames)}/{total_considered} frames "
                  f"(skipped {skipped_count} with entropy < {self.min_entropy:.1f})")
            if frame_entropies:
                print(f"Entropy range: {min(frame_entropies):.2f} - {max(frame_entropies):.2f}, "
                      f"mean: {np.mean(frame_entropies):.2f}")

        if len(frames) == 0:
            raise ValueError(f"No frames passed the entropy filter (min_entropy={self.min_entropy}). "
                             "Try lowering the threshold.")

        self._frames = np.array(frames)
        self.num_frames = len(frames)
        # Note: OpenCV returns frames as (height, width, channels)
        self.height = self._frames.shape[1]
        self.width = self._frames.shape[2]

    def _apply_crop(self):
        """Apply center crop based on crop_percent, maintaining aspect ratio."""
        if self.crop_percent >= 100:
            return

        scale = self.crop_percent / 100.0
        orig_h, orig_w = self.height, self.width

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Calculate crop offsets to center
        x_start = (orig_w - new_w) // 2
        y_start = (orig_h - new_h) // 2

        # Crop all frames
        self._frames = self._frames[:, y_start:y_start + new_h, x_start:x_start + new_w, :]

        # Update dimensions
        self.height = new_h
        self.width = new_w

    def _init_permutation(self):
        """Initialize permutation arrays to identity mapping."""
        # _perm_y[y, x] and _perm_x[y, x] give the original (y, x) of the TV at position (y, x)
        self._perm_y, self._perm_x = np.mgrid[0:self.height, 0:self.width]

    @property
    def num_tvs(self):
        """Total number of TVs in the wall."""
        return self.width * self.height

    @property
    def gpu_enabled(self):
        """Check if GPU acceleration is active."""
        return self._gpu is not None and self._gpu.is_gpu_enabled

    def get_original_position(self, x, y):
        """
        Get the original TV position that is currently displayed at (x, y).

        Parameters:
            x (int): Current x position.
            y (int): Current y position.

        Returns:
            tuple: (orig_x, orig_y) - the original position of the TV at this location.
        """
        return (self._perm_x[y, x], self._perm_y[y, x])

    def get_current_position(self, orig_x, orig_y):
        """
        Get the current position of a TV that was originally at (orig_x, orig_y).

        Parameters:
            orig_x (int): Original x position.
            orig_y (int): Original y position.

        Returns:
            tuple: (cur_x, cur_y) - the current position of this TV.
        """
        # Find where in the permutation arrays the original position appears
        mask = (self._perm_x == orig_x) & (self._perm_y == orig_y)
        positions = np.argwhere(mask)
        if len(positions) > 0:
            cur_y, cur_x = positions[0]
            return (cur_x, cur_y)
        return (orig_x, orig_y)

    def swap(self, orig_pos, new_pos):
        """
        Place the TV originally at orig_pos into the new_pos slot.

        Note: This overwrites whatever was at new_pos. For a true swap
        of two TVs, use swap_positions() instead.

        Parameters:
            orig_pos (tuple): Original position (x, y) of the TV.
            new_pos (tuple): New position (x, y) where the TV should be placed.
        """
        self._perm_x[new_pos[1], new_pos[0]] = orig_pos[0]
        self._perm_y[new_pos[1], new_pos[0]] = orig_pos[1]

    def swap_positions(self, pos1, pos2):
        """
        Swap the TVs at two positions with each other.

        Parameters:
            pos1 (tuple): First position (x, y).
            pos2 (tuple): Second position (x, y).
        """
        x1, y1 = pos1
        x2, y2 = pos2

        # Swap the permutation values
        self._perm_x[y1, x1], self._perm_x[y2, x2] = self._perm_x[y2, x2], self._perm_x[y1, x1]
        self._perm_y[y1, x1], self._perm_y[y2, x2] = self._perm_y[y2, x2], self._perm_y[y1, x1]

        # Invalidate GPU caches
        if self._gpu is not None:
            self._gpu.cache_permutation(self._perm_x, self._perm_y)
            self._gpu.invalidate_series_cache()

    def scramble(self, seed=None):
        """
        Randomly scramble all TV positions.

        Parameters:
            seed (int, optional): Random seed for reproducibility.
        """
        self.random_swaps(self.num_tvs, seed=seed)

    def random_swaps(self, num_positions, seed=None):
        """
        Randomly shuffle a subset of TV positions.

        Parameters:
            num_positions (int): Number of positions to randomly shuffle.
                                 If >= num_tvs, all positions are shuffled.
            seed (int, optional): Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset to identity first
        self._init_permutation()

        # Clamp to total number of TVs
        num_positions = min(num_positions, self.num_tvs)

        # Select random flat indices to shuffle
        all_indices = np.arange(self.num_tvs)
        selected_indices = np.random.choice(all_indices, size=num_positions, replace=False)

        # Extract the values at those positions
        flat_perm_x = self._perm_x.ravel()
        flat_perm_y = self._perm_y.ravel()

        selected_x = flat_perm_x[selected_indices].copy()
        selected_y = flat_perm_y[selected_indices].copy()

        # Shuffle
        shuffle_order = np.random.permutation(num_positions)
        flat_perm_x[selected_indices] = selected_x[shuffle_order]
        flat_perm_y[selected_indices] = selected_y[shuffle_order]

        # Update GPU cache
        if self._gpu is not None:
            self._gpu.cache_permutation(self._perm_x, self._perm_y)
            self._gpu.invalidate_series_cache()

    def reset_swaps(self):
        """Reset all swaps, returning TVs to their original positions."""
        self._init_permutation()
        if self._gpu is not None:
            self._gpu.cache_permutation(self._perm_x, self._perm_y)
            self._gpu.invalidate_series_cache()

    def short_swaps(self, num_swaps, max_dist, seed=None):
        """
        Perform random swaps within a maximum distance.

        Each swap pairs two positions that are within max_dist pixels of each other.
        Useful for testing local unscrambling algorithms.

        Parameters:
            num_swaps (int): Number of swap pairs to make.
            max_dist (int): Maximum Euclidean distance between swapped positions.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            list: List of ((x1, y1), (x2, y2)) tuples for each swap made.
        """
        if seed is not None:
            np.random.seed(seed)

        used = set()
        swap_pairs = []
        attempts = 0
        max_attempts = num_swaps * 100

        while len(swap_pairs) < num_swaps and attempts < max_attempts:
            attempts += 1
            x1 = np.random.randint(0, self.width)
            y1 = np.random.randint(0, self.height)

            if (x1, y1) in used:
                continue

            # Find candidates within distance
            candidates = []
            for dx in range(-max_dist, max_dist + 1):
                for dy in range(-max_dist, max_dist + 1):
                    if dx == 0 and dy == 0:
                        continue
                    x2, y2 = x1 + dx, y1 + dy
                    if 0 <= x2 < self.width and 0 <= y2 < self.height:
                        if np.sqrt(dx**2 + dy**2) <= max_dist and (x2, y2) not in used:
                            candidates.append((x2, y2))

            if not candidates:
                continue

            x2, y2 = candidates[np.random.randint(len(candidates))]
            pos1, pos2 = (x1, y1), (x2, y2)
            self.swap_positions(pos1, pos2)
            used.add(pos1)
            used.add(pos2)
            swap_pairs.append((pos1, pos2))

        return swap_pairs

    def pair_swaps(self, num_swaps, seed=None):
        """
        Perform random swaps with no distance limit.

        Randomly selects pairs of positions and swaps them.

        Parameters:
            num_swaps (int): Number of swap pairs to make.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            list: List of ((x1, y1), (x2, y2)) tuples for each swap made.
        """
        if seed is not None:
            np.random.seed(seed)

        all_positions = [(x, y) for y in range(self.height) for x in range(self.width)]
        np.random.shuffle(all_positions)

        swap_pairs = []
        actual_swaps = min(num_swaps, len(all_positions) // 2)
        for i in range(actual_swaps):
            pos1 = all_positions[i * 2]
            pos2 = all_positions[i * 2 + 1]
            self.swap_positions(pos1, pos2)
            swap_pairs.append((pos1, pos2))

        return swap_pairs

    def get_tv_color_series(self, orig_x, orig_y):
        """
        Get the color time-series for a TV at its original position.

        Parameters:
            orig_x (int): Original x position.
            orig_y (int): Original y position.

        Returns:
            np.array: Array of shape (num_frames, 3) with RGB values.
        """
        return self._frames[:, orig_y, orig_x]

    def get_frame_image(self, timestep):
        """
        Get the image at a specific timestep with current swap configuration.

        Parameters:
            timestep (int): Frame index (0 to num_frames-1).

        Returns:
            PIL.Image: The frame image with TVs in their swapped positions.
        """
        if timestep < 0 or timestep >= self.num_frames:
            raise ValueError(f"Timestep {timestep} out of range [0, {self.num_frames - 1}]")

        # Use advanced indexing to remap all pixels at once
        output = self._frames[timestep, self._perm_y, self._perm_x]

        return Image.fromarray(output, mode='RGB')

    def save_frame(self, timestep, output_path):
        """
        Save the frame at a specific timestep to an image file.

        Parameters:
            timestep (int): Frame index (0 to num_frames-1).
            output_path (str): Path to save the image (e.g., 'frame.png').
        """
        img = self.get_frame_image(timestep)
        img.save(output_path)
        print(f"Saved frame {timestep} to {output_path}")

    def save_video(self, output_path, fps=30):
        """
        Save the video with current swap configuration using ffmpeg.

        Parameters:
            output_path (str): Path to save the video (e.g., 'output.mp4').
            fps (int): Frames per second for the output video (default: 30).
        """
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all frames as images
            for i in tqdm(range(self.num_frames), desc="Generating frames"):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                img = self.get_frame_image(i)
                img.save(frame_path)

            # Use ffmpeg to encode the video
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if exists
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # High quality
                output_path
            ]

            print(f"Encoding video with ffmpeg...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

            print(f"Saved video to {output_path}")

    @property
    def num_swapped(self):
        """Count of TVs not in their original position."""
        identity_y, identity_x = np.mgrid[0:self.height, 0:self.width]
        return np.sum((self._perm_x != identity_x) | (self._perm_y != identity_y))

    def get_neighbors(self, x, y, kernel_size=3):
        """
        Get valid neighbor positions for a given (x, y) coordinate.

        Parameters:
            x (int): X coordinate.
            y (int): Y coordinate.
            kernel_size (int): Size of the neighborhood kernel (default: 3).

        Returns:
            list: List of (x, y) tuples for valid neighbor positions.
        """
        neighbors = []
        radius = kernel_size // 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def get_all_series(self, force_cpu=False):
        """
        Get all TV color series with current swap configuration.

        Parameters:
            force_cpu (bool): If True, always return a numpy array on CPU.
                             If False and GPU is enabled, may return a cupy array.

        Returns:
            np.ndarray: Array of shape (height, width, 3, num_frames) with color series
                       for each position according to current permutation.
        """
        if self._gpu is not None and not force_cpu:
            # Use GPU-accelerated version
            series = self._gpu.get_all_series()
            return series
        else:
            # Use vectorized indexing: _frames is (num_frames, height, width, 3)
            # _perm_y and _perm_x give original positions for each current position
            # Result: (num_frames, height, width, 3) -> transpose to (height, width, 3, num_frames)
            series = self._frames[:, self._perm_y, self._perm_x, :].transpose(1, 2, 3, 0).astype(np.float32)
            return series

    def compute_position_dissonance(self, x, y, all_series=None, kernel_size=3,
                                     distance_metric='dtw', window=0.1):
        """
        Compute neighbor dissonance for a single position.

        Parameters:
            x (int): X coordinate.
            y (int): Y coordinate.
            all_series (np.ndarray, optional): Precomputed series array from get_all_series().
                                               If None, will compute on-the-fly.
            kernel_size (int): Size of neighborhood kernel (default: 3).
            distance_metric (str): Distance metric to use. Options:
                - 'dtw': Dynamic Time Warping (default)
                - 'euclidean': Euclidean distance
                - 'squared': Squared Euclidean distance
                See aeon.distances for more options.
            window (float): Sakoe-Chiba band window for DTW (0.0-1.0).

        Returns:
            float: Mean distance to neighbors (dissonance).
        """
        neighbors = self.get_neighbors(x, y, kernel_size)
        if not neighbors:
            return 0.0

        if all_series is None:
            all_series = self.get_all_series()

        # Use GPU acceleration for simple metrics if available
        if self._gpu is not None and distance_metric in ('euclidean', 'squared', 'manhattan', 'cosine'):
            return self._gpu.compute_position_dissonance(
                x, y, all_series, neighbors, distance_metric
            )

        # CPU path: ensure we have numpy arrays
        all_series_cpu = to_cpu(all_series)
        center_series = all_series_cpu[y, x]  # shape: (3, num_frames)

        # For simple metrics, use fast NumPy computation instead of aeon
        if distance_metric in ('euclidean', 'squared', 'manhattan', 'cosine'):
            # Stack neighbor series: shape (n_neighbors, 3, num_frames)
            neighbor_series = np.array([all_series_cpu[ny, nx] for nx, ny in neighbors])

            if distance_metric == 'cosine':
                # Cosine distance: 1 - (a·b / (|a|*|b|))
                # Flatten to 1D vectors for dot product
                center_flat = center_series.flatten().astype(np.float64)
                neighbor_flat = neighbor_series.reshape(len(neighbors), -1).astype(np.float64)
                dots = neighbor_flat @ center_flat
                center_norm = np.linalg.norm(center_flat)
                neighbor_norms = np.linalg.norm(neighbor_flat, axis=1)
                # Avoid division by zero
                denom = center_norm * neighbor_norms
                denom = np.where(denom == 0, 1.0, denom)
                distances = 1.0 - dots / denom
            else:
                # Compute distances from center to each neighbor
                diff = neighbor_series - center_series  # broadcasting

                if distance_metric == 'euclidean':
                    # Euclidean: sqrt(sum of squared differences)
                    distances = np.sqrt(np.sum(diff ** 2, axis=(1, 2)))
                elif distance_metric == 'squared':
                    # Squared Euclidean: sum of squared differences
                    distances = np.sum(diff ** 2, axis=(1, 2))
                else:  # manhattan
                    # Manhattan: sum of absolute differences
                    distances = np.sum(np.abs(diff), axis=(1, 2))

            return float(distances.mean())

        # For DTW and other complex metrics, use aeon (CPU only)
        series_list = [center_series]
        for nx, ny in neighbors:
            series_list.append(all_series_cpu[ny, nx])

        stacked = np.array(series_list)

        if distance_metric == 'dtw':
            from aeon.distances import dtw_pairwise_distance
            distances = dtw_pairwise_distance(stacked, window=window)
        else:
            # Try to import the requested metric from aeon.distances
            import aeon.distances as aeon_dist
            pairwise_func = getattr(aeon_dist, f'{distance_metric}_pairwise_distance', None)
            if pairwise_func is None:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            distances = pairwise_func(stacked)

        return float(distances[0, 1:].mean())

    def compute_total_dissonance(self, all_series=None, kernel_size=3,
                                  distance_metric='dtw', window=0.1, positions=None):
        """
        Compute sum of dissonance over specified positions (or all if None).

        Parameters:
            all_series (np.ndarray, optional): Precomputed series array from get_all_series().
            kernel_size (int): Size of neighborhood kernel (default: 3).
            distance_metric (str): Distance metric to use ('dtw', 'euclidean', 'squared').
            window (float): Sakoe-Chiba band window for DTW (0.0-1.0).
            positions (list, optional): List of (x, y) positions to compute. If None, all positions.

        Returns:
            float: Total dissonance.
        """
        if all_series is None:
            all_series = self.get_all_series()

        if positions is None:
            positions = [(x, y) for y in range(self.height) for x in range(self.width)]

        total = 0.0
        for x, y in positions:
            total += self.compute_position_dissonance(
                x, y, all_series, kernel_size, distance_metric, window
            )
        return total

    def compute_dissonance_map(self, all_series=None, kernel_size=3,
                                distance_metric='dtw', window=0.1):
        """
        Compute dissonance for all positions.

        Parameters:
            all_series (np.ndarray, optional): Precomputed series array from get_all_series().
            kernel_size (int): Size of neighborhood kernel (default: 3).
            distance_metric (str): Distance metric to use ('dtw', 'euclidean', 'squared').
            window (float): Sakoe-Chiba band window for DTW (0.0-1.0).

        Returns:
            np.ndarray: Array of shape (height, width) with dissonance values.
        """
        if all_series is None:
            all_series = self.get_all_series()

        # For GPU-accelerated simple metrics, use fully vectorized map computation
        if self._gpu is not None and distance_metric in ('euclidean', 'squared', 'manhattan', 'cosine'):
            return self._gpu.compute_dissonance_map_gpu(
                all_series, kernel_size, distance_metric
            )

        # Vectorized CPU path for simple metrics (including cosine)
        if distance_metric in ('euclidean', 'squared', 'manhattan', 'cosine'):
            return self._compute_dissonance_map_vectorized(
                all_series, kernel_size, distance_metric
            )

        # Slow per-pixel CPU path (DTW and other complex metrics)
        dissonance_map = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                dissonance_map[y, x] = self.compute_position_dissonance(
                    x, y, all_series, kernel_size, distance_metric, window
                )
        return dissonance_map

    def _compute_dissonance_map_vectorized(self, all_series, kernel_size=3,
                                            distance_metric='euclidean'):
        """
        Vectorized CPU dissonance map for simple metrics.

        Iterates over kernel offsets (not pixels), computing distances for all
        pixels at once using NumPy broadcasting. Same approach as the GPU path.
        """
        all_series = to_cpu(all_series).astype(np.float64)
        height, width = self.height, self.width
        radius = kernel_size // 2

        total_distances = np.zeros((height, width), dtype=np.float64)
        neighbor_counts = np.zeros((height, width), dtype=np.float64)

        # Precompute norms for cosine distance
        if distance_metric == 'cosine':
            # Flatten channels+frames: (H, W, 3*F)
            flat_series = all_series.reshape(height, width, -1)
            norms = np.linalg.norm(flat_series, axis=2)  # (H, W)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue

                # Compute overlap regions
                c_x_start = max(0, -dx)
                c_x_end = min(width, width - dx)
                c_y_start = max(0, -dy)
                c_y_end = min(height, height - dy)
                n_x_start = c_x_start + dx
                n_x_end = c_x_end + dx
                n_y_start = c_y_start + dy
                n_y_end = c_y_end + dy

                center = all_series[c_y_start:c_y_end, c_x_start:c_x_end]
                neighbor = all_series[n_y_start:n_y_end, n_x_start:n_x_end]

                if distance_metric == 'cosine':
                    center_flat = flat_series[c_y_start:c_y_end, c_x_start:c_x_end]
                    neighbor_flat = flat_series[n_y_start:n_y_end, n_x_start:n_x_end]
                    dots = np.sum(center_flat * neighbor_flat, axis=2)
                    c_norms = norms[c_y_start:c_y_end, c_x_start:c_x_end]
                    n_norms = norms[n_y_start:n_y_end, n_x_start:n_x_end]
                    denom = c_norms * n_norms
                    denom = np.where(denom == 0, 1.0, denom)
                    dist = 1.0 - dots / denom
                else:
                    diff = neighbor - center
                    if distance_metric == 'euclidean':
                        dist = np.sqrt(np.sum(diff ** 2, axis=(2, 3)))
                    elif distance_metric == 'squared':
                        dist = np.sum(diff ** 2, axis=(2, 3))
                    elif distance_metric == 'manhattan':
                        dist = np.sum(np.abs(diff), axis=(2, 3))

                total_distances[c_y_start:c_y_end, c_x_start:c_x_end] += dist
                neighbor_counts[c_y_start:c_y_end, c_x_start:c_x_end] += 1.0

        dissonance_map = np.where(
            neighbor_counts > 0,
            total_distances / neighbor_counts,
            0.0
        )
        return dissonance_map

    def compute_batch_dissonance(self, positions, all_series=None, kernel_size=3,
                                  distance_metric='dtw', window=0.1):
        """
        Compute dissonance for multiple positions efficiently.

        This method batches all the pairwise distance computations together
        for better performance when computing dissonance for many positions.
        Note: Batching is most beneficial for expensive metrics like DTW.
        For simple metrics like euclidean, individual computation may be faster
        for small numbers of positions.

        Parameters:
            positions (list): List of (x, y) tuples to compute dissonance for.
            all_series (np.ndarray, optional): Precomputed series array from get_all_series().
            kernel_size (int): Size of neighborhood kernel (default: 3).
            distance_metric (str): Distance metric to use ('dtw', 'euclidean', etc.).
            window (float): Sakoe-Chiba band window for DTW (0.0-1.0).

        Returns:
            dict: Mapping from (x, y) -> dissonance value.
        """
        if all_series is None:
            all_series = self.get_all_series()

        if not positions:
            return {}

        # Use GPU acceleration for simple metrics if available
        if self._gpu is not None and distance_metric in ('euclidean', 'squared', 'manhattan', 'cosine'):
            return self._gpu.compute_batch_dissonance_gpu(
                positions, all_series, self.get_neighbors,
                kernel_size, distance_metric
            )

        # For simple metrics without GPU, use individual computation (fast NumPy path).
        # Batch computation only beneficial for DTW due to aeon overhead.
        if distance_metric != 'dtw':
            return {pos: self.compute_position_dissonance(pos[0], pos[1], all_series,
                                                          kernel_size, distance_metric, window)
                    for pos in positions}

        # Build list of all unique (center, neighbor) pairs we need to compute
        # and track which positions need which distances
        all_pairs = []  # List of ((x1, y1), (x2, y2)) pairs
        pair_to_idx = {}  # Map (pos1, pos2) -> index in all_pairs
        pos_to_neighbor_pairs = {}  # Map pos -> list of pair indices

        for pos in positions:
            x, y = pos
            neighbors = self.get_neighbors(x, y, kernel_size)
            if not neighbors:
                pos_to_neighbor_pairs[pos] = []
                continue

            pair_indices = []
            for nx, ny in neighbors:
                # Use canonical ordering to avoid duplicate pairs
                pair = (pos, (nx, ny))
                if pair not in pair_to_idx:
                    pair_to_idx[pair] = len(all_pairs)
                    all_pairs.append(pair)
                pair_indices.append(pair_to_idx[pair])
            pos_to_neighbor_pairs[pos] = pair_indices

        if not all_pairs:
            return {pos: 0.0 for pos in positions}

        # Collect all series we need
        all_positions_needed = set()
        for (p1, p2) in all_pairs:
            all_positions_needed.add(p1)
            all_positions_needed.add(p2)

        pos_list = list(all_positions_needed)
        pos_to_series_idx = {pos: i for i, pos in enumerate(pos_list)}

        # Stack all needed series: shape (n_series, 3, num_frames)
        series_stack = np.array([all_series[y, x] for x, y in pos_list])

        # Compute pairwise distances for all series at once
        if distance_metric == 'dtw':
            from aeon.distances import dtw_pairwise_distance
            dist_matrix = dtw_pairwise_distance(series_stack, window=window)
        elif distance_metric == 'euclidean':
            from aeon.distances import euclidean_pairwise_distance
            dist_matrix = euclidean_pairwise_distance(series_stack)
        elif distance_metric == 'squared':
            from aeon.distances import squared_pairwise_distance
            dist_matrix = squared_pairwise_distance(series_stack)
        elif distance_metric == 'manhattan':
            from aeon.distances import manhattan_pairwise_distance
            dist_matrix = manhattan_pairwise_distance(series_stack)
        elif distance_metric == 'cosine':
            flat = series_stack.reshape(series_stack.shape[0], -1)
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(flat, flat, metric='cosine')
        else:
            import aeon.distances as aeon_dist
            pairwise_func = getattr(aeon_dist, f'{distance_metric}_pairwise_distance', None)
            if pairwise_func is None:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            dist_matrix = pairwise_func(series_stack)

        # Extract distances for each pair
        pair_distances = np.zeros(len(all_pairs))
        for i, (p1, p2) in enumerate(all_pairs):
            idx1 = pos_to_series_idx[p1]
            idx2 = pos_to_series_idx[p2]
            pair_distances[i] = dist_matrix[idx1, idx2]

        # Compute dissonance for each position
        result = {}
        for pos in positions:
            pair_indices = pos_to_neighbor_pairs[pos]
            if not pair_indices:
                result[pos] = 0.0
            else:
                result[pos] = np.mean([pair_distances[i] for i in pair_indices])

        return result

    def __repr__(self):
        return (f"TVWall(video='{self.video_path}', width={self.width}, "
                f"height={self.height}, frames={self.num_frames}, "
                f"swapped={self.num_swapped})")
