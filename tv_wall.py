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

    def __init__(self, video_path, num_frames=None, start_frame=0, stride=1):
        """
        Initialize a TVWall from a video file.

        Parameters:
            video_path (str): Path to the video file.
            num_frames (int, optional): Number of frames to extract. If None, extracts all frames.
            start_frame (int): Starting frame number (default: 0).
            stride (int): Frame skip interval (default: 1 for every frame).
        """
        self.video_path = video_path
        self.start_frame = start_frame
        self.stride = stride

        self._load_video(num_frames)
        self._init_permutation()

    def _load_video(self, num_frames):
        """Load frames from the video file."""
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
        for i in tqdm(range(num_frames), desc="Loading video frames"):
            frame_idx = self.start_frame + (i * self.stride)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break

        cap.release()

        self._frames = np.array(frames)
        self.num_frames = len(frames)
        # Note: OpenCV returns frames as (height, width, channels)
        self.height = self._frames.shape[1]
        self.width = self._frames.shape[2]

    def _init_permutation(self):
        """Initialize permutation arrays to identity mapping."""
        # _perm_y[y, x] and _perm_x[y, x] give the original (y, x) of the TV at position (y, x)
        self._perm_y, self._perm_x = np.mgrid[0:self.height, 0:self.width]

    @property
    def num_tvs(self):
        """Total number of TVs in the wall."""
        return self.width * self.height

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

    def reset_swaps(self):
        """Reset all swaps, returning TVs to their original positions."""
        self._init_permutation()

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

    def get_all_series(self):
        """
        Get all TV color series with current swap configuration.

        Returns:
            np.ndarray: Array of shape (height, width, 3, num_frames) with color series
                       for each position according to current permutation.
        """
        # Build series array using current permutation
        series = np.zeros((self.height, self.width, 3, self.num_frames), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                orig_x, orig_y = self.get_original_position(x, y)
                series[y, x] = self.get_tv_color_series(orig_x, orig_y).T
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

        center_series = all_series[y, x]
        series_list = [center_series]
        for nx, ny in neighbors:
            series_list.append(all_series[ny, nx])

        stacked = np.array(series_list)

        if distance_metric == 'dtw':
            from aeon.distances import dtw_pairwise_distance
            distances = dtw_pairwise_distance(stacked, window=window)
        elif distance_metric == 'euclidean':
            from aeon.distances import euclidean_pairwise_distance
            distances = euclidean_pairwise_distance(stacked)
        elif distance_metric == 'squared':
            from aeon.distances import squared_pairwise_distance
            distances = squared_pairwise_distance(stacked)
        else:
            # Try to import the requested metric from aeon.distances
            import aeon.distances as aeon_dist
            pairwise_func = getattr(aeon_dist, f'{distance_metric}_pairwise_distance', None)
            if pairwise_func is None:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            if distance_metric == 'dtw':
                distances = pairwise_func(stacked, window=window)
            else:
                distances = pairwise_func(stacked)

        return distances[0, 1:].mean()

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

        dissonance_map = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                dissonance_map[y, x] = self.compute_position_dissonance(
                    x, y, all_series, kernel_size, distance_metric, window
                )
        return dissonance_map

    def __repr__(self):
        return (f"TVWall(video='{self.video_path}', width={self.width}, "
                f"height={self.height}, frames={self.num_frames}, "
                f"swapped={self.num_swapped})")
