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
        swaps (dict): Maps original TV positions to their current positions.
                      Key: (orig_x, orig_y) -> Value: (new_x, new_y)
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
        self.swaps = {}

        self._load_video(num_frames)

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
        # Check if any TV was swapped to this position
        for orig_pos, new_pos in self.swaps.items():
            if new_pos == (x, y):
                return orig_pos
        return (x, y)

    def get_current_position(self, orig_x, orig_y):
        """
        Get the current position of a TV that was originally at (orig_x, orig_y).

        Parameters:
            orig_x (int): Original x position.
            orig_y (int): Original y position.

        Returns:
            tuple: (cur_x, cur_y) - the current position of this TV.
        """
        return self.swaps.get((orig_x, orig_y), (orig_x, orig_y))

    def swap(self, orig_pos, new_pos):
        """
        Swap a TV from its original position to a new position.

        Parameters:
            orig_pos (tuple): Original position (x, y) of the TV.
            new_pos (tuple): New position (x, y) where the TV should be placed.
        """
        self.swaps[orig_pos] = new_pos

    def swap_positions(self, pos1, pos2):
        """
        Swap the TVs at two positions with each other.

        Parameters:
            pos1 (tuple): First position (x, y).
            pos2 (tuple): Second position (x, y).
        """
        # Find what original TVs are currently at these positions
        orig1 = self.get_original_position(*pos1)
        orig2 = self.get_original_position(*pos2)

        # Swap their positions
        self.swaps[orig1] = pos2
        self.swaps[orig2] = pos1

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

        # Clamp to total number of TVs
        num_positions = min(num_positions, self.num_tvs)

        # Create list of all positions
        all_positions = [(x, y) for y in range(self.height) for x in range(self.width)]

        # Select random positions to shuffle
        selected_indices = np.random.choice(len(all_positions), size=num_positions, replace=False)
        selected_positions = [all_positions[i] for i in selected_indices]

        # Shuffle only the selected positions among themselves
        shuffled = selected_positions.copy()
        np.random.shuffle(shuffled)

        # Reset and create new swaps mapping
        self.swaps = {}
        for orig, new in zip(selected_positions, shuffled):
            if orig != new:
                self.swaps[orig] = new

    def reset_swaps(self):
        """Reset all swaps, returning TVs to their original positions."""
        self.swaps = {}

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

        # Create output image array
        output = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Fill in pixel values based on swaps
        for y in range(self.height):
            for x in range(self.width):
                orig_x, orig_y = self.get_original_position(x, y)
                output[y, x] = self._frames[timestep, orig_y, orig_x]

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

    def __repr__(self):
        return (f"TVWall(video='{self.video_path}', width={self.width}, "
                f"height={self.height}, frames={self.num_frames}, "
                f"swaps={len(self.swaps)})")
