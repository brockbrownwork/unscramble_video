#!/usr/bin/env python
# coding: utf-8

"""Test script for TVWall class."""

import os
import sys

# Add ffmpeg to PATH
ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

from tv_wall import TVWall

video_path = r"cab_ride_trimmed.mkv"

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "video_outputs")
os.makedirs(output_dir, exist_ok=True)

print("Creating TVWall with 100 frames...")
wall = TVWall(video_path, num_frames=100)
print(wall)

# Shuffle just 50 random positions
print("\nShuffling 50 random positions...")
wall.random_swaps(50, seed=42)
print(f"Number of swaps: {wall.num_swapped}")

# Save a single frame
wall.save_frame(0, os.path.join(output_dir, "partially_scrambled_frame.png"))

# Save video
print("\nSaving partially scrambled video...")
wall.save_video(os.path.join(output_dir, "partially_scrambled.mp4"), fps=30)

# Scramble 50% of positions
print("\nScrambling 50% of positions...")
wall.random_swaps(wall.num_tvs // 2, seed=99)
print(f"Number of swaps: {wall.num_swapped}")

print("\nSaving 50% scrambled video...")
wall.save_video(os.path.join(output_dir, "half_scrambled.mp4"), fps=30)

# Now fully scramble
print("\nFully scrambling all positions...")
wall.scramble(seed=123)
print(f"Number of swaps: {wall.num_swapped}")

# Save fully scrambled video
print("\nSaving fully scrambled video...")
wall.save_video(os.path.join(output_dir, "fully_scrambled.mp4"), fps=30)

print("\nDone!")
