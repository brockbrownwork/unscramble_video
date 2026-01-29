#!/usr/bin/env python
# coding: utf-8

"""Test script for TVWall class."""

import os
import sys

# Add ffmpeg to PATH
ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

from tv_wall import TVWall

# Use video from main repo
video_path = r"C:\Users\Brock\Documents\code\unscramble_video\cab_ride_trimmed.mkv"

print("Creating TVWall with 100 frames...")
wall = TVWall(video_path, num_frames=100)
print(wall)

# Shuffle just 50 random positions
print("\nShuffling 50 random positions...")
wall.random_swaps(50, seed=42)
print(f"Number of swaps: {len(wall.swaps)}")

# Save a single frame
wall.save_frame(0, "partially_scrambled_frame.png")

# Save video
print("\nSaving partially scrambled video...")
wall.save_video("partially_scrambled.mp4", fps=30)

# Now fully scramble
print("\nFully scrambling all positions...")
wall.scramble(seed=123)
print(f"Number of swaps: {len(wall.swaps)}")

# Save fully scrambled video
print("\nSaving fully scrambled video...")
wall.save_video("fully_scrambled.mp4", fps=30)

print("\nDone!")
