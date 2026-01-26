#!/usr/bin/env python
# coding: utf-8

# Unscramble Video using Self-Organizing Maps (SOMs)
#
# SOMs are more biologically plausible than UMAP/t-SNE - they learn through
# local, competitive, Hebbian-like rules similar to how cortical maps form.
# The topology emerges from local interactions, not global optimization.
#
# Usage:
#   python unscramble_som.py --frames 200 --tvs 500 --stride 60 --grid 50
#   python unscramble_som.py -f 100 -t 1000 -s 30 -g 30 --iterations 10000

import argparse
import random
import time

import cv2
import numpy as np
from tqdm import tqdm
from minisom import MiniSom
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description='Unscramble video using Self-Organizing Maps')
    parser.add_argument('-f', '--frames', type=int, default=200,
                        help='Number of frames to extract (default: 200)')
    parser.add_argument('-t', '--tvs', type=int, default=1000,
                        help='Number of TVs/pixels to sample (default: 1000)')
    parser.add_argument('-s', '--stride', type=int, default=60,
                        help='Frame skip interval (default: 60 for 1 fps at 60fps video)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting frame number (default: 0)')
    parser.add_argument('-g', '--grid', type=int, default=50,
                        help='SOM grid size (grid x grid neurons) (default: 50)')
    parser.add_argument('-i', '--iterations', type=int, default=20000,
                        help='Number of SOM training iterations (default: 20000)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='SOM neighborhood radius (default: grid/2)')
    parser.add_argument('--learning-rate', type=float, default=0.5,
                        help='SOM initial learning rate (default: 0.5)')
    parser.add_argument('-v', '--video', type=str, default='cab_ride_trimmed.mkv',
                        help='Input video file (default: cab_ride_trimmed.mkv)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--jitter', type=float, default=0.3,
                        help='Jitter amount for TVs mapped to same neuron (default: 0.3)')
    return parser.parse_args()


def sequential_frames_from_video(video_path, num_frames=200, start=0, stride=1):
    """
    Extracts sequential frames from a video starting at a given frame.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return []

    frames_span = num_frames * stride
    if start + frames_span > total_frames:
        start = max(0, total_frames - frames_span)
        print(f"Adjusted start frame to {start} to fit {num_frames} frames with stride {stride}.")

    print(f"Extracting {num_frames} frames from {start} to {start + frames_span} (stride={stride})")

    frames = []
    for i in tqdm(range(num_frames), desc="Loading sequential video frames"):
        frame_idx = start + (i * stride)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()
    return np.array(frames)


def list_to_colors(input_list):
    if len(input_list) % 3 != 0:
        raise ValueError("The length of the list must be a multiple of 3.")
    return [tuple(input_list[i:i + 3]) for i in range(0, len(input_list), 3)]


def normalize_list(input_list):
    min_val = min(input_list)
    max_val = max(input_list)
    if max_val == min_val:
        return [0.5 for _ in input_list]
    return [(x - min_val) / (max_val - min_val) for x in input_list]


def create_color_animation(color_list, x_list, y_list, output_filename='animation.gif', num_frames=None):
    img_size = (500, 500)
    point_radius = 3
    background_color = (255, 255, 255)
    x_list, y_list = normalize_list(x_list), normalize_list(y_list)

    if not num_frames:
        num_frames = len(color_list[0])

    images = []

    for frame in tqdm(range(num_frames), desc="Creating animation"):
        img = Image.new('RGB', img_size, background_color)
        draw = ImageDraw.Draw(img)

        for i in range(len(x_list)):
            x = int(x_list[i] * (img_size[0] - 2 * point_radius)) + point_radius
            y = int(y_list[i] * (img_size[1] - 2 * point_radius)) + point_radius
            color = tuple(map(int, color_list[i][frame]))
            draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=color)

        images.append(img)

    images[0].save(output_filename, save_all=True, append_images=images[1:], duration=300, loop=0)
    print(f"GIF created and saved as {output_filename}")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Default sigma to half grid size
    sigma = args.sigma if args.sigma else args.grid / 2

    # Generate output filename with parameters and timestamp
    timestamp = int(time.time())
    base_filename = f"som_f{args.frames}_t{args.tvs}_s{args.stride}_g{args.grid}_i{args.iterations}_{timestamp}"

    print(f"=== SOM Unscramble Experiment ===")
    print(f"Frames: {args.frames}, TVs: {args.tvs}, Stride: {args.stride}")
    print(f"Grid: {args.grid}x{args.grid}, Iterations: {args.iterations}")
    print(f"Sigma: {sigma}, Learning rate: {args.learning_rate}")
    print(f"Output: {base_filename}.gif")
    print()

    # Step 1: Load frames
    print("Step 1: Loading sequential video frames...")
    frames = sequential_frames_from_video(args.video, num_frames=args.frames, start=args.start, stride=args.stride)
    width = frames.shape[1]
    height = frames.shape[2]
    random_indices = np.random.permutation(width * height)

    # Step 2: Build TV list
    print("Step 2: Building TV color series...")

    def tv(position):
        position = random_indices[position]
        x_position = position % width
        y_position = position // width
        return frames[:, x_position, y_position]

    tvs = []
    for i in tqdm(range(args.tvs), desc="Flattening tvs"):
        tvs.append(tv(i).flatten().astype(np.float32))

    tvs = np.array(tvs)

    # Normalize TVs for SOM (important for training stability)
    tvs_normalized = (tvs - tvs.min()) / (tvs.max() - tvs.min())

    del frames

    # Step 3: Train SOM
    print("Step 3: Training Self-Organizing Map...")
    print(f"Input dimension: {tvs_normalized.shape[1]}")

    som = MiniSom(
        args.grid, args.grid,
        tvs_normalized.shape[1],
        sigma=sigma,
        learning_rate=args.learning_rate,
        neighborhood_function='gaussian',
        random_seed=args.seed
    )

    # Initialize weights
    som.random_weights_init(tvs_normalized)

    # Train with progress bar
    for i in tqdm(range(args.iterations), desc="Training SOM"):
        idx = np.random.randint(len(tvs_normalized))
        som.update(tvs_normalized[idx], som.winner(tvs_normalized[idx]), i, args.iterations)

    # Step 4: Map TVs to SOM positions
    print("Step 4: Mapping TVs to SOM grid positions...")
    x_coordinates = []
    y_coordinates = []

    for i in tqdm(range(len(tvs_normalized)), desc="Finding BMUs"):
        winner = som.winner(tvs_normalized[i])
        # Add jitter to avoid all TVs at same neuron overlapping
        jitter_x = (np.random.random() - 0.5) * args.jitter
        jitter_y = (np.random.random() - 0.5) * args.jitter
        x_coordinates.append(winner[0] + jitter_x)
        y_coordinates.append(winner[1] + jitter_y)

    x_coordinates = np.array(x_coordinates)
    y_coordinates = np.array(y_coordinates)

    # Step 5: Convert TVs to colors
    print("Step 5: Converting color data for visualization...")
    relevant_tvs = []
    for tv_data in tqdm(tvs, desc="Converting lists back to colors"):
        relevant_tvs.append(list_to_colors(tv_data.astype(int)))

    # Step 6: Create animation
    print("Step 6: Creating animation...")
    output_gif = f"{base_filename}.gif"
    create_color_animation(relevant_tvs, x_coordinates, y_coordinates, num_frames=10, output_filename=output_gif)

    # Step 7: Show some stats
    print()
    print(f"=== Done! ===")
    print(f"Output: {output_gif}")
    print(f"Quantization error: {som.quantization_error(tvs_normalized):.4f}")


if __name__ == '__main__':
    main()
