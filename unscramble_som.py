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
#   python unscramble_som.py --dtw --landmarks 50  # Use DTW-based features

import argparse
import random
import time

import cv2
import numpy as np
from tqdm import tqdm
from minisom import MiniSom
from PIL import Image, ImageDraw
from fastdtw import fastdtw


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
    parser.add_argument('--dtw', action='store_true',
                        help='Use DTW-based landmark features instead of raw color sequences')
    parser.add_argument('--landmarks', type=int, default=50,
                        help='Number of landmark TVs for DTW features (default: 50)')
    parser.add_argument('--uniform-grid', action='store_true',
                        help='Place TVs on uniform grid (assign each TV to best available cell)')
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


def dtw_distance_grayscale(series1, series2):
    """
    Compute DTW distance using grayscale intensity instead of RGB.
    """
    s1 = series1.reshape(-1, 3)
    s2 = series2.reshape(-1, 3)

    gray1 = s1.mean(axis=1)
    gray2 = s2.mean(axis=1)

    distance, _ = fastdtw(gray1, gray2)
    return distance


def compute_dtw_landmark_features(tvs_array, num_landmarks):
    """
    Compute DTW distances from each TV to a set of landmark TVs.
    Returns a feature matrix where each row is the DTW distance profile to landmarks.
    """
    n = len(tvs_array)

    # Select random landmark indices
    landmark_indices = np.random.choice(n, size=num_landmarks, replace=False)
    landmarks = tvs_array[landmark_indices]

    print(f"Computing DTW distances to {num_landmarks} landmarks...")
    feature_matrix = np.zeros((n, num_landmarks))

    for i in tqdm(range(n), desc="Computing DTW landmark features"):
        for j, landmark in enumerate(landmarks):
            feature_matrix[i, j] = dtw_distance_grayscale(tvs_array[i], landmark)

    return feature_matrix


def assign_to_uniform_grid(som, features_normalized, grid_size):
    """
    Assign each TV to a unique grid cell, creating an evenly-spaced layout.
    Uses a greedy assignment: for each cell, find the unassigned TV with
    the smallest distance to that cell's weight vector.
    """
    from scipy.spatial.distance import cdist

    n_tvs = len(features_normalized)
    n_cells = grid_size * grid_size

    if n_tvs > n_cells:
        print(f"Warning: More TVs ({n_tvs}) than grid cells ({n_cells}). Some TVs will be excluded.")

    # Get all neuron weight vectors
    weights = som.get_weights().reshape(-1, features_normalized.shape[1])

    # Compute distance from each TV to each neuron
    distances = cdist(features_normalized, weights, metric='euclidean')

    # Greedy assignment: iterate through TVs sorted by their min distance to any cell
    assigned_cells = set()
    tv_assignments = {}  # tv_index -> (grid_x, grid_y)

    # For each TV, find its best unassigned cell
    tv_order = list(range(n_tvs))
    # Sort by how "desperate" each TV is (min distance to any cell)
    tv_order.sort(key=lambda i: distances[i].min())

    for tv_idx in tqdm(tv_order, desc="Assigning TVs to grid"):
        # Find best unassigned cell for this TV
        sorted_cells = np.argsort(distances[tv_idx])
        for cell_idx in sorted_cells:
            if cell_idx not in assigned_cells:
                assigned_cells.add(cell_idx)
                grid_x = cell_idx % grid_size
                grid_y = cell_idx // grid_size
                tv_assignments[tv_idx] = (grid_x, grid_y)
                break

    # Convert to coordinate lists (in original TV order)
    x_coords = []
    y_coords = []
    valid_tv_indices = []

    for i in range(n_tvs):
        if i in tv_assignments:
            x_coords.append(tv_assignments[i][0])
            y_coords.append(tv_assignments[i][1])
            valid_tv_indices.append(i)

    return np.array(x_coords), np.array(y_coords), valid_tv_indices


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
    dtw_suffix = f"_dtw{args.landmarks}" if args.dtw else ""
    base_filename = f"som_f{args.frames}_t{args.tvs}_s{args.stride}_g{args.grid}_i{args.iterations}{dtw_suffix}_{timestamp}"

    print(f"=== SOM Unscramble Experiment ===")
    print(f"Frames: {args.frames}, TVs: {args.tvs}, Stride: {args.stride}")
    print(f"Grid: {args.grid}x{args.grid}, Iterations: {args.iterations}")
    print(f"Sigma: {sigma}, Learning rate: {args.learning_rate}")
    if args.dtw:
        print(f"DTW mode: {args.landmarks} landmarks")
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

    del frames

    # Step 3: Prepare features (either raw or DTW-based)
    if args.dtw:
        print("Step 3: Computing DTW landmark features...")
        features = compute_dtw_landmark_features(tvs, args.landmarks)
        # Normalize DTW features
        features_normalized = (features - features.min()) / (features.max() - features.min())
    else:
        # Normalize raw TVs for SOM (important for training stability)
        features_normalized = (tvs - tvs.min()) / (tvs.max() - tvs.min())

    # Step 4: Train SOM
    print("Step 4: Training Self-Organizing Map...")
    print(f"Input dimension: {features_normalized.shape[1]}")

    som = MiniSom(
        args.grid, args.grid,
        features_normalized.shape[1],
        sigma=sigma,
        learning_rate=args.learning_rate,
        neighborhood_function='gaussian',
        random_seed=args.seed
    )

    # Initialize weights
    som.random_weights_init(features_normalized)

    # Train with progress bar
    for i in tqdm(range(args.iterations), desc="Training SOM"):
        idx = np.random.randint(len(features_normalized))
        som.update(features_normalized[idx], som.winner(features_normalized[idx]), i, args.iterations)

    # Step 5: Map TVs to SOM positions
    print("Step 5: Mapping TVs to SOM grid positions...")

    if args.uniform_grid:
        # Assign each TV to a unique grid cell for even spacing
        x_coordinates, y_coordinates, valid_indices = assign_to_uniform_grid(
            som, features_normalized, args.grid
        )
        # Filter TVs to only those that got assigned
        tvs = tvs[valid_indices]
    else:
        # Original behavior: place at BMU with jitter
        x_coordinates = []
        y_coordinates = []

        for i in tqdm(range(len(features_normalized)), desc="Finding BMUs"):
            winner = som.winner(features_normalized[i])
            jitter_x = (np.random.random() - 0.5) * args.jitter
            jitter_y = (np.random.random() - 0.5) * args.jitter
            x_coordinates.append(winner[0] + jitter_x)
            y_coordinates.append(winner[1] + jitter_y)

        x_coordinates = np.array(x_coordinates)
        y_coordinates = np.array(y_coordinates)

    # Step 6: Convert TVs to colors
    print("Step 6: Converting color data for visualization...")
    relevant_tvs = []
    for tv_data in tqdm(tvs, desc="Converting lists back to colors"):
        relevant_tvs.append(list_to_colors(tv_data.astype(int)))

    # Step 7: Create animation
    print("Step 7: Creating animation...")
    output_gif = f"{base_filename}.gif"
    create_color_animation(relevant_tvs, x_coordinates, y_coordinates, num_frames=10, output_filename=output_gif)

    # Step 8: Show some stats
    print()
    print(f"=== Done! ===")
    print(f"Output: {output_gif}")
    print(f"Quantization error: {som.quantization_error(features_normalized):.4f}")


if __name__ == '__main__':
    main()
