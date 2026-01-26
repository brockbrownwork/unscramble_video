#!/usr/bin/env python
# coding: utf-8

# Unscramble Video using DTW (Dynamic Time Warping) + UMAP
#
# This experiment uses DTW as the distance metric instead of Euclidean distance.
# DTW can capture temporal relationships between pixel color sequences, accounting
# for time-shifts that occur when edges/objects move across adjacent pixels.

import os
import random
import time

import cv2
import numpy as np
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.display import Image as DisplayImage
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# Parameters
number_of_frames = 200  # Sequential frames for proper DTW temporal analysis
number_of_tvs = 500  # Small number to make O(n^2) feasible
n_neighbors = 50
start_frame = 0  # Starting frame for sequential extraction
stride = 60  # One frame per second (video is 60 FPS)


np.random.seed(1)


def sequential_frames_from_video(video_path, num_frames=200, start=0, stride=1):
    """
    Extracts sequential frames from a video starting at a given frame.

    Parameters:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
        start (int): Starting frame number.
        stride (int): Frame skip interval (1=every frame, 2=every other, etc.)

    Returns:
        np.array: NumPy array of frames.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return []

    # Calculate frames needed with stride
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
    frames = np.array(frames)

    return frames


def dtw_distance(series1, series2):
    """
    Compute DTW distance between two color series.

    Each series is flattened (R,G,B,R,G,B,...), so we reshape back to (n_frames, 3)
    and compute DTW on the color trajectories.
    """
    # Reshape from flat to (n_frames, 3) for RGB
    s1 = series1.reshape(-1, 3)
    s2 = series2.reshape(-1, 3)

    distance, _ = fastdtw(s1, s2, dist=euclidean)
    return distance


def dtw_distance_grayscale(series1, series2):
    """
    Compute DTW distance using grayscale intensity instead of RGB.
    This is faster and may work just as well for detecting temporal patterns.
    """
    # Reshape from flat to (n_frames, 3) for RGB
    s1 = series1.reshape(-1, 3)
    s2 = series2.reshape(-1, 3)

    # Convert to grayscale (simple average)
    gray1 = s1.mean(axis=1)
    gray2 = s2.mean(axis=1)

    distance, _ = fastdtw(gray1, gray2)
    return distance


# Extract sequential frames from the video
print("Step 1: Loading sequential video frames...")
frames = sequential_frames_from_video('cab_ride_trimmed.mkv', num_frames=number_of_frames, start=start_frame, stride=stride)
width = frames.shape[1]
height = frames.shape[2]
random_indices = np.random.permutation(width * height)


# Preview frames
def display_random_frames():
    display_frames = frames[:5]

    fig, axs = plt.subplots(1, len(display_frames), figsize=(20, 5))

    for ax, frame in zip(axs, display_frames):
        ax.imshow(frame)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

print("Step 2: Displaying sample frames...")
display_random_frames()


# Build TV list
def tv(position):
    position = random_indices[position]
    x_position = position % width
    y_position = position // width
    return frames[:, x_position, y_position]

print("Step 3: Building TV color series...")
tvs = []
for i in tqdm(range(number_of_tvs), desc="Flattening tvs and turning them into a list"):
    tvs.append(tv(i).flatten())

tvs = np.array(tvs)

del frames


# Precompute DTW distance matrix
print("Step 4: Precomputing DTW distance matrix...")
print(f"Computing {number_of_tvs * (number_of_tvs - 1) // 2} pairwise distances...")

def compute_distance_matrix(tvs_array):
    """
    Precompute pairwise DTW distances for all TVs.
    Returns a symmetric distance matrix.
    """
    n = len(tvs_array)
    dist_matrix = np.zeros((n, n))

    total_comparisons = (n * (n - 1)) // 2

    with tqdm(total=total_comparisons, desc="Computing DTW distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                d = dtw_distance_grayscale(tvs_array[i], tvs_array[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                pbar.update(1)

    return dist_matrix

distance_matrix = compute_distance_matrix(tvs)

print("Step 5: Running UMAP with precomputed distance matrix...")

reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.1,
    n_components=2,
    metric='precomputed',
    verbose=True
)

embedding = reducer.fit_transform(distance_matrix)
x_coordinates, y_coordinates = embedding[:, 0], embedding[:, 1]


# Convert TVs back to color format for visualization
def list_to_colors(input_list):
    if len(input_list) % 3 != 0:
        raise ValueError("The length of the list must be a multiple of 3.")
    return [tuple(input_list[i:i + 3]) for i in range(0, len(input_list), 3)]

print("Step 6: Converting color data for visualization...")
relevant_tvs = []
for tv_data in tqdm(tvs, desc="Converting lists back to colors"):
    relevant_tvs.append(list_to_colors(tv_data))


# Animation functions
def normalize_list(input_list):
    min_val = min(input_list)
    max_val = max(input_list)
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
            x = int(x_list[i] * (img_size[0]))
            y = int(y_list[i] * (img_size[1]))
            color = tuple(map(int, color_list[i][frame]))
            draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=color)

        images.append(img)

    images[0].save(output_filename, save_all=True, append_images=images[1:], duration=300, loop=0)

    print(f"GIF created and saved as {output_filename}")


print("Step 7: Creating animation...")
create_color_animation(relevant_tvs, x_coordinates, y_coordinates, num_frames=10, output_filename='animation_dtw.gif')

DisplayImage(filename='animation_dtw.gif')


# Optional: Apply perspective correction
from quadrilateral_fitter import QuadrilateralFitter

def apply_perspective_correction(x_coords, y_coords):
    points = np.array(list(zip(x_coords, y_coords)))
    fitter = QuadrilateralFitter(polygon=points)
    fitted_quadrilateral = fitter.fit()

    x_fitted = [point[0] for point in fitted_quadrilateral]
    y_fitted = [point[1] for point in fitted_quadrilateral]

    return x_fitted, y_fitted

print("Step 8: Applying perspective correction...")
x_transformed, y_transformed = apply_perspective_correction(x_coordinates, y_coordinates)

create_color_animation(relevant_tvs, x_transformed, y_transformed, num_frames=10, output_filename='perspective_fixed_animation_dtw.gif')

DisplayImage(filename='perspective_fixed_animation_dtw.gif')
