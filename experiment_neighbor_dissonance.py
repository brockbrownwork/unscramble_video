#!/usr/bin/env python
# coding: utf-8
"""
Experiment: Neighbor Dissonance Heuristic Evaluation

Tests how well the neighbor dissonance metric identifies misplaced TVs.
For each position, computes average DTW distance to 8 neighbors.
High dissonance = likely misplaced.

Outputs:
- Precision/recall curves at different thresholds
- Heatmap visualization of dissonance scores
- ROC curve for the heuristic
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from tv_wall import TVWall


def parse_args():
    parser = argparse.ArgumentParser(description='Test neighbor dissonance heuristic')
    parser.add_argument('-f', '--frames', type=int, default=100,
                        help='Number of frames to use (default: 100)')
    parser.add_argument('-s', '--stride', type=int, default=30,
                        help='Frame stride (default: 30)')
    parser.add_argument('-n', '--num-swaps', type=int, default=20,
                        help='Number of TVs to swap (default: 20)')
    parser.add_argument('-v', '--video', type=str, default='cab_ride_trimmed.mkv',
                        help='Input video file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('-w', '--window', type=float, default=0.1,
                        help='DTW Sakoe-Chiba window as fraction (default: 0.1)')
    parser.add_argument('-m', '--metric', type=str, default='dtw',
                        choices=['dtw', 'euclidean', 'squared'],
                        help='Distance metric (default: dtw)')
    return parser.parse_args()


def compute_neighbor_dissonance_fast(wall, window=0.1, distance_metric='dtw'):
    """
    Compute neighbor dissonance for all positions using TVWall methods.
    """
    print("Precomputing all TV color series...")
    all_series = wall.get_all_series()

    print(f"Computing dissonance using {distance_metric} metric...")
    dissonance = wall.compute_dissonance_map(
        all_series=all_series,
        kernel_size=3,
        distance_metric=distance_metric,
        window=window
    )

    return dissonance


def get_swapped_mask(wall):
    """
    Create a binary mask where True = TV is not in original position.
    """
    height, width = wall.height, wall.width
    mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            orig_x, orig_y = wall.get_original_position(x, y)
            if orig_x != x or orig_y != y:
                mask[y, x] = True

    return mask


def plot_results(dissonance, swapped_mask, wall, output_prefix):
    """
    Create visualizations of the experiment results.
    """
    height, width = wall.height, wall.width

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Original frame
    ax = axes[0, 0]
    wall.reset_swaps()
    orig_frame = wall.get_frame_image(0)
    ax.imshow(orig_frame)
    ax.set_title('Original Frame')
    ax.axis('off')

    # Restore swaps for other visualizations
    # (We'll need to re-apply them)

    # 2. Scrambled frame (we need to reload the wall state)
    ax = axes[0, 1]
    ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))  # placeholder
    ax.set_title('Scrambled Frame (see separate)')
    ax.axis('off')

    # 3. Ground truth swap locations
    ax = axes[0, 2]
    ax.imshow(swapped_mask, cmap='Reds', interpolation='nearest')
    ax.set_title(f'Ground Truth: {swapped_mask.sum()} Swapped Positions')
    ax.axis('off')

    # 4. Dissonance heatmap
    ax = axes[1, 0]
    im = ax.imshow(dissonance, cmap='hot', interpolation='nearest')
    ax.set_title('Neighbor Dissonance Heatmap')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 5. Precision-Recall Curve
    ax = axes[1, 1]
    y_true = swapped_mask.flatten().astype(int)
    y_scores = dissonance.flatten()

    # Handle edge cases
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # Calculate AUC-PR
        auc_pr = auc(recall, precision)
        ax.text(0.6, 0.1, f'AUC-PR: {auc_pr:.3f}', transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax.set_title('Precision-Recall Curve')

    # 6. ROC Curve
    ax = axes[1, 2]
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax.set_title('ROC Curve')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_analysis.png")

    # Create overlay visualization
    create_overlay_visualization(dissonance, swapped_mask, output_prefix)


def create_overlay_visualization(dissonance, swapped_mask, output_prefix):
    """
    Create an overlay showing dissonance heatmap with swap locations marked.
    """
    height, width = dissonance.shape

    # Normalize dissonance to 0-1
    d_min, d_max = dissonance.min(), dissonance.max()
    if d_max > d_min:
        d_norm = (dissonance - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(dissonance)

    # Create RGB image: dissonance as red channel intensity
    overlay = np.zeros((height, width, 3))
    overlay[:, :, 0] = d_norm  # Red channel = dissonance

    # Mark swapped positions with green outline
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(overlay)

    # Find swap positions and mark them
    swap_y, swap_x = np.where(swapped_mask)
    ax.scatter(swap_x, swap_y, c='lime', s=50, marker='o',
               facecolors='none', edgecolors='lime', linewidths=2,
               label='Actual swaps')

    # Find top-K highest dissonance positions
    k = min(swapped_mask.sum() * 2, 50)  # 2x the actual swaps
    flat_indices = np.argsort(dissonance.flatten())[-k:]
    top_y, top_x = np.unravel_index(flat_indices, dissonance.shape)

    ax.scatter(top_x, top_y, c='cyan', s=30, marker='x', linewidths=1,
               label=f'Top {k} dissonance')

    ax.set_title('Dissonance Heatmap with Swap Locations')
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overlay.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_overlay.png")


def print_statistics(dissonance, swapped_mask):
    """Print statistics about detection performance."""
    y_true = swapped_mask.flatten()
    y_scores = dissonance.flatten()

    n_swapped = y_true.sum()
    n_total = len(y_true)

    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    print(f"Total positions: {n_total}")
    print(f"Swapped positions: {n_swapped} ({100*n_swapped/n_total:.1f}%)")

    # Dissonance stats for swapped vs non-swapped
    swapped_dissonance = y_scores[y_true.astype(bool)]
    normal_dissonance = y_scores[~y_true.astype(bool)]

    print(f"\nDissonance (swapped):     mean={swapped_dissonance.mean():.2f}, std={swapped_dissonance.std():.2f}")
    print(f"Dissonance (not swapped): mean={normal_dissonance.mean():.2f}, std={normal_dissonance.std():.2f}")

    # Separation ratio
    if normal_dissonance.std() > 0:
        separation = (swapped_dissonance.mean() - normal_dissonance.mean()) / normal_dissonance.std()
        print(f"Separation (z-score): {separation:.2f}")

    # Top-K accuracy
    for k_factor in [1, 2, 3]:
        k = n_swapped * k_factor
        top_k_indices = np.argsort(y_scores)[-k:]
        hits = y_true[top_k_indices].sum()
        print(f"Top-{k} ({k_factor}x swaps): {hits}/{n_swapped} swapped found ({100*hits/n_swapped:.1f}% recall)")

    print("="*50)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_prefix = f"dissonance_exp_n{args.num_swaps}_f{args.frames}_s{args.stride}_{args.metric}"

    print("="*50)
    print("NEIGHBOR DISSONANCE HEURISTIC EXPERIMENT")
    print("="*50)
    print(f"Video: {args.video}")
    print(f"Frames: {args.frames}, Stride: {args.stride}")
    print(f"Num swaps: {args.num_swaps}")
    print(f"Distance metric: {args.metric}")
    print(f"DTW window: {args.window}")
    print(f"Seed: {args.seed}")
    print()

    # Load video
    print("Loading video...")
    wall = TVWall(args.video, num_frames=args.frames, stride=args.stride)
    print(f"Wall size: {wall.width}x{wall.height} = {wall.num_tvs} TVs")
    print(f"Frames loaded: {wall.num_frames}")

    # Save original frame
    wall.save_frame(0, f"{output_prefix}_original.png")

    # Scramble some positions
    print(f"\nScrambling {args.num_swaps} positions...")
    wall.random_swaps(args.num_swaps, seed=args.seed)
    print(f"Actual swapped TVs: {wall.num_swapped}")

    # Save scrambled frame
    wall.save_frame(0, f"{output_prefix}_scrambled.png")

    # Get ground truth
    swapped_mask = get_swapped_mask(wall)

    # Compute neighbor dissonance
    dissonance = compute_neighbor_dissonance_fast(wall, window=args.window, distance_metric=args.metric)

    # Print statistics
    print_statistics(dissonance, swapped_mask)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_results(dissonance, swapped_mask, wall, output_prefix)

    # Save raw data
    np.save(f"{output_prefix}_dissonance.npy", dissonance)
    np.save(f"{output_prefix}_swapped_mask.npy", swapped_mask)
    print(f"Saved: {output_prefix}_dissonance.npy")
    print(f"Saved: {output_prefix}_swapped_mask.npy")

    print("\nDone!")


if __name__ == '__main__':
    main()
