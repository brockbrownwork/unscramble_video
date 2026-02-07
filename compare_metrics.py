#!/usr/bin/env python
"""
Compare distance metrics for neighbor dissonance.

Computes dissonance heatmaps using 3 metrics side-by-side:
  - Euclidean distance
  - Cosine distance (1 - cosine similarity)
  - Negative dot product (raw dot product, negated so high = dissimilar)

Usage:
    python compare_metrics.py -v cab_ride_trimmed.mkv -f 100 -s 30 -n 20
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from tv_wall import TVWall


def parse_args():
    parser = argparse.ArgumentParser(description='Compare distance metrics for neighbor dissonance')
    parser.add_argument('-v', '--video', type=str, default='cab_ride_trimmed.mkv',
                        help='Input video file (default: cab_ride_trimmed.mkv)')
    parser.add_argument('-f', '--frames', type=int, default=100,
                        help='Number of frames to use (default: 100)')
    parser.add_argument('-s', '--stride', type=int, default=30,
                        help='Frame stride (default: 30)')
    parser.add_argument('-n', '--num-swaps', type=int, default=20,
                        help='Number of TVs to swap (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Save figure to file (default: show interactively)')
    return parser.parse_args()


def compute_dot_product_dissonance(wall, all_series, kernel_size=3):
    """
    Compute neighbor dissonance using negative dot product.

    For each position, computes the mean dot product with its neighbors,
    then negates it so that high values = dissimilar (consistent with
    other dissonance metrics).

    The dot product of two flattened color series measures how much they
    "agree" in absolute terms. Unlike cosine similarity, it is NOT
    normalized by magnitude — brighter pixels produce larger dot products
    regardless of pattern similarity.
    """
    height, width = wall.height, wall.width
    # Ensure CPU numpy array
    if hasattr(all_series, 'get'):
        all_series_np = all_series.get()
    elif not isinstance(all_series, np.ndarray):
        all_series_np = np.array(all_series)
    else:
        all_series_np = all_series

    all_series_np = all_series_np.astype(np.float64)
    flat_series = all_series_np.reshape(height, width, -1)
    radius = kernel_size // 2

    total_dots = np.zeros((height, width), dtype=np.float64)
    neighbor_counts = np.zeros((height, width), dtype=np.float64)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue

            c_x_start = max(0, -dx)
            c_x_end = min(width, width - dx)
            c_y_start = max(0, -dy)
            c_y_end = min(height, height - dy)
            n_x_start = c_x_start + dx
            n_x_end = c_x_end + dx
            n_y_start = c_y_start + dy
            n_y_end = c_y_end + dy

            center_flat = flat_series[c_y_start:c_y_end, c_x_start:c_x_end]
            neighbor_flat = flat_series[n_y_start:n_y_end, n_x_start:n_x_end]
            dots = np.sum(center_flat * neighbor_flat, axis=2)

            total_dots[c_y_start:c_y_end, c_x_start:c_x_end] += dots
            neighbor_counts[c_y_start:c_y_end, c_x_start:c_x_end] += 1.0

    # Negate: high dot product = similar, so negative = low dissonance
    dissonance_map = np.where(
        neighbor_counts > 0,
        -total_dots / neighbor_counts,
        0.0
    )
    return dissonance_map


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 50)
    print("METRIC COMPARISON")
    print("=" * 50)
    print(f"Video: {args.video}")
    print(f"Frames: {args.frames}, Stride: {args.stride}")
    print(f"Num swaps: {args.num_swaps}, Seed: {args.seed}")
    print()

    # Load video
    print("Loading video...")
    wall = TVWall(args.video, num_frames=args.frames, stride=args.stride)
    print(f"Wall size: {wall.width}x{wall.height} = {wall.num_tvs} TVs")

    # Scramble
    print(f"Scrambling {args.num_swaps} positions...")
    wall.random_swaps(args.num_swaps, seed=args.seed)
    print(f"Actual swapped TVs: {wall.num_swapped}")

    # Precompute all series once
    print("Precomputing all TV color series...")
    all_series = wall.get_all_series()

    # Compute dissonance for each metric
    metrics = {}

    print("\n--- Euclidean ---")
    metrics['Euclidean'] = wall.compute_dissonance_map(
        all_series=all_series, kernel_size=3, distance_metric='euclidean')

    print("--- Cosine ---")
    metrics['Cosine'] = wall.compute_dissonance_map(
        all_series=all_series, kernel_size=3, distance_metric='cosine')

    print("--- Dot Product (negated) ---")
    metrics['Dot Product (neg)'] = compute_dot_product_dissonance(
        wall, all_series, kernel_size=3)

    # Identify which positions are shuffled vs not
    identity_y, identity_x = np.mgrid[0:wall.height, 0:wall.width]
    is_shuffled = (wall._perm_x != identity_x) | (wall._perm_y != identity_y)

    num_shuffled = is_shuffled.sum()
    num_correct = (~is_shuffled).sum()
    print(f"\nPositions: {num_shuffled} shuffled, {num_correct} correct")

    # Print average dissonance + shuffled vs correct breakdown
    print("\n" + "=" * 60)
    print("DISSONANCE: SHUFFLED vs CORRECT POSITIONS")
    print("=" * 60)
    for name, dmap in metrics.items():
        shuffled_vals = dmap[is_shuffled]
        correct_vals = dmap[~is_shuffled]
        sep = (shuffled_vals.mean() - correct_vals.mean()) / np.sqrt(
            (shuffled_vals.std()**2 + correct_vals.std()**2) / 2) if correct_vals.std() > 0 else float('inf')
        print(f"  {name}:")
        print(f"    Shuffled  (n={len(shuffled_vals):5d}): mean={shuffled_vals.mean():.4f}  std={shuffled_vals.std():.4f}")
        print(f"    Correct   (n={len(correct_vals):5d}): mean={correct_vals.mean():.4f}  std={correct_vals.std():.4f}")
        print(f"    Separation (Cohen's d): {sep:.2f}")

        # Overlap analysis
        bins = np.linspace(dmap.min(), dmap.max(), 50)
        correct_counts, _ = np.histogram(correct_vals, bins=bins)
        shuffled_counts, _ = np.histogram(shuffled_vals, bins=bins)
        overlap_mask = (correct_counts > 0) & (shuffled_counts > 0)
        n_overlap_bins = overlap_mask.sum()
        if n_overlap_bins > 0:
            overlap_lo = bins[:-1][overlap_mask].min()
            overlap_hi = bins[1:][overlap_mask].max()
            n_correct_in = int(((correct_vals >= overlap_lo) & (correct_vals <= overlap_hi)).sum())
            n_shuffled_in = int(((shuffled_vals >= overlap_lo) & (shuffled_vals <= overlap_hi)).sum())
        else:
            n_correct_in = n_shuffled_in = 0
        print(f"    Overlap bins: {n_overlap_bins}/{len(bins)-1}")
        print(f"    Correct positions in overlap zone: {n_correct_in}")
        print(f"    Shuffled positions in overlap zone: {n_shuffled_in}")
        print(f"    Total positions in overlap zone: {n_correct_in + n_shuffled_in}")
        print()
    print("=" * 60)

    # Plot: row 1 = heatmaps, row 2 = histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Metric Comparison — {args.num_swaps} swaps, {args.frames} frames, stride {args.stride}',
                 fontsize=14, fontweight='bold')

    for col, (name, dmap) in enumerate(metrics.items()):
        # Row 1: heatmap
        ax_heat = axes[0, col]
        im = ax_heat.imshow(dmap, cmap='hot', interpolation='nearest')
        ax_heat.set_title(f'{name}\nmean={dmap.mean():.2f}', fontsize=11)
        ax_heat.axis('off')
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        # Row 2: histogram comparing shuffled vs correct
        ax_hist = axes[1, col]
        shuffled_vals = dmap[is_shuffled]
        correct_vals = dmap[~is_shuffled]

        bins = np.linspace(dmap.min(), dmap.max(), 50)
        ax_hist.hist(correct_vals, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_vals)})',
                     color='steelblue', density=True)
        ax_hist.hist(shuffled_vals, bins=bins, alpha=0.6, label=f'Shuffled (n={len(shuffled_vals)})',
                     color='tomato', density=True)
        ax_hist.axvline(correct_vals.mean(), color='steelblue', linestyle='--', linewidth=1.5)
        ax_hist.axvline(shuffled_vals.mean(), color='tomato', linestyle='--', linewidth=1.5)

        # Compute overlap: bins where both distributions have counts
        correct_counts, _ = np.histogram(correct_vals, bins=bins)
        shuffled_counts, _ = np.histogram(shuffled_vals, bins=bins)
        overlap_mask = (correct_counts > 0) & (shuffled_counts > 0)
        n_overlap_bins = overlap_mask.sum()

        # Shade overlap region
        if n_overlap_bins > 0:
            overlap_lo = bins[:-1][overlap_mask].min()
            overlap_hi = bins[1:][overlap_mask].max()
            ax_hist.axvspan(overlap_lo, overlap_hi, alpha=0.15, color='purple',
                            label='Overlap zone')

        # Count positions that fall in the overlap range
        n_correct_in_overlap = 0
        n_shuffled_in_overlap = 0
        if n_overlap_bins > 0:
            n_correct_in_overlap = int(((correct_vals >= overlap_lo) & (correct_vals <= overlap_hi)).sum())
            n_shuffled_in_overlap = int(((shuffled_vals >= overlap_lo) & (shuffled_vals <= overlap_hi)).sum())

        # Annotate on figure
        total_in_overlap = n_correct_in_overlap + n_shuffled_in_overlap
        ax_hist.text(0.98, 0.95,
                     f'Overlap bins: {n_overlap_bins}/{len(bins)-1}\n'
                     f'Correct in overlap: {n_correct_in_overlap}\n'
                     f'Shuffled in overlap: {n_shuffled_in_overlap}\n'
                     f'Total in overlap: {total_in_overlap}',
                     transform=ax_hist.transAxes, fontsize=8,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0e0ff', alpha=0.9))

        ax_hist.set_xlabel('Dissonance')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'{name} — Distribution', fontsize=11)
        ax_hist.legend(fontsize=9)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"\nSaved: {args.output}")
    else:
        plt.show()

    print("Done!")


if __name__ == '__main__':
    main()
