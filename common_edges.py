"""
Common Edge Detection across video frames.

Finds persistent/structural edges that appear across many frames,
robust to frames where those edges are absent (scene changes, motion, etc.).

Pipeline:
  1. Load all PNGs from a directory
  2. Run Canny edge detection on each frame
  3. Accumulate edge frequency (fraction of frames with an edge at each pixel)
  4. Optional Gaussian blur on accumulator to handle sub-pixel drift
  5. Threshold at a percentage (default 40%) to get persistent edges
  6. Output: binary edge image + frequency heatmap
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


def detect_common_edges(
    input_dir: str,
    threshold: float = 0.4,
    blur_sigma: float = 1.0,
    canny_low: int = 50,
    canny_high: int = 150,
    output_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find edges that persist across many frames.

    Args:
        input_dir:   Directory containing PNG frames.
        threshold:   Fraction of frames an edge must appear in (0.0-1.0).
        blur_sigma:  Gaussian blur sigma on the accumulator to handle
                     sub-pixel drift. Set to 0 to disable.
        canny_low:   Canny lower hysteresis threshold.
        canny_high:  Canny upper hysteresis threshold.
        output_dir:  Where to save results. Defaults to input_dir/../common_edges.

    Returns:
        (frequency_map, binary_edges)
        - frequency_map: float32 array [0,1], fraction of frames with edge
        - binary_edges: uint8 array {0,255}, thresholded persistent edges
    """
    # Discover frames
    patterns = [os.path.join(input_dir, f"*.{ext}") for ext in ("png", "PNG")]
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))
    if not paths:
        print(f"No PNG files found in {input_dir}")
        sys.exit(1)

    num_frames = len(paths)
    print(f"Found {num_frames} frames in {input_dir}")
    print(f"Threshold: {threshold:.0%} ({int(threshold * num_frames)} frames)")
    print(f"Canny thresholds: {canny_low}/{canny_high}")
    print(f"Blur sigma: {blur_sigma}" if blur_sigma > 0 else "Blur: disabled")

    # Read first frame to get dimensions
    sample = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    if sample is None:
        print(f"Failed to read {paths[0]}")
        sys.exit(1)
    h, w = sample.shape
    print(f"Frame size: {w}x{h}")

    # Accumulate edge detections
    accumulator = np.zeros((h, w), dtype=np.float64)

    for path in tqdm(paths, desc="Edge detection"):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  Warning: skipping unreadable file {path}")
            continue
        edges = cv2.Canny(gray, canny_low, canny_high)
        accumulator += (edges > 0).astype(np.float64)

    # Normalize to frequency [0, 1]
    frequency_map = (accumulator / num_frames).astype(np.float32)

    # Optional blur to account for sub-pixel edge drift between frames
    if blur_sigma > 0:
        # Kernel size must be odd, choose based on sigma
        ksize = int(np.ceil(blur_sigma * 3)) * 2 + 1
        frequency_map = cv2.GaussianBlur(frequency_map, (ksize, ksize), blur_sigma)

    # Threshold
    binary_edges = (frequency_map >= threshold).astype(np.uint8) * 255

    # Save results
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir.rstrip("/\\")), "common_edges")
    os.makedirs(output_dir, exist_ok=True)

    # Binary edge map
    binary_path = os.path.join(output_dir, "common_edges_binary.png")
    cv2.imwrite(binary_path, binary_edges)
    print(f"Saved binary edges: {binary_path}")

    # Frequency heatmap (colorized)
    heatmap_normalized = np.clip(frequency_map / max(frequency_map.max(), 1e-6), 0, 1)
    heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)
    heatmap_path = os.path.join(output_dir, "common_edges_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_color)
    print(f"Saved frequency heatmap: {heatmap_path}")

    # Raw frequency as 32-bit float for further processing
    raw_path = os.path.join(output_dir, "edge_frequency.npy")
    np.save(raw_path, frequency_map)
    print(f"Saved raw frequency map: {raw_path}")

    # Stats
    edge_pixels = np.count_nonzero(binary_edges)
    total_pixels = h * w
    print(f"\nResults:")
    print(f"  Persistent edge pixels: {edge_pixels:,} / {total_pixels:,} ({edge_pixels/total_pixels:.2%})")
    print(f"  Max edge frequency: {frequency_map.max():.2%}")
    print(f"  Mean edge frequency: {frequency_map.mean():.4%}")

    return frequency_map, binary_edges


def main():
    parser = argparse.ArgumentParser(
        description="Find persistent edges across video frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-dir",
        default=r"C:\Users\Brock\Documents\code\unscramble_video\bfs_output\result_frames",
        help="Directory containing PNG frames",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float, default=0.4,
        help="Fraction of frames an edge must appear in (0.0-1.0)",
    )
    parser.add_argument(
        "-b", "--blur-sigma",
        type=float, default=1.0,
        help="Gaussian blur sigma on accumulator for sub-pixel drift (0 to disable)",
    )
    parser.add_argument(
        "--canny-low",
        type=int, default=50,
        help="Canny lower hysteresis threshold",
    )
    parser.add_argument(
        "--canny-high",
        type=int, default=150,
        help="Canny upper hysteresis threshold",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: <input_dir>/../common_edges)",
    )

    args = parser.parse_args()

    detect_common_edges(
        input_dir=args.input_dir,
        threshold=args.threshold,
        blur_sigma=args.blur_sigma,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
