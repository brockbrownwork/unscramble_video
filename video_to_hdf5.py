"""Extract video frames into an HDF5 dataset with shape (T, C, H, W).

Optimized for fast pixel time-series access: dataset[:, :, y, x] reads
the full color history of a single pixel in ~16 chunk reads.

Usage:
    python video_to_hdf5.py cab_ride_trimmed.mkv --output videos.h5 --dataset "videos/cab_ride_trimmed"
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import h5py
import numpy as np
from tqdm import tqdm


BATCH_SIZE = 1000


def extract_frames(video_path, output_dir):
    """Extract all frames from video as BMP files using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        os.path.join(output_dir, "%06d.bmp"),
    ]
    print(f"Extracting frames: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)


def get_sorted_frame_paths(frame_dir):
    """Return sorted list of absolute BMP paths."""
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".bmp"))
    return [os.path.join(frame_dir, f) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Convert video to HDF5 dataset (T, C, H, W)")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--output", "-o", default="videos.h5", help="Output HDF5 file (default: videos.h5)")
    parser.add_argument("--dataset", "-d", required=True, help='Dataset path inside HDF5 (e.g. "videos/cab_ride_trimmed")')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Frames to load per batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}")
        sys.exit(1)

    # Extract frames to a temp directory
    frame_dir = tempfile.mkdtemp(prefix="hdf5_frames_")
    print(f"Temp frame directory: {frame_dir}")

    try:
        extract_frames(args.video, frame_dir)
        frame_paths = get_sorted_frame_paths(frame_dir)
        T = len(frame_paths)
        print(f"Extracted {T} frames")

        if T == 0:
            print("Error: no frames extracted")
            sys.exit(1)

        # Read first frame to get dimensions
        sample = cv2.imread(frame_paths[0])
        H, W, _ = sample.shape
        print(f"Frame dimensions: {W}x{H} (W x H)")
        print(f"Dataset shape: ({T}, 3, {H}, {W})")
        print(f"Dataset size: {T * 3 * H * W / 1e9:.2f} GB")

        # Chunk temporal size = batch size for aligned writes
        chunk_t = min(args.batch_size, T)
        chunks = (chunk_t, 3, 8, 8)
        print(f"Chunk shape: {chunks}")

        # Create/open HDF5 file and dataset
        with h5py.File(args.output, "a") as f:
            if args.dataset in f:
                print(f"Warning: dataset '{args.dataset}' already exists, overwriting")
                del f[args.dataset]

            ds = f.create_dataset(
                args.dataset,
                shape=(T, 3, H, W),
                dtype=np.uint8,
                chunks=chunks,
            )

            # Process frames in batches
            num_batches = (T + args.batch_size - 1) // args.batch_size
            for batch_idx in tqdm(range(num_batches), desc="Writing batches"):
                start = batch_idx * args.batch_size
                end = min(start + args.batch_size, T)
                batch_paths = frame_paths[start:end]
                batch_len = end - start

                # Load batch into memory: (batch_len, C, H, W)
                batch = np.empty((batch_len, 3, H, W), dtype=np.uint8)
                for i, path in enumerate(batch_paths):
                    img = cv2.imread(path)  # BGR, (H, W, 3)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
                    batch[i] = img.transpose(2, 0, 1)  # (3, H, W)

                # Write batch to HDF5
                ds[start:end] = batch

                # Delete BMP files for this batch
                for path in batch_paths:
                    os.remove(path)

            print(f"Done. Dataset '{args.dataset}' written to '{args.output}'")
            print(f"Shape: {ds.shape}, dtype: {ds.dtype}, chunks: {ds.chunks}")

    finally:
        # Clean up temp directory
        if os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
