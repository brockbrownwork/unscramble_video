"""Rechunk an HDF5 dataset from (1000, 3, 8, 8) to (T, 3, 1, 1) for fast pixel time-series access.

Reads the source dataset in spatial batches (row strips), writes each pixel's full
time-series contiguously into the new dataset. Deletes the old dataset and renames
the new one in place.

Usage:
    python ONESHOT_rechunk_h5.py videos.h5 --dataset "videos/cab_ride_trimmed"
"""

import argparse
import sys

import h5py
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Rechunk HDF5 dataset to (T, 3, 1, 1)")
    parser.add_argument("h5file", help="HDF5 file to rechunk")
    parser.add_argument("--dataset", "-d", required=True, help='Dataset path (e.g. "videos/cab_ride_trimmed")')
    parser.add_argument("--row-batch", type=int, default=8, help="Rows to process per batch (default: 8)")
    args = parser.parse_args()

    with h5py.File(args.h5file, "a") as f:
        if args.dataset not in f:
            print(f"Error: dataset '{args.dataset}' not found")
            sys.exit(1)

        src = f[args.dataset]
        T, C, H, W = src.shape
        print(f"Source: shape={src.shape}, chunks={src.chunks}, dtype={src.dtype}")

        new_chunks = (T, C, 1, 1)
        print(f"Target chunks: {new_chunks}")
        print(f"Chunk size: {T * C:.0f} bytes ({T * C / 1024:.1f} KB)")
        print(f"Total chunks: {H * W:,} ({H} x {W})")

        tmp_name = args.dataset + "_rechunked"
        if tmp_name in f:
            del f[tmp_name]

        dst = f.create_dataset(
            tmp_name,
            shape=(T, C, H, W),
            dtype=np.uint8,
            chunks=new_chunks,
        )

        # Process in row strips to balance memory vs speed.
        # Each batch reads (T, C, row_batch, W) from source and writes to dest.
        num_batches = (H + args.row_batch - 1) // args.row_batch
        for batch_idx in tqdm(range(num_batches), desc="Rechunking rows"):
            y0 = batch_idx * args.row_batch
            y1 = min(y0 + args.row_batch, H)
            # Read a strip of rows across all frames
            strip = src[:, :, y0:y1, :]  # (T, C, batch_h, W)
            dst[:, :, y0:y1, :] = strip

        # Replace old dataset with new one
        print(f"Replacing '{args.dataset}' with rechunked version...")
        del f[args.dataset]
        f.move(tmp_name, args.dataset)

        ds = f[args.dataset]
        print(f"Done. shape={ds.shape}, chunks={ds.chunks}, dtype={ds.dtype}")


if __name__ == "__main__":
    main()
