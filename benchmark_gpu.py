#!/usr/bin/env python
"""
Benchmark GPU vs CPU performance for TVWall operations.
"""

import time
import numpy as np
from tv_wall import TVWall
from gpu_utils import check_gpu, CUPY_AVAILABLE

# Configuration
VIDEO_PATH = "cab_ride_trimmed.mkv"  # Adjust to your video
NUM_FRAMES = 50
CROP_PERCENT = 30  # Use smaller crop for faster testing
NUM_POSITIONS = 500  # Number of positions to test batch dissonance


def benchmark_get_all_series(wall_gpu, wall_cpu, num_runs=10):
    """Benchmark get_all_series()."""
    print("\n--- Benchmarking get_all_series() ---")

    # Warmup
    _ = wall_gpu.get_all_series()
    _ = wall_cpu.get_all_series()

    # GPU timing
    if wall_gpu.gpu_enabled:
        start = time.perf_counter()
        for _ in range(num_runs):
            series = wall_gpu.get_all_series()
        # Sync GPU
        if CUPY_AVAILABLE:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        gpu_time = (time.perf_counter() - start) / num_runs
        print(f"  GPU: {gpu_time * 1000:.2f} ms per call")
    else:
        gpu_time = None
        print("  GPU: Not available")

    # CPU timing
    start = time.perf_counter()
    for _ in range(num_runs):
        series = wall_cpu.get_all_series()
    cpu_time = (time.perf_counter() - start) / num_runs
    print(f"  CPU: {cpu_time * 1000:.2f} ms per call")

    if gpu_time:
        speedup = cpu_time / gpu_time
        print(f"  Speedup: {speedup:.1f}x")

    return gpu_time, cpu_time


def benchmark_compute_position_dissonance(wall_gpu, wall_cpu, num_runs=100):
    """Benchmark compute_position_dissonance()."""
    print("\n--- Benchmarking compute_position_dissonance() ---")

    # Get series first
    series_gpu = wall_gpu.get_all_series() if wall_gpu.gpu_enabled else None
    series_cpu = wall_cpu.get_all_series()

    # Pick a central position
    x, y = wall_cpu.width // 2, wall_cpu.height // 2

    for metric in ['euclidean', 'squared', 'manhattan']:
        print(f"\n  Metric: {metric}")

        # GPU timing
        if wall_gpu.gpu_enabled:
            start = time.perf_counter()
            for _ in range(num_runs):
                d = wall_gpu.compute_position_dissonance(
                    x, y, series_gpu, distance_metric=metric
                )
            if CUPY_AVAILABLE:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            gpu_time = (time.perf_counter() - start) / num_runs
            print(f"    GPU: {gpu_time * 1000:.3f} ms per call")
        else:
            gpu_time = None
            print("    GPU: Not available")

        # CPU timing
        start = time.perf_counter()
        for _ in range(num_runs):
            d = wall_cpu.compute_position_dissonance(
                x, y, series_cpu, distance_metric=metric
            )
        cpu_time = (time.perf_counter() - start) / num_runs
        print(f"    CPU: {cpu_time * 1000:.3f} ms per call")

        if gpu_time:
            speedup = cpu_time / gpu_time
            print(f"    Speedup: {speedup:.1f}x")


def benchmark_compute_batch_dissonance(wall_gpu, wall_cpu, num_positions=500, num_runs=5):
    """Benchmark compute_batch_dissonance()."""
    print(f"\n--- Benchmarking compute_batch_dissonance() ({num_positions} positions) ---")

    # Generate random positions
    np.random.seed(42)
    all_positions = [(x, y) for y in range(wall_cpu.height) for x in range(wall_cpu.width)]
    np.random.shuffle(all_positions)
    positions = all_positions[:num_positions]

    series_gpu = wall_gpu.get_all_series() if wall_gpu.gpu_enabled else None
    series_cpu = wall_cpu.get_all_series()

    # Warmup GPU for all metrics
    if wall_gpu.gpu_enabled:
        for metric in ['euclidean', 'squared', 'manhattan']:
            _ = wall_gpu.compute_batch_dissonance(positions, series_gpu, distance_metric=metric)
        if CUPY_AVAILABLE:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

    for metric in ['euclidean', 'squared', 'manhattan']:
        print(f"\n  Metric: {metric}")

        # GPU timing
        if wall_gpu.gpu_enabled:
            if CUPY_AVAILABLE:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                result = wall_gpu.compute_batch_dissonance(
                    positions, series_gpu, distance_metric=metric
                )
            if CUPY_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
            gpu_time = (time.perf_counter() - start) / num_runs
            print(f"    GPU: {gpu_time * 1000:.1f} ms per call")
        else:
            gpu_time = None
            print("    GPU: Not available")

        # CPU timing
        start = time.perf_counter()
        for _ in range(num_runs):
            result = wall_cpu.compute_batch_dissonance(
                positions, series_cpu, distance_metric=metric
            )
        cpu_time = (time.perf_counter() - start) / num_runs
        print(f"    CPU: {cpu_time * 1000:.1f} ms per call")

        if gpu_time:
            speedup = cpu_time / gpu_time
            print(f"    Speedup: {speedup:.1f}x")


def benchmark_compute_dissonance_map(wall_gpu, wall_cpu, num_runs=3):
    """Benchmark compute_dissonance_map()."""
    print("\n--- Benchmarking compute_dissonance_map() ---")

    series_gpu = wall_gpu.get_all_series() if wall_gpu.gpu_enabled else None
    series_cpu = wall_cpu.get_all_series()

    for metric in ['euclidean']:
        print(f"\n  Metric: {metric}")

        # GPU timing
        if wall_gpu.gpu_enabled:
            start = time.perf_counter()
            for _ in range(num_runs):
                dmap = wall_gpu.compute_dissonance_map(
                    series_gpu, distance_metric=metric
                )
            if CUPY_AVAILABLE:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            gpu_time = (time.perf_counter() - start) / num_runs
            print(f"    GPU: {gpu_time * 1000:.1f} ms per call")
        else:
            gpu_time = None
            print("    GPU: Not available")

        # CPU timing
        start = time.perf_counter()
        for _ in range(num_runs):
            dmap = wall_cpu.compute_dissonance_map(
                series_cpu, distance_metric=metric
            )
        cpu_time = (time.perf_counter() - start) / num_runs
        print(f"    CPU: {cpu_time * 1000:.1f} ms per call")

        if gpu_time:
            speedup = cpu_time / gpu_time
            print(f"    Speedup: {speedup:.1f}x")


def main():
    print("=" * 60)
    print("GPU Acceleration Benchmark for TVWall")
    print("=" * 60)

    # Check GPU availability
    print("\nChecking GPU availability...")
    gpu_available = check_gpu()

    # Load video with GPU
    print(f"\nLoading video: {VIDEO_PATH}")
    print(f"  Frames: {NUM_FRAMES}, Crop: {CROP_PERCENT}%")

    try:
        wall_gpu = TVWall(VIDEO_PATH, num_frames=NUM_FRAMES, crop_percent=CROP_PERCENT, use_gpu=True)
        wall_cpu = TVWall(VIDEO_PATH, num_frames=NUM_FRAMES, crop_percent=CROP_PERCENT, use_gpu=False)
    except Exception as e:
        print(f"Error loading video: {e}")
        print("Make sure the video file exists.")
        return

    print(f"  Video size: {wall_cpu.width}x{wall_cpu.height}")
    print(f"  Total TVs: {wall_cpu.num_tvs}")
    print(f"  GPU enabled: {wall_gpu.gpu_enabled}")

    # Scramble both walls identically
    wall_gpu.scramble(seed=42)
    wall_cpu.scramble(seed=42)

    # Run benchmarks
    benchmark_get_all_series(wall_gpu, wall_cpu)
    benchmark_compute_position_dissonance(wall_gpu, wall_cpu)
    benchmark_compute_batch_dissonance(wall_gpu, wall_cpu, num_positions=NUM_POSITIONS)
    benchmark_compute_dissonance_map(wall_gpu, wall_cpu)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
