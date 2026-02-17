#!/usr/bin/env python
"""
generate_stimulus.py - Synthetic stimulus video generator.

Generates a video of randomly-colored balls floating across a screen
with a slowly shifting background hue. Designed to maximize per-pixel
color variety for testing the unscramble_video solver pipeline.

Usage:
    python generate_stimulus.py -o stimulus.mkv --frames 10000 --motion linear --seed 42
    python generate_stimulus.py --motion brownian --num-balls 200 --seed 42
"""

import argparse
import colorsys
import subprocess
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ── Hardcoded parameters ─────────────────────────────────────────────
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 10000
DEFAULT_FPS = 30
DEFAULT_OUTPUT = "stimulus.mkv"

# Ball spawning
MIN_BALL_RADIUS = 1
MAX_BALL_RADIUS = 80
MAX_BALLS = 2000

# Background
BG_HUE_SPEED = 0.00015      # hue shift per frame (cycles/frame)
BG_SATURATION = 0.3
BG_VALUE = 0.25

# Brownian motion
BROWNIAN_STEP_STD = 2.0      # pixels per frame standard deviation

# Linear motion
LINEAR_SPEED_MIN = 0.5
LINEAR_SPEED_MAX = 3.0


def generate_vibrant_color_pair(rng):
    """Generate two contrasting vibrant colors (center + edge) in BGR format.

    The edge color is offset by 0.3-0.7 in hue from the center to guarantee
    a wide, distinct gradient across each ball.
    """
    h1 = rng.uniform(0, 1)
    s1 = rng.uniform(0.7, 1.0)
    v1 = rng.uniform(0.8, 1.0)

    h2 = (h1 + rng.uniform(0.3, 0.7)) % 1.0
    s2 = rng.uniform(0.7, 1.0)
    v2 = rng.uniform(0.8, 1.0)

    r1, g1, b1 = colorsys.hsv_to_rgb(h1, s1, v1)
    r2, g2, b2 = colorsys.hsv_to_rgb(h2, s2, v2)

    center = np.array([int(b1 * 255), int(g1 * 255), int(r1 * 255)], dtype=np.uint8)
    edge = np.array([int(b2 * 255), int(g2 * 255), int(r2 * 255)], dtype=np.uint8)
    return center, edge


def draw_gradient_ball(canvas, cx, cy, radius, color_center, color_edge):
    """Draw a ball with radial gradient from center color to edge color."""
    if radius <= 2:
        # Too small for gradient, just draw flat
        cv2.circle(canvas, (cx, cy), radius,
                   tuple(int(c) for c in color_center), -1, cv2.LINE_AA)
        return

    h, w = canvas.shape[:2]
    # Bounding box clipped to canvas
    x0 = max(cx - radius, 0)
    y0 = max(cy - radius, 0)
    x1 = min(cx + radius + 1, w)
    y1 = min(cy + radius + 1, h)
    if x0 >= x1 or y0 >= y1:
        return

    # Pixel coordinates relative to center
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    # Normalized distance [0, 1], clamped
    t = np.clip(dist / radius, 0.0, 1.0)

    # Mask: pixels inside the circle
    mask = dist <= radius

    # Interpolate colors
    c_center = color_center.astype(np.float32)
    c_edge = color_edge.astype(np.float32)
    patch = canvas[y0:y1, x0:x1]
    for ch in range(3):
        blended = c_center[ch] * (1 - t) + c_edge[ch] * t
        patch[:, :, ch] = np.where(mask, blended.astype(np.uint8), patch[:, :, ch])


def compute_ball_radius(frame_idx, total_frames, decay_frac=0.95,
                        max_radius=MAX_BALL_RADIUS, min_radius=MIN_BALL_RADIUS):
    """Linearly decay ball radius from max to min over decay_frac of frames."""
    decay_end = decay_frac * total_frames
    t = min(frame_idx / decay_end, 1.0) if decay_end > 0 else 1.0
    return max(min_radius, int(max_radius - (max_radius - min_radius) * t))


def compute_background_color(frame_idx):
    """Compute slowly cycling background color in BGR."""
    h = (frame_idx * BG_HUE_SPEED) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, BG_SATURATION, BG_VALUE)
    return (int(b * 255), int(g * 255), int(r * 255))


def start_ffmpeg_encoder(width, height, fps, output_path):
    """Launch ffmpeg subprocess that reads raw BGR frames from stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg and ensure it's on your PATH.")
        sys.exit(1)
    return proc


def generate_stimulus(width, height, num_frames, fps, motion_mode, num_balls,
                      seed, output_path, spawn_rate=3.0, decay_end=0.95,
                      max_active=500):
    """Generate the stimulus video."""
    rng = np.random.default_rng(seed)

    # Ball state arrays
    positions = np.zeros((MAX_BALLS, 2), dtype=np.float64)
    velocities = np.zeros((MAX_BALLS, 2), dtype=np.float64)
    radii = np.zeros(MAX_BALLS, dtype=np.int32)
    # Two colors per ball: center and edge for gradient
    colors_center = np.zeros((MAX_BALLS, 3), dtype=np.uint8)
    colors_edge = np.zeros((MAX_BALLS, 3), dtype=np.uint8)
    active = np.zeros(MAX_BALLS, dtype=bool)
    num_active = 0
    next_slot = 0

    # Spawn accumulator: fractional spawn rates (e.g. 0.5 = one ball every 2 frames)
    spawn_accumulator = 0.0
    # FIFO queue for evicting oldest balls when at capacity
    from collections import deque
    active_queue = deque()

    # Canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Start encoder
    proc = start_ffmpeg_encoder(width, height, fps, output_path)

    ball_limit = num_balls if num_balls > 0 else None
    print(f"Generating {num_frames} frames at {width}x{height}, "
          f"balls={'unlimited' if ball_limit is None else ball_limit}, "
          f"spawn_rate={spawn_rate}/frame, motion={motion_mode}, seed={seed}")
    print(f"Output: {output_path}")

    try:
        for frame_idx in tqdm(range(num_frames), desc="Rendering", unit="frame"):
            # ── Spawn new balls ──
            spawn_accumulator += spawn_rate
            while spawn_accumulator >= 1.0:
                if ball_limit is not None and num_active >= ball_limit:
                    break
                spawn_accumulator -= 1.0

                # Evict oldest ball if at capacity
                if num_active >= max_active:
                    oldest = active_queue.popleft()
                    active[oldest] = False
                    num_active -= 1

                # Reuse evicted slot or allocate new one
                if next_slot < MAX_BALLS:
                    slot = next_slot
                    next_slot += 1
                else:
                    # Find a free slot from evicted balls
                    free = np.where(~active[:next_slot])[0]
                    if len(free) == 0:
                        break
                    slot = free[0]

                radius = compute_ball_radius(frame_idx, num_frames, decay_end)
                radii[slot] = radius
                colors_center[slot], colors_edge[slot] = generate_vibrant_color_pair(rng)
                active[slot] = True
                num_active += 1
                active_queue.append(slot)

                # Random position within bounds
                positions[slot, 0] = rng.uniform(radius, width - radius)
                positions[slot, 1] = rng.uniform(radius, height - radius)

                if motion_mode == "linear":
                    angle = rng.uniform(0, 2 * np.pi)
                    speed = rng.uniform(LINEAR_SPEED_MIN, LINEAR_SPEED_MAX)
                    velocities[slot, 0] = speed * np.cos(angle)
                    velocities[slot, 1] = speed * np.sin(angle)

            # ── Update positions ──
            active_mask = active[:next_slot]
            if active_mask.any():
                if motion_mode == "linear":
                    # Move
                    positions[:next_slot][active_mask] += velocities[:next_slot][active_mask]

                    # Bounce off edges
                    for i in np.where(active_mask)[0]:
                        r = radii[i]
                        if positions[i, 0] - r < 0:
                            positions[i, 0] = r
                            velocities[i, 0] = abs(velocities[i, 0])
                        elif positions[i, 0] + r > width:
                            positions[i, 0] = width - r
                            velocities[i, 0] = -abs(velocities[i, 0])
                        if positions[i, 1] - r < 0:
                            positions[i, 1] = r
                            velocities[i, 1] = abs(velocities[i, 1])
                        elif positions[i, 1] + r > height:
                            positions[i, 1] = height - r
                            velocities[i, 1] = -abs(velocities[i, 1])
                else:
                    # Brownian jitter
                    n_active = active_mask.sum()
                    jitter = rng.normal(0, BROWNIAN_STEP_STD, size=(n_active, 2))
                    positions[:next_slot][active_mask] += jitter

                    # Clamp to bounds
                    for i in np.where(active_mask)[0]:
                        r = radii[i]
                        positions[i, 0] = np.clip(positions[i, 0], r, width - r)
                        positions[i, 1] = np.clip(positions[i, 1], r, height - r)

            # ── Render frame ──
            bg = compute_background_color(frame_idx)
            canvas[:] = bg

            # Draw balls oldest first (newer/smaller on top)
            for i in range(next_slot):
                if active[i]:
                    cx = int(round(positions[i, 0]))
                    cy = int(round(positions[i, 1]))
                    draw_gradient_ball(canvas, cx, cy, int(radii[i]),
                                       colors_center[i], colors_edge[i])

            # Write to ffmpeg
            proc.stdin.write(canvas.tobytes())

    except BrokenPipeError:
        print("\nError: ffmpeg pipe broke unexpectedly.")
        sys.exit(1)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            print(f"ffmpeg exited with code {proc.returncode}")

    print(f"Done. Wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic stimulus video with floating colored balls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="Output video path")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help="Video width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help="Video height in pixels")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Frames per second")
    parser.add_argument("--motion", choices=["linear", "brownian"],
                        default="linear",
                        help="Ball motion mode")
    parser.add_argument("--num-balls", type=int, default=0,
                        help="Max number of balls (0 = no limit, keep spawning)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--spawn-rate", type=float, default=3.0,
                        help="New balls spawned per frame (e.g. 1.0 = one per frame, "
                             "0.5 = one every 2 frames, 3.0 = three per frame)")
    parser.add_argument("--max-active", type=int, default=500,
                        help="Max balls on screen at once (oldest evicted FIFO when full)")
    parser.add_argument("--decay-end", type=float, default=0.95,
                        help="Fraction of video over which ball radius shrinks from max to min "
                             "(0.5 = shrink fast, 1.0 = shrink over entire video)")
    return parser.parse_args()


def main():
    args = parse_args()

    generate_stimulus(
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        fps=args.fps,
        motion_mode=args.motion,
        num_balls=args.num_balls,
        seed=args.seed,
        output_path=args.output,
        spawn_rate=args.spawn_rate,
        decay_end=args.decay_end,
        max_active=args.max_active,
    )


if __name__ == "__main__":
    main()
