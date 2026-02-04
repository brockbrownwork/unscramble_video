"""
Animated Pixel Shuffle Visualization

Creates a video/GIF showing pixels being shuffled in real-time as the video plays.
The viewer sees the video playing while pixels get "plucked" and moved to new positions.
"""

import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm


def load_video_frames(video_path, num_frames=None, start_frame=0, scale=1.0):
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_path)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if num_frames is None:
        num_frames = total_frames - start_frame

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if scale != 1.0:
            h, w = frame.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        frames.append(frame)

    cap.release()
    return np.array(frames), fps


def ease_in_out_cubic(t):
    """Cubic ease-in-out for smooth animation."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def draw_circle(frame, center_x, center_y, radius, color, thickness=2):
    """Draw a circle on the frame using cv2."""
    cv2.circle(frame, (int(center_x), int(center_y)), int(radius), color, thickness)


def create_swap_animation(frames, swap_pairs, swap_schedule, output_path="shuffle_animation.gif",
                          pixel_scale=8, swap_duration_frames=15, fps=30, highlight_frames=20):
    """
    Create an animation showing pixels being swapped as the video plays.

    Args:
        frames: numpy array of video frames (num_frames, H, W, 3)
        swap_pairs: list of ((x1, y1), (x2, y2)) tuples - which pixels to swap
        swap_schedule: list of frame indices when each swap starts
        output_path: where to save the animation
        pixel_scale: how much to scale up each pixel for visibility
        swap_duration_frames: how many output frames the swap animation takes
        fps: output frame rate
        highlight_frames: how many frames to show red circles before swap starts
    """
    num_frames, orig_h, orig_w, _ = frames.shape

    # Scale up the frames so individual pixels are visible
    scaled_h = orig_h * pixel_scale
    scaled_w = orig_w * pixel_scale

    # Track current permutation (which original pixel is at each position)
    # perm[y, x] = (orig_y, orig_x)
    perm = np.zeros((orig_h, orig_w, 2), dtype=np.int32)
    for y in range(orig_h):
        for x in range(orig_w):
            perm[y, x] = [y, x]

    # Track active swaps with phases: 'highlight' -> 'swapping' -> 'completed'
    active_swaps = []
    swap_idx = 0

    output_frames = []

    # Calculate total output frames needed
    # The animation should end when the last swap completes
    if swap_pairs and swap_schedule:
        last_swap_start = swap_schedule[-1]
        last_swap_end = last_swap_start + highlight_frames + swap_duration_frames
        total_output_frames = max(num_frames, last_swap_end)
    else:
        total_output_frames = num_frames

    print(f"Creating animation: {orig_w}x{orig_h} pixels, scaled to {scaled_w}x{scaled_h}")
    print(f"Video frames: {num_frames}, Output frames: {total_output_frames}")
    print(f"Swaps scheduled: {len(swap_pairs)}")

    for out_frame_idx in tqdm(range(total_output_frames), desc="Rendering"):
        # Which video frame to show (clamp to last frame)
        video_frame_idx = min(out_frame_idx, num_frames - 1)
        video_frame = frames[video_frame_idx]

        # Check if we should start a new swap (begins with highlight phase)
        while swap_idx < len(swap_schedule) and swap_schedule[swap_idx] <= out_frame_idx:
            pos1, pos2 = swap_pairs[swap_idx]
            active_swaps.append({
                'start_frame': out_frame_idx,
                'pos1': pos1,
                'pos2': pos2,
                'phase': 'highlight',  # 'highlight' -> 'swapping' -> 'completed'
            })
            swap_idx += 1

        # Create the output frame
        output_frame = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)

        # Set of positions currently being animated (don't draw them in the grid)
        # Only exclude during 'swapping' phase, not during 'highlight'
        animating_positions = set()
        for swap in active_swaps:
            if swap['phase'] == 'swapping':
                animating_positions.add(swap['pos1'])
                animating_positions.add(swap['pos2'])

        # Draw each pixel at its current position (scaled up)
        for y in range(orig_h):
            for x in range(orig_w):
                if (x, y) in animating_positions:
                    continue  # Will be drawn separately during animation

                # Get the original position of the pixel currently at (x, y)
                orig_y, orig_x = perm[y, x]
                color = video_frame[orig_y, orig_x]

                # Fill the scaled pixel area
                y_start = y * pixel_scale
                y_end = (y + 1) * pixel_scale
                x_start = x * pixel_scale
                x_end = (x + 1) * pixel_scale
                output_frame[y_start:y_end, x_start:x_end] = color

        # Draw grid lines (subtle)
        grid_color = 40  # Dark gray
        for y in range(0, scaled_h, pixel_scale):
            output_frame[y, :] = grid_color
        for x in range(0, scaled_w, pixel_scale):
            output_frame[:, x] = grid_color

        # Process active swaps through phases: highlight -> swapping -> completed
        for swap in active_swaps:
            if swap['phase'] == 'completed':
                continue

            frames_elapsed = out_frame_idx - swap['start_frame']
            x1, y1 = swap['pos1']
            x2, y2 = swap['pos2']

            # Calculate centers for circle drawing
            center1_x = x1 * pixel_scale + pixel_scale // 2
            center1_y = y1 * pixel_scale + pixel_scale // 2
            center2_x = x2 * pixel_scale + pixel_scale // 2
            center2_y = y2 * pixel_scale + pixel_scale // 2

            if swap['phase'] == 'highlight':
                if frames_elapsed >= highlight_frames:
                    # Transition to swapping phase
                    swap['phase'] = 'swapping'
                    swap['swap_start_frame'] = out_frame_idx
                else:
                    # Pulsing effect - intensity varies over highlight duration
                    pulse_progress = frames_elapsed / highlight_frames
                    pulse = 0.5 + 0.5 * np.sin(pulse_progress * np.pi * 3)  # 1.5 pulses

                    for cx, cy in [(center1_x, center1_y), (center2_x, center2_y)]:
                        # Draw red glow around the pixel (larger area)
                        glow_radius = int(pixel_scale * 2 + 4)
                        glow_intensity = int(80 * pulse)
                        for gy in range(max(0, cy - glow_radius), min(scaled_h, cy + glow_radius)):
                            for gx in range(max(0, cx - glow_radius), min(scaled_w, cx + glow_radius)):
                                dist = np.sqrt((gx - cx)**2 + (gy - cy)**2)
                                if dist < glow_radius:
                                    # Red glow that fades with distance
                                    fade = 1 - (dist / glow_radius)
                                    add_red = int(glow_intensity * fade)
                                    output_frame[gy, gx] = np.clip(
                                        output_frame[gy, gx].astype(np.int32) + np.array([add_red, 0, 0]),
                                        0, 255
                                    ).astype(np.uint8)

                        # Draw thick red border rectangle around the pixel
                        border_thickness = max(2, pixel_scale // 3)
                        half = pixel_scale // 2
                        border_color = (255, int(50 * pulse), int(50 * pulse))  # Pulsing red-orange

                        # Top and bottom borders
                        for t in range(border_thickness):
                            y_top = max(0, cy - half - t)
                            y_bot = min(scaled_h - 1, cy + half + t)
                            for bx in range(max(0, cx - half - border_thickness),
                                          min(scaled_w, cx + half + border_thickness + 1)):
                                output_frame[y_top, bx] = border_color
                                output_frame[y_bot, bx] = border_color

                        # Left and right borders
                        for t in range(border_thickness):
                            x_left = max(0, cx - half - t)
                            x_right = min(scaled_w - 1, cx + half + t)
                            for by in range(max(0, cy - half - border_thickness),
                                          min(scaled_h, cy + half + border_thickness + 1)):
                                output_frame[by, x_left] = border_color
                                output_frame[by, x_right] = border_color

                    continue

            if swap['phase'] == 'swapping':
                swap_elapsed = out_frame_idx - swap['swap_start_frame']
                progress = swap_elapsed / swap_duration_frames

                if progress >= 1.0:
                    # Swap complete - update permutation
                    perm[y1, x1], perm[y2, x2] = perm[y2, x2].copy(), perm[y1, x1].copy()
                    swap['phase'] = 'completed'

                    # Draw final positions
                    for (x, y) in [swap['pos1'], swap['pos2']]:
                        orig_y, orig_x = perm[y, x]
                        color = video_frame[orig_y, orig_x]
                        y_start = y * pixel_scale
                        y_end = (y + 1) * pixel_scale
                        x_start = x * pixel_scale
                        x_end = (x + 1) * pixel_scale
                        output_frame[y_start:y_end, x_start:x_end] = color
                    continue

                # Ease the animation
                t = ease_in_out_cubic(progress)

                # Get colors of the two pixels being swapped
                orig_y1, orig_x1 = perm[y1, x1]
                orig_y2, orig_x2 = perm[y2, x2]
                color1 = video_frame[orig_y1, orig_x1]
                color2 = video_frame[orig_y2, orig_x2]

                # Interpolate positions (in scaled coordinates)
                center1 = np.array([center1_x, center1_y], dtype=float)
                center2 = np.array([center2_x, center2_y], dtype=float)

                # Pixel 1 moves toward position 2, pixel 2 moves toward position 1
                current_center1 = center1 + t * (center2 - center1)
                current_center2 = center2 + t * (center1 - center2)

                # Add a slight arc to the motion (lift up in the middle)
                arc_height = pixel_scale * 2 * np.sin(progress * np.pi)
                current_center1[1] -= arc_height
                current_center2[1] -= arc_height

                # Draw the moving pixels (with a highlight/glow effect)
                for center, color in [(current_center1, color1), (current_center2, color2)]:
                    cx, cy = int(center[0]), int(center[1])
                    half = pixel_scale // 2

                    # Draw glow/shadow
                    glow_size = half + 2
                    for gy in range(max(0, cy - glow_size), min(scaled_h, cy + glow_size)):
                        for gx in range(max(0, cx - glow_size), min(scaled_w, cx + glow_size)):
                            # Yellow glow
                            output_frame[gy, gx] = np.clip(
                                output_frame[gy, gx].astype(np.int32) + np.array([30, 30, 0]),
                                0, 255
                            ).astype(np.uint8)

                    # Draw the pixel itself
                    for py in range(max(0, cy - half), min(scaled_h, cy + half)):
                        for px in range(max(0, cx - half), min(scaled_w, cx + half)):
                            output_frame[py, px] = color

        output_frames.append(Image.fromarray(output_frame))

    # Save as GIF
    print(f"Saving to {output_path}...")
    output_frames[0].save(
        output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"Done! Saved {len(output_frames)} frames to {output_path}")


def generate_swap_schedule(num_video_frames, num_swaps, grid_width, grid_height,
                           stagger_frames=20, seed=42, highlight_frames=0, swap_duration_frames=0,
                           sequential=True):
    """
    Generate random swap pairs and when they should occur.

    Args:
        stagger_frames: delay between swap starts (used when sequential=False)
        highlight_frames: frames for highlight phase (used when sequential=True)
        swap_duration_frames: frames for swap animation (used when sequential=True)
        sequential: if True, each swap waits for the previous to complete

    Returns:
        swap_pairs: list of ((x1, y1), (x2, y2))
        swap_schedule: list of frame indices when each swap starts (highlight begins)
    """
    np.random.seed(seed)

    swap_pairs = []
    used_positions = set()

    for _ in range(num_swaps):
        # Pick two random positions that haven't been used
        attempts = 0
        while attempts < 100:
            x1 = np.random.randint(0, grid_width)
            y1 = np.random.randint(0, grid_height)
            x2 = np.random.randint(0, grid_width)
            y2 = np.random.randint(0, grid_height)

            if (x1, y1) != (x2, y2) and (x1, y1) not in used_positions and (x2, y2) not in used_positions:
                swap_pairs.append(((x1, y1), (x2, y2)))
                used_positions.add((x1, y1))
                used_positions.add((x2, y2))
                break
            attempts += 1

    # Schedule swaps
    if sequential and highlight_frames > 0 and swap_duration_frames > 0:
        # Each swap waits for the previous one to complete
        total_swap_time = highlight_frames + swap_duration_frames
        swap_schedule = [i * total_swap_time for i in range(len(swap_pairs))]
    else:
        # Original behavior: stagger by fixed amount (may overlap)
        swap_schedule = [i * stagger_frames for i in range(len(swap_pairs))]

    return swap_pairs, swap_schedule


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create animated pixel shuffle visualization")
    parser.add_argument("-v", "--video", type=str, help="Input video path", default="cab_ride_trimmed.mkv")
    parser.add_argument("-o", "--output", type=str, default="shuffle_animation.gif",
                       help="Output GIF path")
    parser.add_argument("-n", "--num-swaps", type=int, default=10,
                       help="Number of pixel swaps to animate")
    parser.add_argument("-f", "--frames", type=int, default=600,
                       help="Number of video frames to use")
    parser.add_argument("-s", "--scale", type=float, default=0.1,
                       help="Scale factor for input video (to reduce pixel count)")
    parser.add_argument("--pixel-scale", type=int, default=12,
                       help="How much to scale up each pixel in output")
    parser.add_argument("--fps", type=int, default=20,
                       help="Output GIF frame rate")
    parser.add_argument("--swap-duration", type=int, default=48,
                       help="Frames per swap animation")
    parser.add_argument("--highlight-frames", type=int, default=12,
                       help="Frames to show red circles before swap")
    parser.add_argument("--stagger", type=int, default=15,
                       help="Frames between swap starts (only used with --overlap)")
    parser.add_argument("--overlap", action="store_true",
                       help="Allow swaps to overlap (default: sequential, one at a time)")
    parser.add_argument("--seed", type=int, default=420,
                       help="Random seed for swap generation")

    args = parser.parse_args()

    # Find a video if not specified
    video_path = args.video
    if not video_path:
        for f in os.listdir('.'):
            if f.endswith(('.mkv', '.mp4', '.avi')):
                video_path = f
                print(f"Found video: {video_path}")
                break

    if not video_path or not os.path.exists(video_path):
        print("No video found! Please specify with -v")
        return

    # Load video
    print(f"Loading {video_path}...")
    frames, original_fps = load_video_frames(video_path, num_frames=args.frames,
                                              start_frame=30, scale=args.scale)
    print(f"Loaded {len(frames)} frames, shape: {frames.shape}")

    # Generate swaps
    h, w = frames.shape[1:3]
    swap_pairs, swap_schedule = generate_swap_schedule(
        len(frames), args.num_swaps, w, h,
        stagger_frames=args.stagger, seed=args.seed,
        highlight_frames=args.highlight_frames,
        swap_duration_frames=args.swap_duration,
        sequential=not args.overlap
    )

    print(f"Generated {len(swap_pairs)} swaps")

    # Create animation
    create_swap_animation(
        frames, swap_pairs, swap_schedule,
        output_path=args.output,
        pixel_scale=args.pixel_scale,
        swap_duration_frames=args.swap_duration,
        fps=args.fps,
        highlight_frames=args.highlight_frames
    )


if __name__ == "__main__":
    main()
