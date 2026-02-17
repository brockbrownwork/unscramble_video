"""
Extract cropped frames from a video.

Takes the center 50% of both x and y dimensions from the first N frames.
"""

import os
import cv2
from tqdm import tqdm

VIDEO_PATH = r"C:\Users\Brock\Documents\code\unscramble_video\cab_ride_trimmed.mkv"
OUTPUT_DIR = r"C:\Users\Brock\Documents\code\unscramble_video\bfs_output\original frames"
NUM_FRAMES = 10_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Center 50% crop bounds
x0 = w // 4
x1 = x0 + w // 2
y0 = h // 4
y1 = y0 + h // 2

print(f"Video: {w}x{h}, {total} total frames")
print(f"Crop: x=[{x0}:{x1}], y=[{y0}:{y1}] -> {x1-x0}x{y1-y0}")
print(f"Extracting {min(NUM_FRAMES, total)} frames to {OUTPUT_DIR}")

for i in tqdm(range(min(NUM_FRAMES, total)), desc="Extracting"):
    ret, frame = cap.read()
    if not ret:
        print(f"Stopped at frame {i} (read failed)")
        break
    cropped = frame[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{i:06d}.png"), cropped)

cap.release()
print("Done!")
