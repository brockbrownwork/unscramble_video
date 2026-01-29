# CLAUDE.md

## Project Overview

**unscramble_video** is an experimental video analysis and visualization tool that reconstructs scrambled video frames by analyzing pixel color sequences over time using UMAP dimensionality reduction.

This project is based on a thought problem. Consider that you have a gigantic wall of CRT TVs stacked in a 2d array. Each TV is only capable of displaying one color on its screen at any given moment. You can play a video back on the TV wall where each TV is a pixel position of the video. If you knock down the wall of TVs, can you figure out where the TVs go again? And what kinds of videos can you use that will work for putting the TVs back in their proper place? For example, if the video is just 30 seconds of white and black blinking it would be impossible to figure out where they go. The idea is that TVs with similar color series will have strong tendencies to belong next to each other as long as the video is rich in color variety.

The thought was based on neuroscience originally. There was an experiment where they rewired ferrets' ocular nerve to the auditory cortex and vise-versa and they were still able to learn how to see. Therefore, the positions of "pixels" in animal visual fields is learned and not hard coded. A similar experiment was done with deafferentation in macaques by Edward Taub. It's based on the notion of topological tendencies in the brain; if you tap the index finger, and tap the ring finger, the region that lights up when the middle finger is tapped will live between the index and ring finger regions that light up. If you rewire the nerve from one finger to another, this topological tendency will be relearned.

## Tech Stack

- **Language:** Python
- **Core Libraries:** opencv-python, numpy, umap-learn, matplotlib, Pillow, tqdm, fastdtw, scipy
- **External Tools:** ffmpeg (via subprocess)
- **Development:** Jupyter Notebooks for experimentation

## Project Structure

```
unscramble_video/
├── tv_wall.py             # TVWall class - core abstraction for TV wall simulation
├── unscramble.py          # Main script (Euclidean distance)
├── unscramble_dtw.py      # DTW experiment (Dynamic Time Warping distance)
├── videos/
│   └── stitch.py          # Video concatenation utility
├── *.ipynb                # Experimental notebooks
├── *.mkv                  # Input video files
├── *.gif                  # Output animations
└── *.npy                  # Cached numpy arrays
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run main script (Euclidean distance)
python unscramble.py

# Run DTW experiment (slower but captures temporal shifts)
python unscramble_dtw.py

# Stitch videos
cd videos && python stitch.py
```

## Key Concepts

- **TVs**: Pixel color time-series (each pixel position across all frames)
- **TVWall**: Class representing the wall of TVs; handles video loading, position swapping, and frame/video export
- **Pipeline**: Video → frames → TVs → UMAP embedding → RGB visualization → animation
- **DTW (Dynamic Time Warping)**: Distance metric that accounts for time-shifts between sequences, useful when edges/objects move across adjacent pixels
- **Stride**: Frame skip interval for capturing longer-term temporal patterns without increasing computation

## DTW Pairwise Distance

Use `aeon.distances.dtw_pairwise_distance` to compute DTW distances between TV color series:

```python
from aeon.distances import dtw_pairwise_distance
import numpy as np

# Get color series for multiple TVs - shape: (n_tvs, n_channels, n_frames)
# Each TV has 3 channels (RGB) and n_frames timepoints
tv_series = np.array([
    wall.get_tv_color_series(x1, y1).T,  # shape: (3, n_frames)
    wall.get_tv_color_series(x2, y2).T,
    wall.get_tv_color_series(x3, y3).T,
])

# Compute pairwise DTW distance matrix - shape: (n_tvs, n_tvs)
distances = dtw_pairwise_distance(tv_series)

# With Sakoe-Chiba band constraint (faster, limits warping)
distances = dtw_pairwise_distance(tv_series, window=0.1)

# Parallel computation
distances = dtw_pairwise_distance(tv_series, n_jobs=-1)

# Distance between one TV and all others
single_tv = wall.get_tv_color_series(x, y).T  # shape: (3, n_frames)
distances_to_one = dtw_pairwise_distance(tv_series, single_tv)  # shape: (n_tvs, 1)
```

**Parameters:**
- `X`: Array of shape `(n_cases, n_channels, n_timepoints)` for multivariate series
- `y`: Optional second collection to compare against
- `window`: Sakoe-Chiba band as fraction of series length (0.0-1.0), limits warping
- `n_jobs`: Parallel jobs (-1 for all cores)

## TVWall Class

The `TVWall` class (`tv_wall.py`) is the core abstraction for simulating the TV wall. It loads a video and tracks which TV is at which position via a swap mapping.

**Usage:**
```python
from tv_wall import TVWall

# Load video (optional: num_frames, start_frame, stride)
wall = TVWall("video.mkv", num_frames=100, stride=2)

# Scramble all TVs randomly
wall.scramble(seed=42)

# Or scramble a subset of positions
wall.random_swaps(num_positions=50, seed=42)

# Get color time-series for a TV at original position
colors = wall.get_tv_color_series(x, y)  # shape: (num_frames, 3)

# Get/save frames with current swap configuration
img = wall.get_frame_image(timestep=0)
wall.save_frame(timestep=0, output_path="frame.png")

# Export video with swapped positions
wall.save_video("scrambled.mp4", fps=30)

# Reset to original positions
wall.reset_swaps()
```

**Key Methods:**
- `scramble(seed)` - Randomly shuffle all TV positions
- `random_swaps(num_positions, seed)` - Shuffle a subset of positions
- `swap(orig_pos, new_pos)` - Move a single TV to a new position
- `swap_positions(pos1, pos2)` - Swap two TVs with each other
- `get_tv_color_series(x, y)` - Get RGB time-series for a TV
- `get_frame_image(timestep)` - Get PIL Image at a timestep
- `save_video(path, fps)` - Export video via ffmpeg

## Coding Conventions

- Lowercase with underscores for functions/variables
- Descriptive function names with docstrings
- Use `tqdm` for progress bars on long operations
- Hardcoded parameters at script start (e.g., `number_of_frames`, `n_neighbors`)
- Scripts may contain `# In[X]:` cell markers (converted from Jupyter)

## Git Conventions

- Commit prefixes: `feat:`, `chore:`, `fix:`, `docs:`
- Keep notebooks and generated media out of git (see .gitignore)

## Unscrambling Approach

### Neighbor Dissonance

The core metric for detecting misplaced TVs. For each position, compute the average DTW distance to its 8 neighbors using a 3x3 kernel:

```
+---+---+---+
| ↖ | ↑ | ↗ |
+---+---+---+
| ← | X | → |   neighbor_dissonance(X) = sum(DTW(X, neighbor)) / num_neighbors
+---+---+---+
| ↙ | ↓ | ↘ |
+---+---+---+
```

High dissonance indicates a TV that doesn't belong with its neighbors - a candidate for being swapped.

### Incremental Solving Strategy

1. Start with the video in correct configuration
2. Swap 2 TVs, then solve
3. Double the number of swapped TVs each iteration until hitting a roadblock
4. Use neighbor dissonance to identify swap candidates

The main challenge is the combinatorial explosion of permutations among swap candidates. A* search may help, using neighbor dissonance as the cost heuristic to prune the search tree.

## Ideas

- **Hierarchical chunking**: Find nearest neighbors in 9x9 blocks, then merge blocks into 18x18, and so on. Refinement stage: randomly remove pixels and fill back in with best DTW fit.

- **Foveal packing**: Use hexagonal packing that's dense at center and sparse at edges (like the retina). Escher-style infinite density gradients could help organize ambiguous regions.

- **Scramble threshold experiment**: Start with correct video, randomize N positions, solve, then increase N until the solver breaks down. Find the critical threshold.

- **Distance metric comparison**: Compare Euclidean vs DTW correlation with actual TV distance to validate DTW as the better metric.
