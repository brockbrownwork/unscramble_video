# CLAUDE.md

## Project Overview

**unscramble_video** is an experimental tool that reconstructs scrambled video frames by analyzing pixel color sequences over time.

### The Thought Experiment

Imagine a giant wall of CRT TVs arranged in a 2D grid, where each TV displays only one color at a time - essentially acting as a single pixel. When you play a video, each TV shows the color sequence for its corresponding pixel position.

Now knock down the wall and scatter the TVs. Can you figure out where each TV belongs?

The key insight: **neighboring pixels in a video tend to have similar color sequences over time.** If the video has rich color variety (not just black/white blinking), TVs that belong next to each other will have correlated color histories. This correlation is what we exploit to solve the puzzle.

### Neuroscience Inspiration

This project is inspired by neural plasticity research:

- **Ferret rewiring experiments**: When scientists rewired ferrets' optic nerves to the auditory cortex (and vice versa), the animals still learned to see. This shows that "pixel positions" in the visual field are *learned*, not hardcoded.

- **Topological organization**: In the brain, neurons that respond to adjacent body parts are physically adjacent (somatotopy). If you tap your index finger, then your ring finger, the neural region responding to your middle finger will be *between* them. This topological tendency is preserved even after nerve rewiring.

The unscramble problem is analogous: we're trying to recover spatial topology from temporal correlation alone.

## Tech Stack

- **Language:** Python
- **Core Libraries:** opencv-python, numpy, umap-learn, matplotlib, Pillow, tqdm, scipy
- **Distance Metrics:** aeon (DTW pairwise distances)
- **ML/Evaluation:** scikit-learn (precision-recall, ROC curves)
- **GUI:** tkinter (interactive experiments)
- **External Tools:** ffmpeg (via subprocess)
- **Development:** Jupyter Notebooks for experimentation

## Project Structure

```
unscramble_video/
├── tv_wall.py                         # TVWall class - core abstraction
├── neighbor_dissonance_gui.py         # Interactive dissonance visualization
├── greedy_solver_gui.py               # Interactive solver with multiple strategies
├── experiment_neighbor_dissonance.py  # CLI experiment with ROC/PR curves
├── test_tv_wall.py                    # Unit tests for TVWall
├── unscramble.py                      # Original UMAP approach (Euclidean)
├── unscramble_dtw.py                  # DTW-based UMAP approach
├── videos/
│   └── stitch.py                      # Video concatenation utility
├── *.ipynb                            # Experimental notebooks
├── *.mkv                              # Input video files
├── *.gif                              # Output animations
└── *.npy                              # Cached numpy arrays
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install aeon scikit-learn  # Additional deps for experiments

# Run interactive GUIs
python neighbor_dissonance_gui.py  # Visualize dissonance heatmaps
python greedy_solver_gui.py        # Run solver with animation

# Run CLI experiment
python experiment_neighbor_dissonance.py -v video.mkv -n 20 -f 100

# Run original UMAP scripts
python unscramble.py
python unscramble_dtw.py

# Stitch videos
cd videos && python stitch.py
```

## Key Concepts

- **TVs**: Pixel color time-series (each pixel position across all frames)
- **TVWall**: Class representing the wall of TVs; handles video loading, position swapping, dissonance computation, and frame/video export
- **Neighbor Dissonance**: For each position, the average distance to its 8 neighbors. High dissonance = likely misplaced
- **Pipeline**: Video → frames → TVs → UMAP embedding → RGB visualization → animation
- **DTW (Dynamic Time Warping)**: Distance metric that accounts for time-shifts between sequences
- **Stride**: Frame skip interval for capturing longer-term temporal patterns

## DTW Pairwise Distance

Use `aeon.distances.dtw_pairwise_distance` to compute DTW distances between TV color series:

```python
from aeon.distances import dtw_pairwise_distance
import numpy as np

# Get color series for multiple TVs - shape: (n_tvs, n_channels, n_frames)
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
```

**Parameters:**
- `X`: Array of shape `(n_cases, n_channels, n_timepoints)` for multivariate series
- `y`: Optional second collection to compare against
- `window`: Sakoe-Chiba band as fraction of series length (0.0-1.0)
- `n_jobs`: Parallel jobs (-1 for all cores)

## TVWall Class

The `TVWall` class (`tv_wall.py`) is the core abstraction for simulating the TV wall. It loads a video and tracks which TV is at which position via a permutation mapping.

### Basic Usage

```python
from tv_wall import TVWall

# Load video (optional: num_frames, start_frame, stride)
wall = TVWall("video.mkv", num_frames=100, stride=2)

# Scramble all TVs randomly
wall.scramble(seed=42)

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

### Swap Methods

```python
# Randomly shuffle all positions
wall.scramble(seed=42)

# Shuffle a subset of positions (random global swaps)
wall.random_swaps(num_positions=50, seed=42)

# Swap pairs with unlimited distance (returns list of swap pairs)
pairs = wall.pair_swaps(num_swaps=10, seed=42)

# Swap pairs within a max distance (for testing local algorithms)
pairs = wall.short_swaps(num_swaps=10, max_dist=5, seed=42)

# Swap two specific positions
wall.swap_positions((x1, y1), (x2, y2))

# Place TV from orig_pos into new_pos (overwrites)
wall.swap(orig_pos=(0, 0), new_pos=(5, 5))
```

### Dissonance Methods

```python
# Get all color series with current permutation
all_series = wall.get_all_series()  # shape: (height, width, 3, num_frames)

# Compute dissonance for a single position
d = wall.compute_position_dissonance(x, y, all_series, kernel_size=3,
                                      distance_metric='dtw', window=0.1)

# Compute dissonance map for all positions
dmap = wall.compute_dissonance_map(all_series, kernel_size=3,
                                    distance_metric='dtw', window=0.1)

# Compute total dissonance (sum over positions)
total = wall.compute_total_dissonance(all_series, positions=[(x1,y1), (x2,y2)])

# Get neighbor positions for a coordinate
neighbors = wall.get_neighbors(x, y, kernel_size=3)  # 8 neighbors for 3x3
```

### Position Tracking

```python
# Get original position of TV currently at (x, y)
orig_x, orig_y = wall.get_original_position(x, y)

# Get current position of TV originally at (orig_x, orig_y)
cur_x, cur_y = wall.get_current_position(orig_x, orig_y)

# Count of misplaced TVs
print(wall.num_swapped)
```

## Neighbor Dissonance

The core metric for detecting misplaced TVs. For each TV position, we measure how similar its color time-series is to its neighbors. The `kernel_size` parameter controls how many neighbors to consider:

```
kernel_size=3 (8 neighbors)       kernel_size=5 (24 neighbors)

    +---+---+---+                 +---+---+---+---+---+
    | N | N | N |                 | N | N | N | N | N |
    +---+---+---+                 +---+---+---+---+---+
    | N | X | N |                 | N | N | N | N | N |
    +---+---+---+                 +---+---+---+---+---+
    | N | N | N |                 | N | N | X | N | N |
    +---+---+---+                 +---+---+---+---+---+
                                  | N | N | N | N | N |
X = center position               +---+---+---+---+---+
N = neighbor (compared to X)      | N | N | N | N | N |
                                  +---+---+---+---+---+
```

**Dissonance formula:**
```
dissonance(X) = mean( distance(X, neighbor) for each neighbor N )
```

- **Low dissonance**: TV fits well with neighbors (likely in correct position)
- **High dissonance**: TV is dissimilar to neighbors (likely misplaced, candidate for swapping)

## Solving Strategies

The `greedy_solver_gui.py` implements three strategies:

1. **Highest Dissonance**: Find the highest dissonance position, try swapping with each neighbor, keep the best improvement

2. **Best of Top-K**: Consider top-K highest dissonance positions, try all pairwise swaps among them, keep the best improvement

3. **Simulated Annealing**: Randomly swap among top-K candidates, accept worse moves probabilistically based on temperature (decays over iterations)

## GUI Tools

### Neighbor Dissonance GUI (`neighbor_dissonance_gui.py`)

Interactive tool for visualizing dissonance:
- Load video and set parameters (frames, stride, kernel size)
- Perform random swaps or short-distance swaps
- Compute DTW and Euclidean dissonance side-by-side
- View heatmaps with swap markers
- Compare dissonance distributions between swapped and non-swapped positions

### Greedy Solver GUI (`greedy_solver_gui.py`)

Interactive solver with real-time animation:
- Load video, scramble, then watch the solver unscramble
- Compare strategies: greedy, best-of-top-K, simulated annealing
- Tune parameters: top-K, max iterations, temperature, cooling rate
- Run batch experiments across multiple scramble levels
- View progress charts and correctness maps

## CLI Experiment (`experiment_neighbor_dissonance.py`)

Command-line tool for systematic evaluation:

```bash
python experiment_neighbor_dissonance.py \
    -v cab_ride_trimmed.mkv \
    -f 100 -s 30 \
    -n 20 \
    -m dtw \
    -w 0.1

# Outputs:
# - Precision-Recall curve
# - ROC curve with AUC
# - Dissonance heatmap overlay
# - Statistics (separation z-score, top-K recall)
```

## Coding Conventions

- Lowercase with underscores for functions/variables
- Descriptive function names with docstrings
- Use `tqdm` for progress bars on long operations
- Hardcoded parameters at script start (e.g., `number_of_frames`, `n_neighbors`)
- Scripts may contain `# In[X]:` cell markers (converted from Jupyter)

## Git Conventions

- Commit prefixes: `feat:`, `chore:`, `fix:`, `docs:`
- Keep notebooks and generated media out of git (see .gitignore)

## Ideas

- **Scramble threshold experiment**: Start with correct video, randomize N positions, solve, then increase N until the solver breaks down. Find the critical threshold.

- **Distance metric comparison**: Compare Euclidean vs DTW correlation with actual TV distance to validate DTW as the better metric.

- **A* search**: Use neighbor dissonance as a heuristic cost function to prune the combinatorial search space of possible swaps.

- **Hierarchical solving**: Solve at low resolution first (pooled pixels), then refine at higher resolution.

- **Subsampled reconstruction**: When sampling a fraction of pixel positions rather than the full grid, the target arrangement should preserve the original aspect ratio. For a video with W×H dimensions and sampling ratio R, arrange TVs in a grid of (W√R) × (H√R). Example: sampling 25% of a 100×200 video → arrange as 50×100.
