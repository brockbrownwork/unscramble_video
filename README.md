# unscramble_video

An experimental tool that reconstructs scrambled video frames by analyzing pixel color sequences over time.

## The Thought Experiment

Imagine a giant wall of CRT TVs arranged in a 2D grid, where each TV displays only one color at a time - essentially acting as a single pixel. When you play a video, each TV shows the color sequence for its corresponding pixel position.

Now knock down the wall and scatter the TVs. Can you figure out where each TV belongs?

The key insight: **neighboring pixels in a video tend to have similar color sequences over time.** If the video has rich color variety (not just black/white blinking), TVs that belong next to each other will have correlated color histories. This correlation is what we exploit to solve the puzzle.

## Neuroscience Inspiration

This project is inspired by neural plasticity research:

- **Neural rewiring experiments**: When scientists rewired ferrets' optic nerves to the auditory cortex (and vice versa), the animals still learned to see. This shows that "pixel positions" in the visual field are *learned*, not hardcoded.

- **Topological organization**: In the brain, neurons that respond to adjacent body parts are physically adjacent (somatotopy). If you tap your index finger, then your ring finger, the neural region responding to your middle finger will be *between* them. This topological tendency is preserved even after nerve rewiring.

The unscramble problem is analogous: we're trying to recover spatial topology from temporal correlation alone.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive GUIs

```bash
# Visualize dissonance heatmaps
python neighbor_dissonance_gui.py

# Run solver with animation
python greedy_solver_gui.py
```

### CLI Experiment

```bash
python experiment_neighbor_dissonance.py -v video.mkv -n 20 -f 100
```

### UMAP Scripts

```bash
python unscramble.py      # Euclidean approach
python unscramble_dtw.py  # DTW-based approach
```

## Key Concepts

- **TVs**: Pixel color time-series (each pixel position across all frames)
- **TVWall**: Class representing the wall of TVs; handles video loading, position swapping, dissonance computation, and frame/video export
- **Neighbor Dissonance**: For each position, the average distance to its 8 neighbors. High dissonance = likely misplaced
- **DTW (Dynamic Time Warping)**: Distance metric that accounts for time-shifts between sequences

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
└── videos/
    └── stitch.py                      # Video concatenation utility
```

## Tech Stack

- **Language:** Python
- **Core Libraries:** opencv-python, numpy, umap-learn, matplotlib, Pillow, tqdm, scipy
- **Distance Metrics:** aeon (DTW pairwise distances)
- **ML/Evaluation:** scikit-learn (precision-recall, ROC curves)
- **GUI:** tkinter

## Status

Still in development.
