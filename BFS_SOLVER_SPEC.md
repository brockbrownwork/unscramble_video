# BFS Pixel Unscrambling Solver — Functional Specification

## Overview

A new solver that reconstructs a scrambled video grid by **building the solution from scratch** rather than fixing a scrambled arrangement via swaps. Starting from a single seed pixel, it grows the solved region outward in a breadth-first manner, always placing the best-fitting unplaced pixel into the next frontier position. As it solves it will visualize the current output in a pygame GUI.

**Analogy**: Assembling a jigsaw puzzle — start with one piece, attach best-fitting pieces to exposed edges, expand outward.

## Algorithm

### Phase 0: Precomputation

1. Load video into a `TVWall` instance (scrambled).
2. Compute `all_series = wall.get_all_series()` — shape `(H, W, 3, T)`.
3. Flatten all pixel positions into an **unplaced pool**: every `(x, y)` in the grid.
4. Create an **output grid** of the same dimensions `(W, H)`, initially empty (all positions unassigned).

### Phase 1: Seed Placement

1. Pick a random pixel position from the scrambled grid as the **seed TV**.
2. Place it at the **center** of the output grid: `(W // 2, H // 2)`.
3. Remove the seed from the unplaced pool.
4. Add the seed's empty neighbor positions (up to 4 cardinal or 8 including diagonals) to the **frontier queue**.

### Phase 2: Initial Neighborhood (Permutation Search)

1. Identify the `K` nearest neighbors of the output grid's center from the frontier (initially 4 cardinal neighbors assuming we start with cardinal-only expansion, or 8 if including diagonals — **configurable**).
2. From the unplaced pool, find the top `C` **candidate pixels** with the lowest pairwise dissonance to the seed TV's color series. `C` should be somewhat larger than `K` to give the permutation search room (e.g., `C = K + margin`).
3. **Exhaustive permutation search**: Try all `P(C, K)` permutations of assigning `K` of the `C` candidates to the `K` frontier positions. For each permutation:
   - Compute total neighbor dissonance among the placed group (seed + `K` neighbors, considering adjacency in the output grid).
   - Track the permutation that yields the **minimum total neighbor dissonance**.
4. Commit the best permutation: place those `K` pixels into their assigned output positions.
5. Remove placed pixels from the unplaced pool.
6. Add newly exposed empty neighbors to the frontier queue.

> **Complexity note**: For `K=4` cardinal neighbors and `C=8` candidates, there are `P(8,4) = 1,680` permutations — very tractable. For `K=8` (all neighbors) and `C=12`, there are `P(12,8) = 19,958,400` — may need pruning or a greedy fallback. The spec should support a configurable limit, falling back to greedy assignment if permutation count exceeds a threshold.

### Phase 3: BFS Expansion

Repeat until the frontier is empty or all pixels are placed:

1. **Select next frontier position**: Pop a position from the frontier queue. Prioritize positions with the **most already-placed neighbors** (more context = better matching). Break ties by insertion order (standard BFS).

2. **Gather context**: Identify the already-placed neighbors of this frontier position in the output grid. Collect their color series.

3. **Find best pixel**: From the unplaced pool, find the pixel whose color series minimizes the mean dissonance to the placed neighbors. Specifically:
   ```
   score(candidate) = mean( distance(candidate_series, neighbor_series)
                             for each placed neighbor )
   ```
   - For efficiency, precompute distances from a shortlist of candidates rather than scanning the entire pool.
   - **Shortlisting strategy**: Use a fast approximate metric (e.g., euclidean distance to the mean neighbor series) to select the top `S` candidates, then evaluate full dissonance for those `S` only.

4. **Place pixel**: Assign the best candidate to the frontier position in the output grid. Remove it from the unplaced pool.

5. **Expand frontier**: Add any empty neighbors of the newly placed position to the frontier queue (if not already in the frontier).

6. **Progress tracking**: Log/emit placement count, current total dissonance of placed region, and percentage complete.

### Phase 4: Finalization

1. After BFS completes, the output grid defines a mapping from output position → original scrambled position.
2. Apply this mapping as the new permutation on the `TVWall` instance.
3. Compute final dissonance map and correctness metrics.

## Output Grid vs. Input Grid

The algorithm builds an **output grid** that is conceptually separate from the scrambled input:

- **Input grid**: The scrambled `TVWall`. Each position `(x, y)` holds some TV with a known color series (accessible via `all_series[y, x]`).
- **Output grid**: A new grid of the same dimensions. Each position maps to a specific input-grid position (i.e., "which scrambled-grid TV goes here?").
- The output grid is essentially a new permutation that we're constructing from scratch.

When we say "place pixel P at output position Q", we mean: "the TV currently at scrambled position P should be displayed at output position Q."

## Distance Metrics

Support the same metrics as the existing solver:

| Metric | Speed | Quality | GPU Support |
|--------|-------|---------|-------------|
| `euclidean` | Fast | Good | Yes |
| `squared` | Fast | Good | Yes |
| `manhattan` | Fast | Good | Yes |
| `cosine` | Fast | Moderate | Yes |
| `dtw` | Slow | Best | No |

Default: `euclidean` (best speed/quality tradeoff for BFS where we evaluate many candidates per step).

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video_path` | (required) | Path to input video file |
| `num_frames` | 100 | Number of frames to load |
| `stride` | 1 | Frame skip interval |
| `min_entropy` | 0.0 | Minimum frame entropy filter |
| `seed_position` | random | Starting pixel in scrambled grid, or `"random"` |
| `neighbor_mode` | `"cardinal"` | `"cardinal"` (4-connected) or `"all"` (8-connected) for BFS expansion |
| `distance_metric` | `"euclidean"` | Distance metric for series comparison |
| `dtw_window` | 0.1 | Sakoe-Chiba band width (only for DTW) |
| `initial_candidates` | 8 | Number of candidate pixels for initial permutation search (`C`) |
| `permutation_limit` | 100_000 | Max permutations to try before falling back to greedy |
| `shortlist_size` | 50 | Number of candidates to evaluate per frontier position (`S`) |
| `scramble_seed` | 42 | RNG seed for scrambling |
| `output_video_path` | None | Optional path to save result video |
| `use_gpu` | True | Use GPU acceleration if available |

## Shortlisting Strategy (Performance)

Evaluating every unplaced pixel for every frontier position is `O(N)` per placement and `O(N^2)` total — too slow for large grids. The shortlisting approach:

1. **Precompute a compact representation** for each TV's color series (e.g., mean color across frames, or a low-dimensional embedding).
2. **For each frontier position**, compute the mean compact representation of its placed neighbors.
3. **Fast approximate search**: Find the top `S` unplaced pixels closest to this mean in the compact space (using a KD-tree, ball tree, or brute-force on the compact vectors).
4. **Precise evaluation**: Compute full dissonance (using the chosen metric) for only these `S` candidates.
5. **Place the best** among the `S`.

**Compact representation options** (in order of preference):
- **Mean RGB over time**: shape `(3,)` per pixel — extremely fast, coarse.
- **Downsampled series**: subsample the time series to `k` frames, flatten to `(3k,)` — fast, preserves temporal structure.
- **UMAP embedding**: precompute 2D/3D embedding of all series — good spatial locality, but upfront cost.

The spec leaves the exact shortlisting implementation flexible. Start with mean-RGB + brute-force for simplicity; optimize later if needed.

## Edge Handling

- The output grid has the same dimensions as the input. Pixels placed near edges will have fewer neighbors (boundary condition).
- Frontier positions at grid edges are valid — they just have fewer placed neighbors to match against.
- The BFS will naturally fill corners and edges last (fewer neighbors → lower priority).

## Correctness Tracking

Since we know the ground truth permutation (identity = correct), we can track:

- **Pixels correctly placed**: count of output positions where the placed TV matches the original.
- **Neighbor correctness**: fraction of placed pixel-pairs that are true neighbors in the original.
- **Dissonance curve**: total dissonance of the placed region over time.

## File Structure

```
bfs_solver.py          # Core BFS solver algorithm (no GUI)
```

### `bfs_solver.py` — Public API

```python
class BFSSolver:
    def __init__(self, wall: TVWall, **kwargs):
        """Initialize solver with a scrambled TVWall and config parameters."""

    def solve(self, callback=None) -> np.ndarray:
        """
        Run BFS placement algorithm.

        Args:
            callback: Optional callable(step_info_dict) invoked after each placement
                      for progress tracking / visualization.

        Returns:
            output_grid: np.ndarray of shape (H, W, 2) where output_grid[y, x] = (orig_x, orig_y)
                         giving the scrambled-grid position of the TV placed at output (x, y).
        """

    def apply_solution(self):
        """Apply the computed output_grid as the TVWall's permutation."""

    def get_stats(self) -> dict:
        """Return solve statistics: total dissonance, correct count, iterations, timing."""
```

### CLI Interface

```bash
python bfs_solver.py -v video.mkv -n 100 -s 1 --metric euclidean \
    --scramble-seed 42 --shortlist 50 --neighbor-mode cardinal
```

Output:
- Print progress (percentage placed, running dissonance).
- Save before/after frame images.
- Print final stats (correct positions, total dissonance, solve time).

## Open Questions

1. **Growth direction bias**: Should the BFS expand uniformly in all directions, or should we bias toward directions with more information (e.g., expand toward where unplaced pixels have stronger signal)?

2. **Error propagation**: Early mistakes compound — a misplaced pixel biases all subsequent placements near it. Should we include a local refinement pass after BFS completes (e.g., run the existing greedy swap solver on the BFS output)?

3. **Seed selection**: Random seed risks starting with a low-information pixel. Should we select the seed as the pixel with highest color variance (most distinctive time series)?

4. **Multi-seed**: Start multiple BFS regions from different seeds, grow them concurrently, and merge when they meet? This could reduce error propagation.

5. **Backtracking**: If a placement looks bad in hindsight (high dissonance after more neighbors are placed), should we allow undoing and re-placing?
