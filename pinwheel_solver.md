# Pinwheel Solver

A reconstruction algorithm that builds the unscrambled video by constructing **pinwheels** — circular, layered pixel arrangements — and tiling them onto a **pinboard** to fill the output grid.

## Core Concepts

### Pinwheel

A pinwheel is a circular arrangement of pixels organized in concentric rings around a **center pixel**. Each ring is populated by the next-most-similar unplaced pixels (by distance to the center's color time-series), and the ordering within each ring is optimized via the **Travelling Salesman Problem** using neighbor dissonance as the edge cost. Finally, each ring is **rotated** into the orientation that minimizes neighbor dissonance with its inner rings and any already-placed neighbors on the pinboard.

#### Ring Structure

Rings grow outward from the center. The number of pixels per ring follows the discrete circle pixel counts (the number of integer-coordinate points at each radial distance):

| Ring | Radius | Pixels in Ring | Cumulative Rank Range |
|------|--------|----------------|-----------------------|
| 0    | 0      | 1 (center)     | Center                |
| 1    | 1      | 8              | 1st - 8th             |
| 2    | 2      | 12             | 9th - 20th            |
| 3    | 3      | 16             | 21st - 36th           |
| ...  | r      | ~4r (approx)   | ...                   |

The exact pixel count per ring is determined by the number of integer-lattice points at each rounded Euclidean distance from the origin — the same counting used in the HTML visualization.

#### Construction Procedure

Given a center pixel `C`:

1. **Rank all unplaced pixels** by their distance (e.g. Euclidean, Mahalanobis, or Summed Color Distance) to `C`'s color time-series. The `k`-th closest unplaced pixel gets rank `k`.

2. **Assign pixels to rings** based on their rank. Ring 1 gets the top 8, ring 2 gets the next 12, ring 3 gets the next 16, etc. The exact counts come from the discrete-circle pixel counts at each integer radius.

3. **Solve TSP within each ring.** For each ring, solve the Travelling Salesman Problem over its member pixels, using pairwise neighbor dissonance as the edge cost. This determines the **ordering** of pixels along the ring (which pixel is next to which), but not yet their absolute positions on the lattice.

4. **Rotate each ring into optimal position.** Each ring has `N` pixels occupying `N` lattice positions at that radius. There are exactly `N` discrete rotations (cyclic shifts) of the TSP ordering onto the lattice positions. For each rotation:
   - Compute the total neighbor dissonance between this ring's pixels and all already-placed inner-ring pixels (and any pinboard neighbors if applicable).
   - Select the rotation with the lowest total dissonance.

5. **Place from inside out.** The center is placed first, then ring 1 is ordered and rotated, then ring 2, and so on. Each ring "locks in" before the next ring is constructed.

#### Rotation Math

A ring at radius `r` has `N(r)` pixel positions arranged in angular order around the center. Applying a rotation of `k` steps means shifting the TSP-ordered sequence by `k` positions along the lattice:

```
position[i] = lattice_points[r][(i + k) % N(r)]
```

where `lattice_points[r]` is the sorted list of integer-coordinate points at distance `r` from the origin (sorted by angle from north, clockwise). The optimal rotation is:

```
k* = argmin_k  sum_{i} dissonance(pixel[i], neighbors_of(position[(i+k) % N]))
```

where neighbors include both inner-ring pixels already placed and any pinboard neighbors from adjacent pinwheels.

### Pinboard

The pinboard is the output grid where pinwheels are placed. It tracks which grid positions have been filled and which pixel (by original identity) occupies each position. When the pinboard is interpreted as a TV wall, each grid position `(x, y)` is assigned the pixel belonging to the **nearest pinwheel center** — that is, the pixel placed by whichever pinwheel's center is geometrically closest to `(x, y)`.

#### Voronoi Assignment

Since pinwheels overlap (each pinwheel claims a circular region of the grid), a grid position may fall within the radius of multiple pinwheels. The assignment rule is:

- For each grid position `(x, y)`, find the pinwheel whose **center** is closest (Euclidean distance on the grid).
- The pixel at `(x, y)` is the one placed by that pinwheel at the corresponding ring and angular position.

This produces a Voronoi-like tiling of the pinboard, where each pinwheel "owns" the region of the grid closest to its center.

## Pinboard Construction Algorithm

### Phase 1: Seed Pinwheel

1. Select a seed pixel (random or user-specified).
2. Construct a pinwheel around it (full ring construction as described above).
3. Place it at the center of the pinboard.

### Phase 2: Iterative Expansion

After the seed pinwheel is placed, new pinwheels are spawned from the boundary of existing pinwheels. Two placement modes control where the next pinwheel center goes:

#### Placement Modes

**Edge Mode:** The new pinwheel center is placed at distance `2R` from the parent pinwheel center (where `R` is the pinwheel radius), along one of the cardinal/diagonal directions. This means adjacent pinwheels are tangent — their outermost rings just touch.

```
Parent center ---- 2R ----> New center

  [  Parent  ] [  New  ]
  Ring boundaries touch but don't overlap.
```

**Midpoint Mode:** The new pinwheel center is placed at distance `R` from the parent center. This means the new pinwheel's center sits on the parent's outermost ring, and the two pinwheels overlap significantly. The overlapping region provides extra constraints for the rotation optimization.

```
Parent center -- R --> New center

  [  Parent  ]
       [ New ]
  Significant overlap region.
```

#### Expansion Order

New pinwheel centers are spawned in **BFS order** from the seed:

1. After placing the seed pinwheel, add its 4 cardinal neighbors (at distance `R` or `2R` depending on mode) to a frontier queue.
2. Pop the next position from the frontier.
3. Construct a pinwheel at that position:
   - The center pixel is selected as the closest unplaced pixel to the expected location (based on the parent pinwheel's edge pixels and their distances).
   - Rings are populated from the remaining unplaced pixel pool, ranked by distance to this new center.
   - TSP ordering and rotation optimization proceed as normal, but now the dissonance calculation also includes already-placed neighbors from adjacent pinwheels on the pinboard.
4. **Rotate the entire pinwheel** into the orientation that minimizes total neighbor dissonance with all already-pinned neighbors on the pinboard.
5. Pin the pinwheel to the board.
6. Add unexplored cardinal neighbors of this new pinwheel to the frontier.
7. Repeat until all pixels are placed or the frontier is exhausted.

### Phase 3: Voronoi Resolution

Once all pinwheels are pinned, resolve any multiply-claimed grid positions using the nearest-center rule. Each pixel on the final output grid is determined by the pinwheel whose center is closest.

## Diagram: Expansion Steps

```
Step 1: Seed pinwheel at origin
         ___
        /   \
       | (0,0)|
        \___/

Step 2: Expand East (edge mode, distance 2R)
         ___     ___
        /   \   /   \
       | (0,0)|| (2R,0)|
        \___/   \___/

Step 3: Expand North from seed
         ___
        /   \
       |(0,-2R)|
        \___/
         ___     ___
        /   \   /   \
       | (0,0)|| (2R,0)|
        \___/   \___/

Step 4: Continue BFS...
```

## Properties

### Why Pinwheels Help

- **Strong local optimization.** Each pinwheel enforces that the pixels closest to a center (in color-time-series space) are placed physically closest to it. The TSP ordering within rings ensures smooth angular transitions. The rotation step ensures radial coherence.

- **Overlap constraints (midpoint mode).** When pinwheels overlap, the shared region must satisfy two pinwheels simultaneously. This over-constrains the placement and reduces ambiguity, similar to how overlapping jigsaw constraints improve global correctness.

- **Graceful degradation.** Even if distant pinwheels accumulate error, each local neighborhood is well-optimized. The Voronoi assignment means errors don't propagate — each region is governed by its nearest center.

### Comparison to BFS Solver

| Property | BFS Solver | Pinwheel Solver |
|----------|-----------|-----------------|
| Placement unit | Single pixel | Circular cluster |
| Local optimization | 1 pixel vs. neighbors | Full ring TSP + rotation |
| Error propagation | Greedy, can cascade | Contained within pinwheel |
| Overlap constraints | None | Yes (midpoint mode) |
| Computational cost | O(N * S) per pixel | O(N * S) per pinwheel + TSP per ring |
| Parallelism | Sequential BFS | Pinwheels on disjoint regions can be parallel |

### Computational Considerations

- **TSP per ring:** Each ring has ~4r pixels. For small rings (r < 10), exact TSP or nearest-neighbor heuristic is fast. For larger rings, use 2-opt or other TSP heuristics.
- **Rotation search:** Each ring has `N(r)` discrete rotations to evaluate. This is linear in ring size.
- **Candidate ranking:** Same as BFS — use mean-RGB coarse filter then full time-series reranking.
- **GPU acceleration:** The dissonance computation during rotation search can be batched and run on GPU, similar to the existing `evaluate_swap_batch()`.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pinwheel_radius` | Number of rings per pinwheel | 5 |
| `placement_mode` | `"edge"` or `"midpoint"` | `"edge"` |
| `distance_metric` | Metric for ranking pixels to center | `"summed color distance"` |
| `tsp_method` | TSP solver for ring ordering | `"nearest_neighbor"` |
| `shortlist_size` | Coarse filter size for candidate ranking | 50 |
| `seed_position` | Starting pixel | `"random"` |

## Future Extensions

- **Adaptive radius:** Use larger pinwheels in low-entropy regions (smooth gradients) and smaller pinwheels in high-entropy regions (edges, texture).
- **Multi-scale:** Build coarse pinwheels at low resolution first, then refine with smaller pinwheels at full resolution.
- **Annealing rotation:** Instead of greedy best-rotation, use simulated annealing over rotation states across multiple pinwheels to escape local minima.
- **Diagonal expansion:** Expand pinwheels in 8 directions (including diagonals) instead of 4 cardinals for denser coverage.
