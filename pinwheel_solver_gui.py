#!/usr/bin/env python
"""
Pinwheel Solver GUI — Reconstructs scrambled video by building concentric
ring structures (pinwheels) and tiling them onto a pinboard.

Each pinwheel:
  1. Ranks unplaced pixels by distance to its center
  2. Assigns them to concentric rings (same counts as integer lattice points)
  3. Orders each ring via nearest-neighbor TSP using pairwise dissonance
  4. Rotates each ring to minimize dissonance with inner rings & pinboard neighbors

Pinwheels are tiled in BFS order; each grid position is owned by the
nearest pinwheel center (Voronoi).

Usage:
    python pinwheel_solver_gui.py -v video.mkv -f 100 --scramble-seed 42
"""

import argparse
import math
import os
import sys
import time
import threading
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

from tv_wall import TVWall

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QGroupBox, QTextEdit,
    QProgressBar, QFileDialog, QMessageBox, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ──────────────────────────────────────────────────────────────────────
# Lattice helpers — build concentric ring pixel positions
# ──────────────────────────────────────────────────────────────────────

def build_lattice_rings(max_radius):
    """Return a list of rings. Each ring is a sorted list of (dx, dy) offsets
    at the same rounded integer distance from the origin.

    Ring 0 is the center (just the origin).
    """
    buckets = defaultdict(list)
    for dy in range(-max_radius - 1, max_radius + 2):
        for dx in range(-max_radius - 1, max_radius + 2):
            r = round(math.sqrt(dx * dx + dy * dy))
            if r <= max_radius:
                buckets[r].append((dx, dy))

    rings = []
    for r in sorted(buckets.keys()):
        pts = buckets[r]
        # Sort by angle from north, clockwise (same convention as the HTML viz)
        pts.sort(key=lambda p: (math.atan2(p[1], p[0]) + math.pi / 2) % (2 * math.pi))
        rings.append(pts)
    return rings


# ──────────────────────────────────────────────────────────────────────
# TSP solver (nearest-neighbour heuristic + optional 2-opt)
# ──────────────────────────────────────────────────────────────────────

def _nn_tsp(dist_matrix):
    """Nearest-neighbour TSP on a full distance matrix. Returns index ordering."""
    n = len(dist_matrix)
    if n <= 1:
        return list(range(n))
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = order[-1]
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist_matrix[last, j] < best_dist:
                best_dist = dist_matrix[last, j]
                best_next = j
        order.append(best_next)
        visited[best_next] = True
    return order


def _two_opt(order, dist_matrix, max_iters=500):
    """Improve a TSP tour via 2-opt (cyclic tour)."""
    n = len(order)
    if n < 4:
        return order

    def tour_len(o):
        return sum(dist_matrix[o[i], o[(i + 1) % n]] for i in range(n))

    best_len = tour_len(order)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # skip full reversal
                new_order = order[:i + 1] + order[i + 1:j + 1][::-1] + order[j + 1:]
                new_len = tour_len(new_order)
                if new_len < best_len - 1e-9:
                    order = new_order
                    best_len = new_len
                    improved = True
    return order


def solve_ring_tsp(pixel_indices, series_flat, use_2opt=True):
    """Given a list of flat pixel indices and all series, solve TSP ordering.

    Returns the re-ordered list of pixel indices.
    """
    n = len(pixel_indices)
    if n <= 2:
        return list(pixel_indices)

    # Build pairwise distance matrix (flattened Euclidean)
    series = series_flat[pixel_indices]  # (n, 3, T)
    flat = series.reshape(n, -1).astype(np.float64)
    # cdist would work but keep it numpy-only
    sq = np.sum(flat ** 2, axis=1)
    dist_matrix = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * flat @ flat.T, 0.0))

    order = _nn_tsp(dist_matrix)
    if use_2opt and n <= 60:
        order = _two_opt(order, dist_matrix)

    return [pixel_indices[i] for i in order]


# ──────────────────────────────────────────────────────────────────────
# Pinwheel
# ──────────────────────────────────────────────────────────────────────

class Pinwheel:
    """A concentric-ring pixel cluster centered on a grid position.

    Pixels are selected from the **entire** pool (including those already
    claimed by other pinwheels).  The final ownership of each grid cell is
    decided by Voronoi (nearest pinwheel centre), so duplicates across
    pinwheels are expected and harmless.
    """

    def __init__(self, center_grid_pos, lattice_rings, series_flat, mean_colors,
                 distance_metric='euclidean', use_2opt=True):
        """
        Parameters
        ----------
        center_grid_pos : (int, int)
            Output grid position for the pinwheel center.
        lattice_rings : list[list[(dx,dy)]]
            Pre-built ring offsets from build_lattice_rings().
        series_flat : (N, 3, T) float32
            All pixel time-series (in scrambled order).
        mean_colors : (N, 3) float32
            Mean color per pixel (for fast shortlisting).
        distance_metric : str
            'euclidean' (default).
        use_2opt : bool
            Apply 2-opt improvement to TSP orderings.
        """
        self.center_pos = center_grid_pos
        self.rings = []           # list of lists of flat pixel indices per ring
        self.ring_offsets = []    # list of lists of (dx, dy) per ring
        self.ring_rotations = []  # int rotation applied per ring
        self.pixel_grid = {}      # (gx, gy) -> flat pixel index  (after placement)

        # Total pixel budget = sum of ring sizes
        total_needed = sum(len(r) for r in lattice_rings)

        self._lattice_rings = lattice_rings
        self._series_flat = series_flat
        self._mean_colors = mean_colors
        self._metric = distance_metric
        self._use_2opt = use_2opt
        self._total_needed = total_needed

    # ── Public API ──────────────────────────────────────────────────

    def build(self, center_pixel_idx, placed_series_at=None, step_callback=None):
        """Populate rings by selecting the closest pixels from the full pool.

        Pixels may be shared across pinwheels — the solver's Voronoi
        resolution phase deduplicates later using ``sorted_candidates``
        as a fallback list.

        Parameters
        ----------
        center_pixel_idx : int
            Flat index of the pixel to place at the center.
        placed_series_at : dict[(gx,gy)] -> (3,T) array, optional
            Series already placed on the pinboard (for rotation optimisation).
        step_callback : callable, optional
            Called after each sub-step with a dict describing current state.
        """
        sf = self._series_flat
        N = len(sf)
        center_series = sf[center_pixel_idx]  # (3, T)
        total_rings = len(self._lattice_rings)

        def _emit(phase, ring_idx=-1, **extra):
            if step_callback is None:
                return
            state = {
                'phase': phase,
                'ring_idx': ring_idx,
                'total_rings': total_rings,
                'center_pos': self.center_pos,
                'rings': [list(r) for r in self.rings],
                'ring_offsets': [list(r) for r in self.ring_offsets],
                'ring_rotations': list(self.ring_rotations) if self.ring_rotations else [0] * len(self.rings),
            }
            state.update(extra)
            step_callback(state)

        # ── 2. Rank ALL pixels by distance to center ──
        needed = self._total_needed - 1  # ring slots minus center

        # Fast two-stage shortlisting: mean-RGB coarse -> full series refine
        # Stage 1: coarse filter by mean color distance
        center_mean = self._mean_colors[center_pixel_idx]  # (3,)
        shortlist_size = min(max(needed * 3, 500), N)
        color_dists = np.sum((self._mean_colors - center_mean) ** 2, axis=1)  # (N,)

        if shortlist_size < N:
            top_coarse = np.argpartition(color_dists, shortlist_size)[:shortlist_size]
        else:
            top_coarse = np.arange(N)

        # Stage 2: full time-series distance on shortlisted candidates
        shortlisted = top_coarse  # already flat pixel indices
        diff = sf[shortlisted] - center_series[np.newaxis, :, :]
        full_dists = np.sum(diff ** 2, axis=(1, 2))  # skip sqrt for ranking
        ranked_order = np.argsort(full_dists)
        # Sorted flat pixel indices (closest first), excluding center itself
        sorted_candidates = shortlisted[ranked_order]
        sorted_candidates = sorted_candidates[sorted_candidates != center_pixel_idx]

        # Store for later dedup (Voronoi resolution can walk this list)
        self.sorted_candidates = sorted_candidates

        # Emit: candidates ranked
        self.rings = []
        self.ring_offsets = []
        self.ring_rotations = []
        _emit('candidates_ranked', top_candidates=sorted_candidates[:50].tolist())

        # ── 3. Assign to rings ──
        pos = 0  # pointer into sorted_candidates

        for ring_idx, ring_pts in enumerate(self._lattice_rings):
            n_ring = len(ring_pts)
            if ring_idx == 0:
                self.rings.append([center_pixel_idx])
                self.ring_offsets.append(ring_pts)
                self.ring_rotations.append(0)
                continue

            # Take the next n_ring closest pixels
            end = min(pos + n_ring, len(sorted_candidates))
            ring_pixels = sorted_candidates[pos:end].tolist()
            pos = end

            if not ring_pixels:
                break

            # Pad if we ran out
            while len(ring_pixels) < n_ring and ring_pixels:
                ring_pixels.append(ring_pixels[-1])

            self.rings.append(ring_pixels)
            self.ring_offsets.append(ring_pts)
            self.ring_rotations.append(0)

            # Emit: ring assigned (before TSP)
            _emit('ring_assigned', ring_idx=ring_idx)

        # ── 4. TSP ordering within each ring ──
        for i in range(1, len(self.rings)):
            if len(self.rings[i]) > 2:
                self.rings[i] = solve_ring_tsp(
                    self.rings[i], self._series_flat, self._use_2opt
                )
                # Emit: ring TSP ordered
                _emit('ring_tsp', ring_idx=i)

        # ── 5. Rotate each ring for best dissonance ──
        self.ring_rotations = [0] * len(self.rings)
        self._optimise_rotations(placed_series_at, step_callback=step_callback)

        # ── 6. Assign pixels to grid positions ──
        self._assign_grid_positions()

    # ── Rotation optimisation ───────────────────────────────────────

    def _optimise_rotations(self, placed_series_at=None, step_callback=None):
        """Try every discrete rotation for each ring (from inside out) and
        keep the rotation that minimises total dissonance with already-placed
        inner rings and pinboard neighbours."""
        gx0, gy0 = self.center_pos
        total_rings = len(self._lattice_rings)

        # Already-placed lookup: grid pos -> series
        placed = {}
        if placed_series_at:
            placed.update(placed_series_at)

        for ring_idx in range(len(self.rings)):
            offsets = self.ring_offsets[ring_idx]
            pixels = self.rings[ring_idx]
            n = len(pixels)
            if n <= 1:
                self.ring_rotations[ring_idx] = 0
                # Register in placed
                if pixels:
                    gx, gy = gx0 + offsets[0][0], gy0 + offsets[0][1]
                    placed[(gx, gy)] = self._series_flat[pixels[0]]
                continue

            best_rot = 0
            best_cost = float('inf')

            for rot in range(n):
                cost = 0.0
                count = 0
                for i, pix_idx in enumerate(pixels):
                    slot = (i + rot) % n
                    dx, dy = offsets[slot]
                    gx, gy = gx0 + dx, gy0 + dy
                    pix_series = self._series_flat[pix_idx]

                    # Check 4 cardinal neighbours in *placed*
                    for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        npos = (gx + ndx, gy + ndy)
                        if npos in placed:
                            diff = pix_series - placed[npos]
                            cost += float(np.sqrt(np.sum(diff ** 2)))
                            count += 1

                if count > 0:
                    cost /= count
                if cost < best_cost:
                    best_cost = cost
                    best_rot = rot

            self.ring_rotations[ring_idx] = best_rot

            # Register this ring in placed for outer rings to reference
            for i, pix_idx in enumerate(pixels):
                slot = (i + best_rot) % n
                dx, dy = offsets[slot]
                gx, gy = gx0 + dx, gy0 + dy
                placed[(gx, gy)] = self._series_flat[pix_idx]

            # Emit: ring rotated
            if step_callback is not None:
                state = {
                    'phase': 'ring_rotated',
                    'ring_idx': ring_idx,
                    'total_rings': total_rings,
                    'center_pos': self.center_pos,
                    'rings': [list(r) for r in self.rings],
                    'ring_offsets': [list(r) for r in self.ring_offsets],
                    'ring_rotations': list(self.ring_rotations),
                    'best_rot': best_rot,
                }
                step_callback(state)

    def _assign_grid_positions(self):
        """Write pixel_grid dict: (gx,gy) -> flat pixel index."""
        gx0, gy0 = self.center_pos
        self.pixel_grid = {}
        for ring_idx in range(len(self.rings)):
            pixels = self.rings[ring_idx]
            offsets = self.ring_offsets[ring_idx]
            rot = self.ring_rotations[ring_idx]
            n = len(pixels)
            for i, pix_idx in enumerate(pixels):
                slot = (i + rot) % n
                if slot < len(offsets):
                    dx, dy = offsets[slot]
                    gx, gy = gx0 + dx, gy0 + dy
                    self.pixel_grid[(gx, gy)] = pix_idx

    # ── Helpers ─────────────────────────────────────────────────────

    def get_boundary_positions(self):
        """Return set of grid positions on the outer edge of this pinwheel."""
        if not self.rings:
            return set()
        last_ring_idx = len(self.rings) - 1
        offsets = self.ring_offsets[last_ring_idx]
        rot = self.ring_rotations[last_ring_idx]
        gx0, gy0 = self.center_pos
        boundary = set()
        for i, (dx, dy) in enumerate(offsets):
            gx, gy = gx0 + dx, gy0 + dy
            boundary.add((gx, gy))
        return boundary

    @property
    def radius(self):
        return len(self.rings) - 1


# ──────────────────────────────────────────────────────────────────────
# PinwheelSolver
# ──────────────────────────────────────────────────────────────────────

class PinwheelSolver:
    """BFS-based pinwheel tiling solver."""

    def __init__(self, wall: TVWall, *,
                 pinwheel_radius=5,
                 placement_mode='edge',
                 distance_metric='euclidean',
                 scramble_seed=42,
                 use_2opt=True,
                 shortlist_size=200):
        self.wall = wall
        self.H = wall.height
        self.W = wall.width
        self.T = wall.num_frames
        self.total_pixels = self.H * self.W

        self.pinwheel_radius = pinwheel_radius
        self.placement_mode = placement_mode
        self.distance_metric = distance_metric
        self.scramble_seed = scramble_seed
        self.use_2opt = use_2opt
        self.shortlist_size = shortlist_size

        # Pre-built lattice rings
        self.lattice_rings = build_lattice_rings(pinwheel_radius)
        self.pixels_per_pinwheel = sum(len(r) for r in self.lattice_rings)

        # State filled during solve
        self.all_series = None     # (H, W, 3, T)
        self.series_flat = None    # (H*W, 3, T)
        self.mean_colors = None    # (H*W, 3)

        # Output grid: (H, W) -> flat pixel idx, -1 = unplaced
        self.output_grid = None

        # Pinwheel list and Voronoi centers
        self.pinwheels = []
        self.pinwheel_centers = []   # (gx, gy) for each pinwheel
        self.placed_series = {}      # (gx,gy) -> (3,T) for rotation context

        # BFS frontier of pinwheel center candidates
        self.frontier = []

        # Stats
        self.stats = {
            'pinwheels_placed': 0,
            'pixels_placed': 0,
            'correct_placements': 0,
            'elapsed': 0.0,
        }
        self._stop = False

        # Formation visualization state
        self._formation_enabled = False
        self._formation_callback = None   # set by GUI
        self._current_formation = None    # latest formation step dict

    # ── Precompute ──────────────────────────────────────────────────

    def _precompute(self):
        print("Precomputing colour series...")
        self.all_series = self.wall.get_all_series(force_cpu=True)  # (H,W,3,T)
        self.series_flat = self.all_series.reshape(self.total_pixels, 3, self.T)
        self.mean_colors = self.series_flat.mean(axis=2)  # (N, 3)
        self.output_grid = np.full((self.H, self.W), -1, dtype=np.int32)
        self.placed_series = {}

    # ── Find best center pixel for a given grid position ───────────

    def _find_center_pixel(self, grid_pos, reference_indices=None):
        """Pick the best pixel for the pinwheel centered at grid_pos.

        Searches the full pixel pool (pixels may be reused across pinwheels).
        If reference_indices is given (flat pixel indices from neighbouring
        pinwheels), picks the pixel closest to their mean series.
        Otherwise picks a random pixel.
        """
        N = self.total_pixels

        if reference_indices is not None and len(reference_indices) > 0:
            ref_series = self.series_flat[reference_indices]  # (K, 3, T)
            ref_mean = ref_series.mean(axis=0)  # (3, T)
            # Shortlist by mean-RGB proximity
            ref_color = ref_mean.mean(axis=1)  # (3,)
            color_dists = np.sum((self.mean_colors - ref_color) ** 2, axis=1)  # (N,)
            S = min(self.shortlist_size, N - 1)
            if S < N:
                top_idx = np.argpartition(color_dists, S)[:S]
            else:
                top_idx = np.arange(N)
            # Refine with full time-series distance
            diffs = self.series_flat[top_idx] - ref_mean[np.newaxis, :, :]
            full_dists = np.sum(diffs ** 2, axis=(1, 2))
            best = top_idx[np.argmin(full_dists)]
            return int(best)
        else:
            # Random pixel (first solve call)
            rng = np.random.RandomState(self.scramble_seed)
            return int(rng.randint(0, N))

    # ── Place a pinwheel ────────────────────────────────────────────

    def _place_pinwheel(self, grid_pos, center_pixel_idx):
        """Build a pinwheel at grid_pos.  Does NOT write to output_grid yet —
        the Voronoi resolution phase handles that with deduplication."""
        pw = Pinwheel(
            center_grid_pos=grid_pos,
            lattice_rings=self.lattice_rings,
            series_flat=self.series_flat,
            mean_colors=self.mean_colors,
            distance_metric=self.distance_metric,
            use_2opt=self.use_2opt,
        )
        step_cb = self._formation_callback if self._formation_enabled else None
        pw.build(center_pixel_idx, placed_series_at=self.placed_series,
                 step_callback=step_cb)

        # Register the pinwheel's ideal placements into placed_series
        # so that *future* pinwheels can use them for rotation context.
        for (gx, gy), pix_idx in pw.pixel_grid.items():
            if 0 <= gx < self.W and 0 <= gy < self.H:
                self.placed_series[(gx, gy)] = self.series_flat[pix_idx]

        self.pinwheels.append(pw)
        self.pinwheel_centers.append(grid_pos)
        self.stats['pinwheels_placed'] += 1
        self.stats['pixels_placed'] = sum(
            sum(1 for (gx, gy) in pw2.pixel_grid
                if 0 <= gx < self.W and 0 <= gy < self.H)
            for pw2 in self.pinwheels
        )

        # Expand frontier (spawn neighbours)
        self._expand_frontier(grid_pos, pw)

        return pw

    def _expand_frontier(self, grid_pos, pw):
        """Add cardinal neighbour positions as new pinwheel center candidates."""
        gx0, gy0 = grid_pos
        if self.placement_mode == 'midpoint':
            step = self.pinwheel_radius
        else:  # edge
            step = self.pinwheel_radius * 2

        existing_centers = set(self.pinwheel_centers)
        frontier_set = set(self.frontier)

        for ddx, ddy in [(-step, 0), (step, 0), (0, -step), (0, step)]:
            nx, ny = gx0 + ddx, gy0 + ddy
            if (nx, ny) not in existing_centers and (nx, ny) not in frontier_set:
                # Only add if the center would be at least partially on-grid
                if -self.pinwheel_radius <= nx < self.W + self.pinwheel_radius and \
                   -self.pinwheel_radius <= ny < self.H + self.pinwheel_radius:
                    self.frontier.append((nx, ny))

    # ── Solve ───────────────────────────────────────────────────────

    def solve(self, callback=None):
        """Run the full solver. Returns output_grid (H, W) of flat pixel indices."""
        self._stop = False
        t0 = time.time()

        self._precompute()

        # Phase 1: Seed pinwheel at grid center
        center_gx, center_gy = self.W // 2, self.H // 2
        seed_pix = self._find_center_pixel((center_gx, center_gy))
        if seed_pix is None:
            return self.output_grid

        self._place_pinwheel((center_gx, center_gy), seed_pix)
        if callback:
            callback(self._make_step_info())

        # Phase 2: BFS expansion — build all pinwheels
        est_total_pw = max(1, self.total_pixels // max(1, self.pixels_per_pinwheel))
        with tqdm(total=est_total_pw, desc="Pinwheel BFS", initial=1) as pbar:
            while self.frontier and not self._stop:
                next_pos = self.frontier.pop(0)

                if next_pos in set(self.pinwheel_centers):
                    continue

                ref_pixels = self._get_reference_pixels(next_pos)
                center_pix = self._find_center_pixel(next_pos, ref_pixels)
                if center_pix is None:
                    continue

                pw = self._place_pinwheel(next_pos, center_pix)
                pbar.update(1)

                if callback:
                    callback(self._make_step_info())

        # Phase 3: Voronoi resolution — deduplicate pixel assignments
        print("Phase 3: Voronoi resolution (dedup)...")
        self._resolve_voronoi()
        if callback:
            callback(self._make_step_info())

        self.stats['elapsed'] = time.time() - t0
        self._count_correct()

        return self.output_grid

    # ── Voronoi resolution with deduplication ───────────────────────

    def _resolve_voronoi(self):
        """Assign each on-grid cell to its nearest pinwheel center, then
        walk each pinwheel's ranked candidate list to produce a unique
        pixel assignment (each flat pixel index used at most once).

        Cells closer to their pinwheel center get first pick.
        """
        K = len(self.pinwheel_centers)
        if K == 0:
            return

        centers = np.array(self.pinwheel_centers, dtype=np.float32)  # (K, 2)

        # ── 1. Build Voronoi map: for every on-grid cell, which pinwheel owns it
        gy_grid, gx_grid = np.mgrid[0:self.H, 0:self.W]  # (H,W)
        # (H, W) distance^2 to each center; find argmin
        best_pw = np.zeros((self.H, self.W), dtype=np.int32)
        best_dist_sq = np.full((self.H, self.W), np.inf, dtype=np.float32)
        for k in range(K):
            cx, cy = centers[k]
            d2 = (gx_grid - cx) ** 2 + (gy_grid - cy) ** 2
            mask = d2 < best_dist_sq
            best_dist_sq[mask] = d2[mask]
            best_pw[mask] = k

        # ── 2. Build a sorted work-list: (dist_to_center, gy, gx)
        # Cells closest to their center get first pick of pixels.
        work = []
        for gy in range(self.H):
            for gx in range(self.W):
                work.append((float(best_dist_sq[gy, gx]), gy, gx))
        work.sort()

        # ── 3. For each pinwheel, build a per-slot candidate iterator
        #    The pinwheel's pixel_grid gives the "ideal" pixel for each
        #    grid pos.  If that pixel is taken, we fall back to
        #    sorted_candidates (the full ranked list for that center).
        used_pixels = set()
        self.output_grid[:] = -1

        # Pre-build per-pinwheel fallback iterators (index into sorted_candidates)
        pw_fallback_pos = [0] * K  # current position in sorted_candidates

        for _, gy, gx in work:
            k = best_pw[gy, gx]
            pw = self.pinwheels[k]

            # First try: the ideal pixel from this pinwheel's layout
            ideal_pix = pw.pixel_grid.get((gx, gy), None)
            if ideal_pix is not None and ideal_pix not in used_pixels:
                self.output_grid[gy, gx] = ideal_pix
                used_pixels.add(ideal_pix)
                continue

            # Second try: walk the pinwheel's sorted_candidates for the
            # next unused pixel (maintains similarity to center).
            candidates = pw.sorted_candidates
            pos = pw_fallback_pos[k]
            placed = False
            while pos < len(candidates):
                cand = int(candidates[pos])
                pos += 1
                if cand not in used_pixels:
                    self.output_grid[gy, gx] = cand
                    used_pixels.add(cand)
                    placed = True
                    break
            pw_fallback_pos[k] = pos

            # Last resort: if the pinwheel's shortlist is exhausted,
            # grab any unused pixel (shouldn't happen often).
            if not placed:
                for pix in range(self.total_pixels):
                    if pix not in used_pixels:
                        self.output_grid[gy, gx] = pix
                        used_pixels.add(pix)
                        break

        self.stats['pixels_placed'] = int(np.sum(self.output_grid >= 0))

    def _get_reference_pixels(self, grid_pos):
        """Get flat pixel indices from already-placed pinwheels near grid_pos
        to serve as references for centre-pixel selection."""
        gx, gy = grid_pos
        ref = []
        # Look at the edge of existing pinwheels closest to this position
        # by scanning the pixel_grid of nearby pinwheels.
        for pw in self.pinwheels:
            cx, cy = pw.center_pos
            # Only consider pinwheels whose rings could reach grid_pos
            dist = abs(cx - gx) + abs(cy - gy)
            if dist > self.pinwheel_radius * 3:
                continue
            for (pgx, pgy), pix_idx in pw.pixel_grid.items():
                if abs(pgx - gx) <= self.pinwheel_radius and \
                   abs(pgy - gy) <= self.pinwheel_radius:
                    ref.append(pix_idx)
        return ref if ref else None

    def _count_correct(self):
        """Count how many output grid positions have the correct pixel."""
        correct = 0
        for gy in range(self.H):
            for gx in range(self.W):
                pix_idx = self.output_grid[gy, gx]
                if pix_idx < 0:
                    continue
                src_x = pix_idx % self.W
                src_y = pix_idx // self.W
                orig_x = int(self.wall._perm_x[src_y, src_x])
                orig_y = int(self.wall._perm_y[src_y, src_x])
                if orig_x == gx and orig_y == gy:
                    correct += 1
        self.stats['correct_placements'] = correct

    def _make_step_info(self):
        return dict(self.stats)

    def apply_solution(self):
        """Write the output grid back into the TVWall permutation."""
        old_px = self.wall._perm_x.copy()
        old_py = self.wall._perm_y.copy()
        new_px = np.zeros_like(old_px)
        new_py = np.zeros_like(old_py)
        identity_y, identity_x = np.mgrid[0:self.H, 0:self.W]

        for gy in range(self.H):
            for gx in range(self.W):
                pix_idx = self.output_grid[gy, gx]
                if pix_idx >= 0:
                    src_x = pix_idx % self.W
                    src_y = pix_idx // self.W
                    new_px[gy, gx] = old_px[src_y, src_x]
                    new_py[gy, gx] = old_py[src_y, src_x]
                else:
                    new_px[gy, gx] = identity_x[gy, gx]
                    new_py[gy, gx] = identity_y[gy, gx]

        self.wall._perm_x = new_px
        self.wall._perm_y = new_py

    def stop(self):
        self._stop = True

    def get_stats(self):
        placed = self.stats['pixels_placed']
        correct = self.stats['correct_placements']
        return {
            'total_pixels': self.total_pixels,
            'pixels_placed': placed,
            'pinwheels_placed': self.stats['pinwheels_placed'],
            'correct_placements': correct,
            'accuracy': 100.0 * correct / max(1, placed),
            'elapsed': self.stats['elapsed'],
            'pixels_per_sec': placed / max(0.001, self.stats['elapsed']),
        }

    # ── Build an image of the current state ─────────────────────────

    def get_state_image(self, frame_idx=0):
        """Return a PIL Image of the current state coloured by frame_idx.

        During BFS (before Voronoi), shows the pinwheels' ideal pixel_grids
        (may contain duplicates — that's fine for preview).
        After Voronoi, shows the deduplicated output_grid.
        """
        img = np.full((self.H, self.W, 3), 40, dtype=np.uint8)
        if self.all_series is None:
            return Image.fromarray(img, 'RGB')

        # After Voronoi: output_grid is populated
        placed_mask = self.output_grid >= 0
        if placed_mask.any():
            ys, xs = np.where(placed_mask)
            pix_indices = self.output_grid[ys, xs]
            src_xs = pix_indices % self.W
            src_ys = pix_indices // self.W
            colours = self.all_series[src_ys, src_xs, :, frame_idx].astype(np.uint8)
            img[ys, xs] = colours
        else:
            # During BFS: build preview from pinwheel pixel_grids
            for pw in self.pinwheels:
                for (gx, gy), pix_idx in pw.pixel_grid.items():
                    if 0 <= gx < self.W and 0 <= gy < self.H:
                        src_x = pix_idx % self.W
                        src_y = pix_idx // self.W
                        img[gy, gx] = self.all_series[src_y, src_x, :, frame_idx].astype(np.uint8)

        return Image.fromarray(img, 'RGB')

    def get_voronoi_image(self):
        """Return a PIL Image where each pixel is coloured by its nearest
        pinwheel center (centre hue)."""
        if not self.pinwheel_centers:
            return Image.new('RGB', (self.W, self.H), (40, 40, 40))

        centers = np.array(self.pinwheel_centers, dtype=np.float32)  # (K, 2)
        img = np.full((self.H, self.W, 3), 40, dtype=np.uint8)

        # Build a Voronoi map
        gy_grid, gx_grid = np.mgrid[0:self.H, 0:self.W]
        for idx, (cx, cy) in enumerate(self.pinwheel_centers):
            # Simple distance — we'll assign after looping
            pass

        # Vectorised: compute distance from each pixel to each center
        K = len(self.pinwheel_centers)
        cx_arr = centers[:, 0]  # (K,)
        cy_arr = centers[:, 1]  # (K,)

        # (H, W) distances to each center
        # Do this in chunks to save memory
        best_center = np.zeros((self.H, self.W), dtype=np.int32)
        best_dist = np.full((self.H, self.W), np.inf, dtype=np.float32)

        for k in range(K):
            d = (gx_grid - cx_arr[k]) ** 2 + (gy_grid - cy_arr[k]) ** 2
            mask = d < best_dist
            best_dist[mask] = d[mask]
            best_center[mask] = k

        # Colour each region by center index hue
        for k in range(K):
            hue = (k * 47) % 360
            r, g, b = self._hsv_to_rgb(hue / 360, 0.6, 0.9)
            mask = best_center == k
            img[mask] = [int(r * 255), int(g * 255), int(b * 255)]

        return Image.fromarray(img, 'RGB')

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)

    def get_pinwheel_overlay_image(self, frame_idx=0):
        """Return the state image with pinwheel center markers and ring outlines."""
        base = self.get_state_image(frame_idx)
        img = np.array(base)

        # Draw pinwheel center markers
        for cx, cy in self.pinwheel_centers:
            if 0 <= cx < self.W and 0 <= cy < self.H:
                # Draw a bright cross
                for d in range(-2, 3):
                    for dx, dy in [(d, 0), (0, d)]:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < self.W and 0 <= py < self.H:
                            img[py, px] = [255, 255, 0]

        return Image.fromarray(img, 'RGB')

    def get_formation_image(self, frame_idx=0, formation_state=None):
        """Return a PIL Image showing the current pinwheel being formed.

        Renders all previously completed pinwheels as background, then overlays
        the partial state of the current pinwheel with per-ring visual
        distinction: assignment, TSP ordering, and rotation.
        """
        img = np.full((self.H, self.W, 3), 40, dtype=np.uint8)
        if self.all_series is None:
            return Image.fromarray(img, 'RGB')

        # ── Background: all completed pinwheels ──
        for pw in self.pinwheels:
            for (gx, gy), pix_idx in pw.pixel_grid.items():
                if 0 <= gx < self.W and 0 <= gy < self.H:
                    src_x = pix_idx % self.W
                    src_y = pix_idx // self.W
                    img[gy, gx] = self.all_series[src_y, src_x, :, frame_idx].astype(np.uint8)

        if formation_state is None:
            return Image.fromarray(img, 'RGB')

        phase = formation_state['phase']
        center_pos = formation_state['center_pos']
        rings = formation_state['rings']
        ring_offsets = formation_state['ring_offsets']
        ring_rotations = formation_state['ring_rotations']
        current_ring = formation_state.get('ring_idx', -1)
        gx0, gy0 = center_pos

        # ── Draw center cross (yellow) ──
        for d in range(-2, 3):
            for dx, dy in [(d, 0), (0, d)]:
                px, py = gx0 + dx, gy0 + dy
                if 0 <= px < self.W and 0 <= py < self.H:
                    img[py, px] = [255, 255, 0]

        if phase == 'candidates_ranked':
            # Show top candidates as a heatmap
            top_cands = formation_state.get('top_candidates', [])
            n_cands = len(top_cands)
            for rank, pix_idx in enumerate(top_cands):
                src_x = pix_idx % self.W
                src_y = pix_idx // self.W
                # Map scrambled pixel index to its current grid position
                # pix_idx is a flat index into the scrambled grid
                gx_c = pix_idx % self.W
                gy_c = pix_idx // self.W
                if 0 <= gx_c < self.W and 0 <= gy_c < self.H:
                    # Brightness by rank: closest = bright, farthest = dim
                    brightness = max(0.2, 1.0 - rank / max(1, n_cands))
                    base_color = self.all_series[src_y, src_x, :, frame_idx].astype(np.float32)
                    # Tint towards white
                    tinted = base_color * (1.0 - brightness * 0.5) + 255 * brightness * 0.5
                    img[gy_c, gx_c] = np.clip(tinted, 0, 255).astype(np.uint8)
            return Image.fromarray(img, 'RGB')

        # ── Determine which rings are complete vs current ──
        # For ring_assigned: rings 0..current_ring exist. current_ring is
        #   assigned but not yet TSP'd.
        # For ring_tsp: ring current_ring just had TSP applied.
        # For ring_rotated: ring current_ring just had rotation applied.

        # Ring color hues for highlighting
        RING_COLORS = {
            'complete': None,       # use actual pixel color
            'assigned': [0, 200, 255],   # cyan
            'tsp':      [255, 220, 50],  # yellow
            'rotated':  [50, 255, 100],  # green
        }

        for ring_idx in range(len(rings)):
            pixels = rings[ring_idx]
            offsets = ring_offsets[ring_idx] if ring_idx < len(ring_offsets) else []
            rot = ring_rotations[ring_idx] if ring_idx < len(ring_rotations) else 0
            n = len(pixels)
            if n == 0 or not offsets:
                continue

            # Determine ring state
            if phase == 'ring_assigned':
                if ring_idx < current_ring:
                    ring_state = 'complete'
                elif ring_idx == current_ring:
                    ring_state = 'assigned'
                    rot = 0  # no rotation yet
                else:
                    continue  # not built yet
            elif phase == 'ring_tsp':
                if ring_idx < current_ring:
                    ring_state = 'complete'
                elif ring_idx == current_ring:
                    ring_state = 'tsp'
                    rot = 0  # no rotation yet
                else:
                    continue
            elif phase == 'ring_rotated':
                if ring_idx < current_ring:
                    ring_state = 'complete'
                elif ring_idx == current_ring:
                    ring_state = 'rotated'
                else:
                    continue
            else:
                ring_state = 'complete'

            # Draw pixels at their grid positions
            for i, pix_idx in enumerate(pixels):
                slot = (i + rot) % n
                if slot >= len(offsets):
                    continue
                dx, dy = offsets[slot]
                gx, gy = gx0 + dx, gy0 + dy
                if not (0 <= gx < self.W and 0 <= gy < self.H):
                    continue

                src_x = pix_idx % self.W
                src_y = pix_idx // self.W
                base_color = self.all_series[src_y, src_x, :, frame_idx].astype(np.float32)

                if ring_state == 'complete' or ring_idx == 0:
                    img[gy, gx] = base_color.astype(np.uint8)
                else:
                    # Blend with highlight color
                    highlight = np.array(RING_COLORS[ring_state], dtype=np.float32)
                    blended = base_color * 0.6 + highlight * 0.4
                    img[gy, gx] = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(img, 'RGB')


# ──────────────────────────────────────────────────────────────────────
# PyQt5 GUI
# ──────────────────────────────────────────────────────────────────────

class WorkerSignals(QObject):
    progress = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    update_display = pyqtSignal()
    formation_step = pyqtSignal(dict)


class PinwheelSolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pinwheel Solver — Unscramble Video")
        self.setGeometry(50, 50, 1600, 900)

        self.wall = None
        self.solver = None
        self.solver_thread = None
        self.solver_running = False

        self.display_scale = 2
        self.view_mode = 'state'

        # Data for graphs
        self.pixels_placed_history = []
        self.accuracy_history = []
        self.pinwheels_placed_history = []

        # Formation visualization state
        self._formation_state = None          # latest formation step dict
        self._formation_step_ack = threading.Event()
        self._formation_step_ack.set()        # start unblocked
        self._formation_delay_ms = 150        # delay between steps

        self.signals = WorkerSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)
        self.signals.update_display.connect(self._update_display)
        self.signals.formation_step.connect(self._on_formation_step)

        self._setup_ui()

    # ── UI ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # ── Left panel: controls ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(5)

        # Load video
        load_group = QGroupBox("Load Video")
        load_lay = QVBoxLayout(load_group)
        self.load_btn = QPushButton("Open Video...")
        self.load_btn.clicked.connect(self._load_video)
        load_lay.addWidget(self.load_btn)
        self.video_label = QLabel("No video loaded")
        self.video_label.setWordWrap(True)
        load_lay.addWidget(self.video_label)
        left_layout.addWidget(load_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_lay = QVBoxLayout(param_group)

        params = [
            ("Frames:", "frames_edit", "1000"),
            ("Stride:", "stride_edit", "100"),
            ("Crop %:", "crop_edit", "50"),
            ("Pinwheel Radius:", "radius_edit", "5"),
            ("Scramble Seed:", "seed_edit", "420"),
            ("Shortlist:", "shortlist_edit", "200"),
        ]
        for label_text, attr, default in params:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(110)
            row.addWidget(lbl)
            ed = QLineEdit(default)
            ed.setFixedWidth(80)
            setattr(self, attr, ed)
            row.addWidget(ed)
            row.addStretch()
            param_lay.addLayout(row)

        # Placement mode
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Placement:")
        mode_lbl.setFixedWidth(110)
        mode_row.addWidget(mode_lbl)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["midpoint", "edge"])
        self.mode_combo.setFixedWidth(100)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        param_lay.addLayout(mode_row)

        # Metric
        metric_row = QHBoxLayout()
        metric_lbl = QLabel("Metric:")
        metric_lbl.setFixedWidth(110)
        metric_row.addWidget(metric_lbl)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["euclidean", "squared", "manhattan", "cosine"])
        self.metric_combo.setFixedWidth(100)
        metric_row.addWidget(self.metric_combo)
        metric_row.addStretch()
        param_lay.addLayout(metric_row)

        left_layout.addWidget(param_group)

        # Actions
        action_group = QGroupBox("Actions")
        action_lay = QVBoxLayout(action_group)

        self.scramble_btn = QPushButton("Scramble")
        self.scramble_btn.clicked.connect(self._scramble)
        action_lay.addWidget(self.scramble_btn)

        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._start_solve)
        action_lay.addWidget(self.solve_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_solve)
        self.stop_btn.setEnabled(False)
        action_lay.addWidget(self.stop_btn)

        self.apply_btn = QPushButton("Apply Solution")
        self.apply_btn.clicked.connect(self._apply_solution)
        self.apply_btn.setEnabled(False)
        action_lay.addWidget(self.apply_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        action_lay.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        action_lay.addWidget(self.status_label)

        left_layout.addWidget(action_group)

        # View controls
        view_group = QGroupBox("View")
        view_lay = QVBoxLayout(view_group)

        view_row = QHBoxLayout()
        view_lbl = QLabel("Mode:")
        view_lbl.setFixedWidth(50)
        view_row.addWidget(view_lbl)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Reconstruction", "Voronoi", "Overlay", "Formation", "Scrambled"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        view_row.addWidget(self.view_combo)
        view_lay.addLayout(view_row)

        # Formation speed
        speed_row = QHBoxLayout()
        speed_lbl = QLabel("Speed:")
        speed_lbl.setFixedWidth(50)
        speed_row.addWidget(speed_lbl)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Fast (50ms)", "Normal (150ms)", "Slow (400ms)", "Step"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_combo)
        view_lay.addLayout(speed_row)

        self.next_step_btn = QPushButton("Next Step")
        self.next_step_btn.clicked.connect(self._on_next_step)
        self.next_step_btn.setEnabled(False)
        view_lay.addWidget(self.next_step_btn)

        self.formation_label = QLabel("")
        self.formation_label.setWordWrap(True)
        self.formation_label.setStyleSheet("font-size: 10px; color: #d63384;")
        view_lay.addWidget(self.formation_label)

        left_layout.addWidget(view_group)

        # Results
        results_group = QGroupBox("Results")
        results_lay = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 9pt;"
            "background-color: #fff; color: #8b4563;"
        )
        results_lay.addWidget(self.results_text)
        left_layout.addWidget(results_group)

        left_layout.addStretch()
        scroll.setWidget(left)
        main_layout.addWidget(scroll)

        # ── Centre panel: image ─────────────────────────────────────
        centre_panel = QWidget()
        centre_layout = QVBoxLayout(centre_panel)
        centre_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.canvas.setStyleSheet(
            "background-color: #f8d7e3; border: 2px solid #ffb6c1; border-radius: 8px;"
        )
        self.canvas.setMinimumSize(400, 300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        centre_layout.addWidget(self.canvas)

        self.info_label = QLabel("")
        centre_layout.addWidget(self.info_label)

        main_layout.addWidget(centre_panel, stretch=1)

        # ── Right panel: graphs ─────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(300)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        title = QLabel("Live Metrics")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #d63384;")
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)

        self.pinwheel_count_label = QLabel("Pinwheels: 0")
        self.pinwheel_count_label.setAlignment(Qt.AlignCenter)
        self.pinwheel_count_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #d63384;"
            "background: #ffe4ec; border: 2px solid #ffb6c1;"
            "border-radius: 10px; padding: 8px;"
        )
        right_layout.addWidget(self.pinwheel_count_label)

        # Pixels placed graph
        self.fig_pixels = Figure(figsize=(3, 2.2), dpi=80, facecolor='#fff0f5')
        self.ax_pixels = self.fig_pixels.add_subplot(111)
        self.canvas_pixels = FigureCanvas(self.fig_pixels)
        self.canvas_pixels.setMinimumHeight(160)

        px_group = QGroupBox("Pixels Placed")
        px_lay = QVBoxLayout(px_group)
        px_lay.addWidget(self.canvas_pixels)
        right_layout.addWidget(px_group)

        # Accuracy graph
        self.fig_acc = Figure(figsize=(3, 2.2), dpi=80, facecolor='#fff0f5')
        self.ax_acc = self.fig_acc.add_subplot(111)
        self.canvas_acc = FigureCanvas(self.fig_acc)
        self.canvas_acc.setMinimumHeight(160)

        acc_group = QGroupBox("Accuracy %")
        acc_lay = QVBoxLayout(acc_group)
        acc_lay.addWidget(self.canvas_acc)
        right_layout.addWidget(acc_group)

        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        main_layout.addWidget(right_scroll)

        self._style_axes()

    def _style_axes(self):
        for ax in [self.ax_pixels, self.ax_acc]:
            ax.set_facecolor('#ffe4ec')
            ax.tick_params(colors='#8b4563', labelsize=7)
            for sp in ax.spines.values():
                sp.set_color('#ffb6c1')

    # ── Slots ───────────────────────────────────────────────────────

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video files (*.mkv *.mp4 *.avi *.mov);;All (*.*)"
        )
        if not path:
            return
        try:
            n_frames = int(self.frames_edit.text())
            stride = int(self.stride_edit.text())
            crop = float(self.crop_edit.text())

            self.status_label.setText("Loading video...")
            QApplication.processEvents()

            self.wall = TVWall(path, num_frames=n_frames, stride=stride,
                               crop_percent=crop)
            fname = os.path.basename(path)
            self.video_label.setText(
                f"{fname}\nSize: {self.wall.width}x{self.wall.height}\n"
                f"Frames: {self.wall.num_frames}"
            )

            # Auto-scale
            cw = self.canvas.width() or 800
            ch = self.canvas.height() or 600
            sx = cw / self.wall.width
            sy = ch / self.wall.height
            self.display_scale = max(1, min(int(min(sx, sy)), 10))

            self._update_display()
            self.status_label.setText("Video loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _scramble(self):
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        seed = int(self.seed_edit.text())
        self.wall.scramble(seed=seed)
        self.status_label.setText(f"Scrambled (seed={seed})")
        self._update_display()

    def _start_solve(self):
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        if self.solver_running:
            return

        self.pixels_placed_history.clear()
        self.accuracy_history.clear()
        self.pinwheels_placed_history.clear()

        pw_radius = int(self.radius_edit.text())
        metric = self.metric_combo.currentText()
        mode = self.mode_combo.currentText()
        seed = int(self.seed_edit.text())
        shortlist = int(self.shortlist_edit.text())

        self.solver = PinwheelSolver(
            self.wall,
            pinwheel_radius=pw_radius,
            placement_mode=mode,
            distance_metric=metric,
            scramble_seed=seed,
            shortlist_size=shortlist,
        )

        # Wire formation visualization
        formation_view = self.view_combo.currentIndex() == 3  # "Formation"
        self.solver._formation_enabled = formation_view
        self._formation_state = None
        self._formation_step_ack.set()

        if formation_view:
            self.solver._formation_callback = self._formation_step_callback

        self.solver_running = True
        self.solve_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.apply_btn.setEnabled(False)
        self.status_label.setText("Solving...")

        self.solver_thread = threading.Thread(target=self._solver_worker, daemon=True)
        self.solver_thread.start()

    def _solver_worker(self):
        try:
            def cb(info):
                self.signals.progress.emit(info)
                # Small sleep to let GUI breathe
                time.sleep(0.02)

            self.solver.solve(callback=cb)
            self.solver._count_correct()
            self.signals.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def _stop_solve(self):
        if self.solver:
            self.solver.stop()
        self.solver_running = False
        # Unblock formation thread if waiting
        self._formation_step_ack.set()

    # ── Formation visualization handlers ─────────────────────────────

    def _formation_step_callback(self, state):
        """Called from the solver thread inside Pinwheel.build().
        Emits a signal to the GUI and waits for acknowledgement."""
        if not self.solver_running:
            return
        self.signals.formation_step.emit(state)
        # Wait for GUI to process and acknowledge
        self._formation_step_ack.clear()
        self._formation_step_ack.wait(timeout=5.0)

    def _on_formation_step(self, state):
        """GUI slot: render the formation state and then acknowledge."""
        self._formation_state = state

        # Update formation info label
        phase = state.get('phase', '')
        ring_idx = state.get('ring_idx', -1)
        total_rings = state.get('total_rings', 0)
        phase_names = {
            'candidates_ranked': 'Candidates ranked',
            'ring_assigned': f'Ring {ring_idx}/{total_rings-1}: Assigned',
            'ring_tsp': f'Ring {ring_idx}/{total_rings-1}: TSP ordered',
            'ring_rotated': f'Ring {ring_idx}/{total_rings-1}: Rotated',
        }
        self.formation_label.setText(phase_names.get(phase, phase))

        # Render formation view if currently selected
        if self.view_combo.currentIndex() == 3:  # Formation
            self._update_display()
            QApplication.processEvents()

        # Determine delay
        speed_idx = self.speed_combo.currentIndex()
        if speed_idx == 3:  # Step mode
            self.next_step_btn.setEnabled(True)
            # Don't ack — wait for user to click Next Step
            return

        delays = [50, 150, 400]
        delay_ms = delays[speed_idx] if speed_idx < len(delays) else 150

        # Schedule ack after delay
        QTimer.singleShot(delay_ms, self._ack_formation_step)

    def _ack_formation_step(self):
        """Acknowledge a formation step, unblocking the solver thread."""
        self._formation_step_ack.set()

    def _on_next_step(self):
        """User clicked Next Step in step mode."""
        self.next_step_btn.setEnabled(False)
        self._formation_step_ack.set()

    def _on_view_changed(self, idx):
        """View combo changed — toggle formation mode on the solver if running."""
        formation_view = (idx == 3)
        if self.solver and self.solver_running:
            if formation_view and not self.solver._formation_enabled:
                # Enable formation mid-solve
                self.solver._formation_enabled = True
                self.solver._formation_callback = self._formation_step_callback
            elif not formation_view and self.solver._formation_enabled:
                # Disable formation mid-solve — unblock if waiting
                self.solver._formation_enabled = False
                self.solver._formation_callback = None
                self._formation_step_ack.set()
                self.next_step_btn.setEnabled(False)
        self._update_display()

    def _on_speed_changed(self, idx):
        """Speed combo changed."""
        # If switching away from Step mode and currently waiting, unblock
        if idx != 3:
            self.next_step_btn.setEnabled(False)
            self._formation_step_ack.set()

    def _on_progress(self, info):
        placed = info.get('pixels_placed', 0)
        total = info.get('pinwheels_placed', 0)
        pct = 100.0 * placed / max(1, self.solver.total_pixels) if self.solver else 0
        self.progress_bar.setValue(int(pct))

        self.pixels_placed_history.append(placed)
        self.pinwheels_placed_history.append(total)

        # Estimate accuracy from placed count (actual accuracy computed at end)
        self.accuracy_history.append(pct)

        self.pinwheel_count_label.setText(f"Pinwheels: {total}")

        self._update_graphs()
        self._update_display()
        self._update_results()

    def _on_finished(self):
        self.solver_running = False
        self.solve_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.apply_btn.setEnabled(True)
        self.next_step_btn.setEnabled(False)
        self._formation_state = None
        self._formation_step_ack.set()
        self.formation_label.setText("")

        if self.solver:
            stats = self.solver.get_stats()
            self.status_label.setText(
                f"Done! {stats['pixels_placed']} px in {stats['elapsed']:.1f}s, "
                f"Accuracy: {stats['accuracy']:.1f}%"
            )
            # Update accuracy history with real values
            if self.accuracy_history:
                self.accuracy_history[-1] = stats['accuracy']

        self._update_display()
        self._update_results()
        self._update_graphs()

    def _on_error(self, msg):
        self.solver_running = False
        self.solve_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.next_step_btn.setEnabled(False)
        self._formation_step_ack.set()
        self._formation_state = None
        self.formation_label.setText("")
        QMessageBox.critical(self, "Error", msg)

    def _apply_solution(self):
        if self.solver is None:
            return
        self.solver.apply_solution()
        self.status_label.setText("Solution applied to TVWall")
        self._update_display()

    # ── Display ─────────────────────────────────────────────────────

    def _update_display(self):
        if self.wall is None:
            return

        mode_idx = self.view_combo.currentIndex() if hasattr(self, 'view_combo') else 0
        mode_names = ['state', 'voronoi', 'overlay', 'formation', 'scrambled']
        mode = mode_names[mode_idx] if mode_idx < len(mode_names) else 'state'

        if mode == 'scrambled' or self.solver is None:
            img = self.wall.get_frame_image(0)
        elif mode == 'voronoi':
            img = self.solver.get_voronoi_image()
        elif mode == 'overlay':
            img = self.solver.get_pinwheel_overlay_image(0)
        elif mode == 'formation':
            img = self.solver.get_formation_image(0, self._formation_state)
        else:
            img = self.solver.get_state_image(0)

        # Scale
        new_w = int(self.wall.width * self.display_scale)
        new_h = int(self.wall.height * self.display_scale)
        img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

        # Convert to QPixmap
        if img.mode != 'RGB':
            img = img.convert('RGB')
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.canvas.setPixmap(pixmap)

        # Info
        if self.solver:
            s = self.solver.stats
            self.info_label.setText(
                f"Pinwheels: {s['pinwheels_placed']} | "
                f"Pixels: {s['pixels_placed']}/{self.solver.total_pixels} | "
                f"Correct: {s['correct_placements']}"
            )

    def _update_results(self):
        if self.solver is None:
            return
        stats = self.solver.get_stats()
        lines = [
            "=" * 36,
            "     PINWHEEL SOLVER RESULTS",
            "=" * 36,
            f"Total pixels:     {stats['total_pixels']}",
            f"Pixels placed:    {stats['pixels_placed']}",
            f"Pinwheels placed: {stats['pinwheels_placed']}",
            f"Correct:          {stats['correct_placements']}",
            f"Accuracy:         {stats['accuracy']:.1f}%",
            f"Elapsed:          {stats['elapsed']:.2f}s",
            f"Speed:            {stats['pixels_per_sec']:.0f} px/s",
            "",
            f"Pinwheel radius:  {self.solver.pinwheel_radius}",
            f"Px/pinwheel:      {self.solver.pixels_per_pinwheel}",
            f"Placement mode:   {self.solver.placement_mode}",
            f"Metric:           {self.solver.distance_metric}",
            "=" * 36,
        ]
        self.results_text.setPlainText("\n".join(lines))

    def _update_graphs(self):
        if not self.pixels_placed_history:
            return

        iters = list(range(len(self.pixels_placed_history)))

        # Pixels placed
        self.ax_pixels.clear()
        self.ax_pixels.set_facecolor('#ffe4ec')
        self.ax_pixels.plot(iters, self.pixels_placed_history, color='#ff85a2', linewidth=2)
        self.ax_pixels.fill_between(iters, self.pixels_placed_history, alpha=0.3, color='#ffb6c1')
        self.ax_pixels.set_xlabel('Step', fontsize=8, color='#8b4563')
        self.ax_pixels.set_ylabel('Pixels', fontsize=8, color='#8b4563')
        self.ax_pixels.tick_params(colors='#8b4563', labelsize=7)
        for sp in self.ax_pixels.spines.values():
            sp.set_color('#ffb6c1')
        self.fig_pixels.tight_layout()
        self.canvas_pixels.draw()

        # Accuracy
        self.ax_acc.clear()
        self.ax_acc.set_facecolor('#ffe4ec')
        self.ax_acc.plot(iters, self.accuracy_history, color='#2196f3', linewidth=2)
        self.ax_acc.fill_between(iters, self.accuracy_history, alpha=0.3, color='#64b5f6')
        self.ax_acc.axhline(y=100, color='#8b4563', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_acc.set_xlabel('Step', fontsize=8, color='#8b4563')
        self.ax_acc.set_ylabel('%', fontsize=8, color='#8b4563')
        self.ax_acc.tick_params(colors='#8b4563', labelsize=7)
        for sp in self.ax_acc.spines.values():
            sp.set_color('#ffb6c1')
        self.fig_acc.tight_layout()
        self.canvas_acc.draw()


# ──────────────────────────────────────────────────────────────────────
# Pink stylesheet (matching the existing GUI theme)
# ──────────────────────────────────────────────────────────────────────

PINK_STYLESHEET = """
QMainWindow { background-color: #fff0f5; }
QWidget { background-color: #fff0f5; color: #8b4563; font-family: 'Segoe UI', Arial, sans-serif; }
QGroupBox {
    background-color: #ffe4ec; border: 2px solid #ffb6c1; border-radius: 10px;
    margin-top: 12px; padding-top: 10px; font-weight: bold; color: #d63384;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 2px 10px; background-color: #ffb6c1; border-radius: 5px; color: #fff;
}
QPushButton {
    background-color: #ff85a2; color: white; border: none; border-radius: 8px;
    padding: 6px 12px; font-weight: bold; min-height: 24px;
}
QPushButton:hover { background-color: #ff6b8a; }
QPushButton:pressed { background-color: #e55a7b; }
QPushButton:disabled { background-color: #ddb8c4; color: #f5e6ea; }
QLineEdit {
    background-color: #fff; border: 2px solid #ffb6c1; border-radius: 6px;
    padding: 4px 8px; color: #8b4563;
}
QLineEdit:focus { border: 2px solid #ff85a2; }
QComboBox {
    background-color: #fff; border: 2px solid #ffb6c1; border-radius: 6px;
    padding: 4px 8px; color: #8b4563; min-height: 20px;
}
QComboBox:hover { border: 2px solid #ff85a2; }
QComboBox QAbstractItemView {
    background-color: #fff; border: 2px solid #ffb6c1;
    selection-background-color: #ffb6c1; selection-color: #8b4563;
}
QLabel { background-color: transparent; color: #8b4563; }
QTextEdit {
    background-color: #fff; border: 2px solid #ffb6c1; border-radius: 8px;
    color: #8b4563;
}
QScrollArea { background-color: #fff0f5; border: none; }
QScrollBar:vertical {
    background-color: #ffe4ec; width: 12px; border-radius: 6px;
}
QScrollBar::handle:vertical {
    background-color: #ffb6c1; border-radius: 5px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background-color: #ff85a2; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QProgressBar {
    background-color: #ffe4ec; border: 2px solid #ffb6c1; border-radius: 8px;
    text-align: center; color: #8b4563; font-weight: bold;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #ff85a2, stop:1 #ffb6c1);
    border-radius: 6px;
}
"""


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pinwheel Solver — reconstruct scrambled video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-v", "--video", type=str, default=None, help="Input video file")
    p.add_argument("-f", "--frames", type=int, default=100, help="Number of frames")
    p.add_argument("-s", "--stride", type=int, default=1, help="Frame stride")
    p.add_argument("--crop", type=float, default=10, help="Crop percent")
    p.add_argument("--scramble-seed", type=int, default=42, help="Scramble seed")
    p.add_argument("--radius", type=int, default=5, help="Pinwheel radius (rings)")
    p.add_argument("--mode", choices=["edge", "midpoint"], default="edge", help="Placement mode")
    p.add_argument("--metric", choices=["euclidean", "squared", "manhattan", "cosine"],
                    default="euclidean", help="Distance metric")
    p.add_argument("--shortlist", type=int, default=500, help="Shortlist size")
    p.add_argument("--no-gui", action="store_true", help="Run without GUI")
    p.add_argument("-o", "--output", type=str, default=None, help="Save result image")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_gui and args.video:
        # Headless mode
        wall = TVWall(args.video, num_frames=args.frames, stride=args.stride,
                      crop_percent=args.crop)
        print(f"Grid: {wall.width}x{wall.height}, Frames: {wall.num_frames}")
        wall.scramble(seed=args.scramble_seed)
        print(f"Scrambled (seed={args.scramble_seed})")

        solver = PinwheelSolver(
            wall,
            pinwheel_radius=args.radius,
            placement_mode=args.mode,
            distance_metric=args.metric,
            scramble_seed=args.scramble_seed,
            shortlist_size=args.shortlist,
        )
        solver.solve()
        solver.apply_solution()

        stats = solver.get_stats()
        print(f"\nResults:")
        print(f"  Pixels placed:    {stats['pixels_placed']}")
        print(f"  Pinwheels placed: {stats['pinwheels_placed']}")
        print(f"  Correct:          {stats['correct_placements']} ({stats['accuracy']:.1f}%)")
        print(f"  Time:             {stats['elapsed']:.2f}s")
        print(f"  Speed:            {stats['pixels_per_sec']:.0f} px/s")

        if args.output:
            wall.save_frame(0, args.output)
            print(f"Saved {args.output}")
        return

    # GUI mode
    app = QApplication(sys.argv)
    app.setStyleSheet(PINK_STYLESHEET)

    gui = PinwheelSolverGUI()

    # Pre-load video if given on command line
    if args.video:
        gui.frames_edit.setText(str(args.frames))
        gui.stride_edit.setText(str(args.stride))
        gui.crop_edit.setText(str(args.crop))
        gui.radius_edit.setText(str(args.radius))
        gui.seed_edit.setText(str(args.scramble_seed))
        gui.shortlist_edit.setText(str(args.shortlist))
        gui.mode_combo.setCurrentText(args.mode)
        gui.metric_combo.setCurrentText(args.metric)

        try:
            gui.wall = TVWall(args.video, num_frames=args.frames, stride=args.stride,
                              crop_percent=args.crop)
            gui.video_label.setText(
                f"{os.path.basename(args.video)}\n"
                f"Size: {gui.wall.width}x{gui.wall.height}\n"
                f"Frames: {gui.wall.num_frames}"
            )
            cw = 800
            ch = 600
            sx = cw / gui.wall.width
            sy = ch / gui.wall.height
            gui.display_scale = max(1, min(int(min(sx, sy)), 10))
            gui._update_display()
            gui.status_label.setText("Video loaded from CLI")
        except Exception as e:
            print(f"Failed to load video: {e}")

    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
