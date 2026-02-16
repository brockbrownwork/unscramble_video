#!/usr/bin/env python
"""
BFS Pixel Unscrambling Solver

Reconstructs a scrambled video grid by building the solution from scratch
(like a jigsaw puzzle). Grows from a seed pixel outward in BFS order,
always placing the best-fitting unplaced pixel into the next frontier position.

Usage:
    python bfs_solver.py -v video.mkv -f 100 -s 1 --metric euclidean \
        --scramble-seed 42 --shortlist 50 --neighbor-mode cardinal
"""

import argparse
import heapq
import itertools
import math
import os
import sys
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from tv_wall import TVWall

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False


CARDINAL_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ALL_DELTAS = [(-1, -1), (0, -1), (1, -1),
              (-1,  0),          (1,  0),
              (-1,  1), (0,  1), (1,  1)]
KERNEL_5x5_DELTAS = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)
                     if (dx, dy) != (0, 0)]


class BFSSolver:
    """
    BFS-based pixel unscrambling solver.

    Builds a solution from scratch by placing pixels into an output grid,
    growing outward from a seed position in breadth-first order.
    """

    def __init__(self, wall: TVWall, *,
                 seed_position="random",
                 neighbor_mode="all",
                 distance_metric="euclidean",
                 dtw_window=0.1,
                 initial_candidates=8,
                 permutation_limit=100_000,
                 shortlist_size=0,
                 rng_seed=None,
                 snapshot_interval=10_000,
                 snapshot_dir="bfs_output"):
        self.wall = wall
        self.seed_position = seed_position
        self.neighbor_mode = neighbor_mode
        self.distance_metric = distance_metric
        self.dtw_window = dtw_window
        self.initial_candidates = initial_candidates
        self.permutation_limit = permutation_limit
        self.shortlist_size = shortlist_size
        self.rng_seed = rng_seed
        self.snapshot_interval = snapshot_interval
        self.snapshot_dir = snapshot_dir

        self.H = wall.height
        self.W = wall.width
        self.T = wall.num_frames
        self.total_pixels = self.H * self.W

        if neighbor_mode == "cardinal":
            self._deltas = CARDINAL_DELTAS
        elif neighbor_mode == "5x5":
            self._deltas = KERNEL_5x5_DELTAS
        else:
            self._deltas = ALL_DELTAS

        # Filled during _precompute
        self.all_series = None
        self.series_flat = None
        self.mean_colors = None
        self._lum_order = None      # pixel indices sorted by luminance
        self._lum_sorted = None     # luminance values in sorted order
        self.output_grid = None
        self.placed_at = None
        self.is_unplaced = None
        self.unplaced = None
        self.frontier = None
        self.in_frontier = None
        self.neighbor_count = None  # tracks # of placed neighbors per position
        self.insertion_counter = 0

        # Mahalanobis state (populated in _precompute)
        self._cov_inv_flat = None  # (H*W, 3, 3) inverse covariance per pixel

        # GPU state
        self._use_gpu = False
        self._series_flat_gpu = None
        try:
            import cupy as cp
            if self.distance_metric != "dtw":
                self._use_gpu = True
                self._cp = cp
        except ImportError:
            pass

        self.stats = {
            "placements": 0,
            "correct_placements": 0,
            "total_pixels": self.total_pixels,
            "elapsed_time": 0.0,
        }
        self._stop = False

    # ------------------------------------------------------------------
    # Phase 0: Precomputation
    # ------------------------------------------------------------------

    def _precompute(self):
        if self.all_series is not None:
            return  # already precomputed
        print("Precomputing color series...")
        self.all_series = self.wall.get_all_series(force_cpu=True)  # (H, W, 3, T)
        self.series_flat = self.all_series.reshape(self.total_pixels, 3, self.T)
        self.mean_colors = self.series_flat.mean(axis=2)  # (H*W, 3)

        self.output_grid = np.full((self.H, self.W, 2), -1, dtype=np.int32)
        self.placed_at = np.full((self.H, self.W), -1, dtype=np.int32)
        self.is_unplaced = np.ones(self.total_pixels, dtype=bool)
        self.unplaced = set(range(self.total_pixels))

        self.frontier = []
        self.in_frontier = np.zeros((self.H, self.W), dtype=bool)
        self.neighbor_count = np.zeros((self.H, self.W), dtype=np.int8)
        self.insertion_counter = 0

        # Build luminance-sorted index for fast nearest-neighbor shortlisting.
        # Luminance reduces 3D color to 1D, enabling binary search + local scan.
        luminance = (0.299 * self.mean_colors[:, 0]
                     + 0.587 * self.mean_colors[:, 1]
                     + 0.114 * self.mean_colors[:, 2])
        self._lum_order = np.argsort(luminance)       # pixel indices sorted by lum
        self._lum_sorted = luminance[self._lum_order]  # sorted luminance values

        # Precompute per-pixel inverse covariance for Mahalanobis
        if self.distance_metric == "mahalanobis":
            print("  Computing per-pixel covariance inverses...")
            # series_flat: (H*W, 3, T) — treat each pixel as T observations of 3 channels
            N = self.total_pixels
            C = 3
            # Compute covariance per pixel: center, then cov = X @ X.T / (T-1)
            means = self.series_flat.mean(axis=2, keepdims=True)  # (N, 3, 1)
            centered = self.series_flat - means  # (N, 3, T)
            # Batch covariance: (N, 3, T) @ (N, T, 3) -> (N, 3, 3)
            cov = np.einsum('nct,ndt->ncd', centered, centered) / max(self.T - 1, 1)
            # Regularise
            cov += np.eye(C)[np.newaxis, :, :] * 1e-6
            # Batch invert
            self._cov_inv_flat = np.linalg.inv(cov)  # (N, 3, 3)
            print(f"  Mahalanobis: precomputed {N} covariance inverses")

        # Transfer series to GPU once if available
        if self._use_gpu:
            cp = self._cp
            self._series_flat_gpu = cp.asarray(self.series_flat)
            print(f"  GPU: transferred series_flat ({self.series_flat.nbytes / 1e6:.1f} MB) to device")

    # ------------------------------------------------------------------
    # Phase 1: Seed Placement
    # ------------------------------------------------------------------

    def _place_seed(self):
        if self.seed_position == "random":
            rng = np.random.RandomState(self.rng_seed)
            seed_idx = int(rng.randint(0, self.total_pixels))
        else:
            sx, sy = self.seed_position
            seed_idx = sy * self.W + sx

        center_x = self.W // 2
        center_y = self.H // 2
        self._place_pixel((center_x, center_y), seed_idx)
        self._expand_frontier((center_x, center_y))

    # ------------------------------------------------------------------
    # Phase 2: Initial Neighborhood (Permutation Search)
    # ------------------------------------------------------------------

    def _initial_neighborhood(self):
        center_x = self.W // 2
        center_y = self.H // 2
        seed_idx = self.placed_at[center_y, center_x]
        seed_series = self.series_flat[seed_idx]  # (3, T)

        # Gather current frontier positions
        frontier_positions = [item[2] for item in self.frontier]
        K = len(frontier_positions)
        if K == 0:
            return

        C = max(self.initial_candidates, K + 1)
        candidates = self._find_closest_candidates(seed_series, C,
                                                    ref_idx=seed_idx)

        perm_count = math.perm(len(candidates), K)

        if perm_count <= self.permutation_limit:
            # Exhaustive permutation search
            best_score = float("inf")
            best_assignment = None

            for assignment in itertools.permutations(range(len(candidates)), K):
                score = self._evaluate_permutation(
                    frontier_positions, candidates, assignment, seed_idx
                )
                if score < best_score:
                    best_score = score
                    best_assignment = assignment

            if best_assignment is not None:
                for i, pos in enumerate(frontier_positions):
                    self._place_pixel(pos, candidates[best_assignment[i]])
        else:
            # Greedy fallback
            for pos in frontier_positions:
                placed_neighbors = self._get_placed_neighbors(pos)
                best_idx = None
                best_score = float("inf")
                for cand_idx in candidates:
                    if not self.is_unplaced[cand_idx]:
                        continue
                    score = self._evaluate_candidate(cand_idx, placed_neighbors)
                    if score < best_score:
                        best_score = score
                        best_idx = cand_idx
                if best_idx is not None:
                    self._place_pixel(pos, best_idx)

        # Rebuild frontier from all placed positions so far
        self.frontier = []
        self.in_frontier[:] = False
        self.insertion_counter = 0
        placed_positions = np.argwhere(self.placed_at >= 0)  # (N, 2) as [y, x]
        for yx in placed_positions:
            self._expand_frontier((int(yx[1]), int(yx[0])))

    def _find_closest_candidates(self, reference_series, count, ref_idx=None):
        """Find top `count` unplaced pixels closest to reference_series.

        Uses GPU acceleration (CuPy) when available and the metric supports it.
        Falls back to CPU (NumPy) otherwise. Only called once (Phase 2).

        For Mahalanobis, ref_idx is the flat pixel index of the reference pixel,
        used to look up its precomputed inverse covariance.
        """
        unplaced_indices = np.where(self.is_unplaced)[0]
        if len(unplaced_indices) <= count:
            return unplaced_indices.tolist()

        # DTW cannot be GPU-accelerated, use CPU path
        if self.distance_metric == "dtw":
            from aeon.distances import dtw_distance
            distances = np.array([
                dtw_distance(reference_series, self.series_flat[idx],
                             window=self.dtw_window)
                for idx in unplaced_indices
            ])
            top = np.argpartition(distances, count)[:count]
            top = top[np.argsort(distances[top])]
            return unplaced_indices[top].tolist()

        # Mahalanobis: two-stage — GPU Euclidean coarse filter, then Mahalanobis rerank
        if self.distance_metric == "mahalanobis":
            RERANK_K = 500
            coarse_count = max(count, RERANK_K)

            # Stage 1: Euclidean coarse filter (GPU if available)
            try:
                import cupy as cp
                use_gpu = True
            except ImportError:
                use_gpu = False

            if use_gpu:
                xp = cp
                unplaced_series_dev = xp.asarray(self.series_flat[unplaced_indices])
                ref_dev = xp.asarray(reference_series[np.newaxis, :, :])
            else:
                xp = np
                unplaced_series_dev = self.series_flat[unplaced_indices]
                ref_dev = reference_series[np.newaxis, :, :]

            diff = unplaced_series_dev - ref_dev
            euc_distances = xp.sqrt(xp.sum(diff ** 2, axis=(1, 2)))

            k = min(coarse_count, len(unplaced_indices))
            top_coarse = xp.argpartition(euc_distances, k)[:k]
            top_coarse = top_coarse[xp.argsort(euc_distances[top_coarse])]

            if use_gpu:
                top_coarse = cp.asnumpy(top_coarse)
                del unplaced_series_dev, ref_dev, diff, euc_distances
                cp.get_default_memory_pool().free_all_blocks()

            coarse_pixel_indices = unplaced_indices[top_coarse]

            # Stage 2: Mahalanobis rerank on the narrowed set
            if ref_idx is not None:
                cov_inv = self._cov_inv_flat[ref_idx]  # (3, 3)
            else:
                colors = reference_series.T  # (T, 3)
                cov = np.cov(colors, rowvar=False) + np.eye(3) * 1e-6
                cov_inv = np.linalg.inv(cov)

            coarse_series = self.series_flat[coarse_pixel_indices]  # (K, 3, T)
            ref = reference_series[np.newaxis, :, :]  # (1, 3, T)
            diff_t = (coarse_series - ref).transpose(0, 2, 1)  # (K, T, 3)
            transformed = diff_t @ cov_inv  # (K, T, 3)
            mahal_sq = np.sum(transformed * diff_t, axis=2)  # (K, T)
            distances = np.sum(np.sqrt(np.maximum(mahal_sq, 0.0)), axis=1)  # (K,)

            top = np.argpartition(distances, count)[:count]
            top = top[np.argsort(distances[top])]
            return coarse_pixel_indices[top].tolist()

        # GPU path for vectorizable metrics
        try:
            import cupy as cp
            use_gpu = True
        except ImportError:
            use_gpu = False

        if use_gpu:
            xp = cp
            unplaced_series = xp.asarray(self.series_flat[unplaced_indices])  # (N, 3, T)
            ref = xp.asarray(reference_series[np.newaxis, :, :])  # (1, 3, T)
        else:
            xp = np
            unplaced_series = self.series_flat[unplaced_indices]  # (N, 3, T)
            ref = reference_series[np.newaxis, :, :]  # (1, 3, T)

        if self.distance_metric == "cosine":
            ref_flat = ref.reshape(-1)
            unplaced_flat = unplaced_series.reshape(len(unplaced_indices), -1)
            dots = unplaced_flat @ ref_flat
            ref_norm = xp.linalg.norm(ref_flat)
            unp_norms = xp.linalg.norm(unplaced_flat, axis=1)
            denom = ref_norm * unp_norms
            denom = xp.where(denom == 0, 1.0, denom)
            distances = 1.0 - dots / denom
        else:
            diff = unplaced_series - ref
            if self.distance_metric == "euclidean":
                distances = xp.sqrt(xp.sum(diff ** 2, axis=(1, 2)))
            elif self.distance_metric == "squared":
                distances = xp.sum(diff ** 2, axis=(1, 2))
            elif self.distance_metric == "manhattan":
                distances = xp.sum(xp.abs(diff), axis=(1, 2))
            else:
                distances = xp.sqrt(xp.sum(diff ** 2, axis=(1, 2)))

        if use_gpu:
            # argpartition and argsort on GPU, then transfer indices to CPU
            top = xp.argpartition(distances, count)[:count]
            top = top[xp.argsort(distances[top])]
            top_cpu = cp.asnumpy(top)
            # Free GPU memory
            del unplaced_series, ref, distances
            cp.get_default_memory_pool().free_all_blocks()
        else:
            top_cpu = np.argpartition(distances, count)[:count]
            top_cpu = top_cpu[np.argsort(distances[top_cpu])]

        return unplaced_indices[top_cpu].tolist()

    def _evaluate_permutation(self, frontier_positions, candidates, assignment,
                              seed_idx):
        """Evaluate total adjacency dissonance for a candidate assignment."""
        # Build temporary mapping: output_pos -> pixel_idx
        temp = {}
        for y in range(self.H):
            for x in range(self.W):
                if self.placed_at[y, x] >= 0:
                    temp[(x, y)] = self.placed_at[y, x]
        for i, pos in enumerate(frontier_positions):
            temp[pos] = candidates[assignment[i]]

        total = 0.0
        counted = set()
        for pos, pix in temp.items():
            for nx, ny in self._get_output_neighbors(pos[0], pos[1]):
                npos = (nx, ny)
                if npos in temp:
                    edge = (min(pos, npos), max(pos, npos))
                    if edge not in counted:
                        counted.add(edge)
                        npix = temp[npos]
                        total += self._compute_pairwise_distance(
                            self.series_flat[pix],
                            self.series_flat[npix],
                            idx_a=pix, idx_b=npix,
                        )
        return total

    # ------------------------------------------------------------------
    # Phase 3: BFS Expansion
    # ------------------------------------------------------------------

    def _bfs_expand(self, callback=None):
        total_to_place = self.total_pixels - self.stats["placements"]

        with tqdm(total=total_to_place, desc="BFS expansion") as pbar:
            while self.frontier and self.unplaced and not self._stop:
                neg_count, _order, pos = heapq.heappop(self.frontier)
                ox, oy = pos
                self.in_frontier[oy, ox] = False

                if self.placed_at[oy, ox] >= 0:
                    continue

                placed_neighbors = self._get_placed_neighbors(pos)
                if not placed_neighbors:
                    # No context — place closest to global mean
                    if self.unplaced:
                        idx = next(iter(self.unplaced))
                        self._place_pixel(pos, idx)
                        self._expand_frontier(pos)
                        pbar.update(1)
                    continue

                shortlist = self._get_shortlist(placed_neighbors)

                best_idx, best_score = self._evaluate_shortlist_batch(
                    shortlist, placed_neighbors)

                if best_idx is None:
                    continue

                self._place_pixel(pos, best_idx)
                self._expand_frontier(pos)
                pbar.update(1)

                if self.snapshot_interval and self.stats["placements"] % self.snapshot_interval == 0:
                    self._save_snapshot(self.stats["placements"])

                if callback is not None:
                    step_info = {
                        "placements": self.stats["placements"],
                        "total_pixels": self.total_pixels,
                        "percent": 100.0 * self.stats["placements"] / self.total_pixels,
                        "position": pos,
                        "pixel_idx": best_idx,
                        "score": best_score,
                        "frontier_size": len(self.frontier),
                        "unplaced_count": len(self.unplaced),
                        "correct_placements": self.stats["correct_placements"],
                    }
                    callback(step_info)

    # ------------------------------------------------------------------
    # Shortlisting & Evaluation
    # ------------------------------------------------------------------

    def _get_shortlist(self, placed_neighbor_positions):
        """Top S unplaced candidates by mean-RGB proximity to placed neighbors.

        Uses a luminance-sorted index with binary search for O(log N) lookup
        followed by a local scan, instead of brute-force O(N) over all
        unplaced pixels.
        """
        S = self.shortlist_size

        unplaced_indices = np.where(self.is_unplaced)[0]
        if not S or len(unplaced_indices) <= S:
            return unplaced_indices.tolist()

        # Target: mean of placed neighbors' mean colors
        neighbor_pixel_indices = np.array([
            self.placed_at[ny, nx]
            for nx, ny in placed_neighbor_positions
        ])
        target = self.mean_colors[neighbor_pixel_indices].mean(axis=0)  # (3,)
        target_lum = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]

        # Binary search to find insertion point in sorted luminance array
        center = np.searchsorted(self._lum_sorted, target_lum)
        N = self.total_pixels

        # Expand outward from center, collecting unplaced pixels until we
        # have S candidates. Two pointers moving left and right.
        result = []
        lo = center - 1
        hi = center
        while len(result) < S and (lo >= 0 or hi < N):
            # Pick the closer side
            if hi >= N:
                idx = self._lum_order[lo]
                lo -= 1
            elif lo < 0:
                idx = self._lum_order[hi]
                hi += 1
            elif (target_lum - self._lum_sorted[lo]) <= (self._lum_sorted[hi] - target_lum):
                idx = self._lum_order[lo]
                lo -= 1
            else:
                idx = self._lum_order[hi]
                hi += 1

            if self.is_unplaced[idx]:
                result.append(idx)

        return result

    def _evaluate_candidate(self, candidate_idx, placed_neighbor_positions):
        """Mean distance from candidate to all placed neighbors."""
        cand_series = self.series_flat[candidate_idx]
        total = 0.0
        for nx, ny in placed_neighbor_positions:
            n_idx = self.placed_at[ny, nx]
            total += self._compute_pairwise_distance(
                cand_series, self.series_flat[n_idx],
                idx_a=candidate_idx, idx_b=n_idx,
            )
        return total / len(placed_neighbor_positions)

    def _evaluate_shortlist_batch(self, shortlist, placed_neighbor_positions):
        """Vectorized evaluation of all shortlist candidates against placed neighbors.

        Returns (best_pixel_idx, best_score).
        """
        S = len(shortlist)
        N = len(placed_neighbor_positions)

        if S == 0:
            return None, float("inf")

        cand_indices = np.array(shortlist, dtype=np.intp)
        neighbor_pixel_indices = np.array([
            self.placed_at[ny, nx]
            for nx, ny in placed_neighbor_positions
        ], dtype=np.intp)

        if self.distance_metric == "dtw":
            # DTW can't be vectorized; fall back to per-candidate loop
            best_idx = None
            best_score = float("inf")
            for cand_idx in shortlist:
                score = self._evaluate_candidate(cand_idx, placed_neighbor_positions)
                if score < best_score:
                    best_score = score
                    best_idx = cand_idx
            return best_idx, best_score

        if self.distance_metric == "mahalanobis":
            # Two-stage: GPU Euclidean to top 500, then CPU Mahalanobis rerank.
            RERANK_K = 500
            if S > RERANK_K:
                # Stage 1: GPU-accelerated Euclidean to narrow candidates
                if self._use_gpu and self._series_flat_gpu is not None:
                    top_k_indices, _ = self._euclidean_top_k_gpu(
                        cand_indices, neighbor_pixel_indices, RERANK_K)
                else:
                    top_k_indices, _ = self._euclidean_top_k_cpu(
                        cand_indices, neighbor_pixel_indices, RERANK_K)
                # Recurse with narrowed set
                return self._evaluate_shortlist_batch(
                    top_k_indices.tolist(), placed_neighbor_positions)

            # Stage 2: Full Mahalanobis on the narrowed set (S <= 500)
            cand_series = self.series_flat[cand_indices]       # (S, 3, T)
            neigh_series = self.series_flat[neighbor_pixel_indices]  # (N, 3, T)

            diff = cand_series[:, np.newaxis, :, :] - neigh_series[np.newaxis, :, :, :]
            diff_t = diff.transpose(0, 1, 3, 2)  # (S, N, T, 3)

            neigh_cov_inv = self._cov_inv_flat[neighbor_pixel_indices]  # (N, 3, 3)

            transformed = np.einsum('sntc,ncd->sntd', diff_t,
                                    neigh_cov_inv)  # (S, N, T, 3)

            mahal_sq = np.sum(transformed * diff_t, axis=3)  # (S, N, T)
            distances = np.sum(np.sqrt(np.maximum(mahal_sq, 0.0)),
                               axis=2)  # (S, N)

            mean_distances = distances.mean(axis=1)  # (S,)
            best_local = int(np.argmin(mean_distances))
            return shortlist[best_local], float(mean_distances[best_local])

        # GPU per-call overhead (~1ms) outweighs compute for typical S*N sizes.
        # Only use GPU when the workload is very large (e.g. S=1000+ candidates).
        if (self._use_gpu and self._series_flat_gpu is not None
                and S >= 1000):
            return self._evaluate_shortlist_batch_gpu(
                cand_indices, neighbor_pixel_indices)

        # CPU vectorized path
        cand_series = self.series_flat[cand_indices]       # (S, 3, T)
        neigh_series = self.series_flat[neighbor_pixel_indices]  # (N, 3, T)

        if self.distance_metric == "cosine":
            cand_flat = cand_series.reshape(S, -1)        # (S, 3*T)
            neigh_flat = neigh_series.reshape(N, -1)      # (N, 3*T)
            dots = cand_flat @ neigh_flat.T               # (S, N)
            cand_norms = np.linalg.norm(cand_flat, axis=1, keepdims=True)
            neigh_norms = np.linalg.norm(neigh_flat, axis=1, keepdims=True).T
            denom = cand_norms * neigh_norms
            denom = np.where(denom == 0, 1.0, denom)
            distances = 1.0 - dots / denom
        else:
            # (S,1,3,T) - (1,N,3,T) -> (S,N,3,T)
            diff = cand_series[:, np.newaxis, :, :] - neigh_series[np.newaxis, :, :, :]
            if self.distance_metric == "euclidean":
                distances = np.sqrt(np.sum(diff ** 2, axis=(2, 3)))
            elif self.distance_metric == "squared":
                distances = np.sum(diff ** 2, axis=(2, 3))
            elif self.distance_metric == "manhattan":
                distances = np.sum(np.abs(diff), axis=(2, 3))
            else:
                distances = np.sqrt(np.sum(diff ** 2, axis=(2, 3)))

        mean_distances = distances.mean(axis=1)  # (S,)
        best_local = int(np.argmin(mean_distances))
        return shortlist[best_local], float(mean_distances[best_local])

    def _evaluate_shortlist_batch_gpu(self, cand_indices, neighbor_pixel_indices):
        """GPU-accelerated batch evaluation using CuPy."""
        cp = self._cp
        S = len(cand_indices)
        N = len(neighbor_pixel_indices)

        cand_series = self._series_flat_gpu[cand_indices]       # (S, 3, T)
        neigh_series = self._series_flat_gpu[neighbor_pixel_indices]  # (N, 3, T)

        if self.distance_metric == "cosine":
            cand_flat = cand_series.reshape(S, -1)
            neigh_flat = neigh_series.reshape(N, -1)
            dots = cand_flat @ neigh_flat.T
            cand_norms = cp.linalg.norm(cand_flat, axis=1, keepdims=True)
            neigh_norms = cp.linalg.norm(neigh_flat, axis=1, keepdims=True).T
            denom = cand_norms * neigh_norms
            denom = cp.where(denom == 0, 1.0, denom)
            distances = 1.0 - dots / denom
        else:
            diff = cand_series[:, cp.newaxis, :, :] - neigh_series[cp.newaxis, :, :, :]
            if self.distance_metric == "euclidean":
                distances = cp.sqrt(cp.sum(diff ** 2, axis=(2, 3)))
            elif self.distance_metric == "squared":
                distances = cp.sum(diff ** 2, axis=(2, 3))
            elif self.distance_metric == "manhattan":
                distances = cp.sum(cp.abs(diff), axis=(2, 3))
            else:
                distances = cp.sqrt(cp.sum(diff ** 2, axis=(2, 3)))

        mean_distances = distances.mean(axis=1)
        best_local = int(cp.argmin(mean_distances))
        best_score = float(mean_distances[best_local].get())
        return int(cand_indices[best_local]), best_score

    def _euclidean_top_k_gpu(self, cand_indices, neighbor_pixel_indices, k):
        """GPU Euclidean to select top-k candidates from a larger shortlist.

        Returns (top_k_pixel_indices, top_k_scores) as numpy arrays.
        """
        cp = self._cp
        S = len(cand_indices)
        k = min(k, S)

        cand_series = self._series_flat_gpu[cand_indices]           # (S, 3, T)
        neigh_series = self._series_flat_gpu[neighbor_pixel_indices] # (N, 3, T)

        diff = cand_series[:, cp.newaxis, :, :] - neigh_series[cp.newaxis, :, :, :]
        distances = cp.sqrt(cp.sum(diff ** 2, axis=(2, 3)))  # (S, N)
        mean_distances = distances.mean(axis=1)               # (S,)

        top_k = cp.argpartition(mean_distances, k)[:k]
        top_k = top_k[cp.argsort(mean_distances[top_k])]
        top_k_cpu = cp.asnumpy(top_k)
        scores = cp.asnumpy(mean_distances[top_k])

        return cand_indices[top_k_cpu], scores

    def _euclidean_top_k_cpu(self, cand_indices, neighbor_pixel_indices, k):
        """CPU Euclidean to select top-k candidates from a larger shortlist.

        Returns (top_k_pixel_indices, top_k_scores) as numpy arrays.
        """
        S = len(cand_indices)
        k = min(k, S)

        cand_series = self.series_flat[cand_indices]           # (S, 3, T)
        neigh_series = self.series_flat[neighbor_pixel_indices] # (N, 3, T)

        diff = cand_series[:, np.newaxis, :, :] - neigh_series[np.newaxis, :, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=(2, 3)))  # (S, N)
        mean_distances = distances.mean(axis=1)               # (S,)

        top_k = np.argpartition(mean_distances, k)[:k]
        top_k = top_k[np.argsort(mean_distances[top_k])]

        return cand_indices[top_k], mean_distances[top_k]

    def _compute_pairwise_distance(self, series_a, series_b,
                                    idx_a=None, idx_b=None):
        """Distance between two series of shape (3, T).

        For Mahalanobis, idx_a and idx_b are flat pixel indices used to look up
        precomputed inverse covariance matrices.  Uses the average of both
        covariances for a symmetric distance.
        """
        if self.distance_metric == "euclidean":
            diff = series_a - series_b
            return float(np.sqrt(np.sum(diff ** 2)))
        elif self.distance_metric == "squared":
            diff = series_a - series_b
            return float(np.sum(diff ** 2))
        elif self.distance_metric == "manhattan":
            return float(np.sum(np.abs(series_a - series_b)))
        elif self.distance_metric == "cosine":
            a = series_a.flatten()
            b = series_b.flatten()
            dot = np.dot(a, b)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            denom = na * nb
            if denom == 0:
                return 1.0
            return float(1.0 - dot / denom)
        elif self.distance_metric == "dtw":
            from aeon.distances import dtw_distance
            return float(dtw_distance(series_a, series_b, window=self.dtw_window))
        elif self.distance_metric == "mahalanobis":
            diff = (series_a - series_b).T  # (T, 3)
            # Average covariance of both pixels for symmetry
            if idx_a is not None and idx_b is not None:
                cov_inv = (self._cov_inv_flat[idx_a] + self._cov_inv_flat[idx_b]) * 0.5
            elif idx_a is not None:
                cov_inv = self._cov_inv_flat[idx_a]
            elif idx_b is not None:
                cov_inv = self._cov_inv_flat[idx_b]
            else:
                # Fallback: compute from series_a
                colors = series_a.T  # (T, 3)
                cov = np.cov(colors, rowvar=False) + np.eye(3) * 1e-6
                cov_inv = np.linalg.inv(cov)
            # Per-frame Mahalanobis: sqrt(Δc^T @ Σ^-1 @ Δc), summed over frames
            transformed = diff @ cov_inv  # (T, 3)
            mahal_sq = np.sum(transformed * diff, axis=1)  # (T,)
            return float(np.sum(np.sqrt(np.maximum(mahal_sq, 0.0))))
        else:
            raise ValueError(f"Unknown metric: {self.distance_metric}")

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    def _place_pixel(self, output_pos, pixel_idx):
        ox, oy = output_pos
        src_x = pixel_idx % self.W
        src_y = pixel_idx // self.W

        self.output_grid[oy, ox, 0] = src_x
        self.output_grid[oy, ox, 1] = src_y
        self.placed_at[oy, ox] = pixel_idx
        self.is_unplaced[pixel_idx] = False
        self.unplaced.discard(pixel_idx)

        self.stats["placements"] += 1

        # Update neighbor counts for all neighbors of this position
        for dx, dy in self._deltas:
            nx, ny = ox + dx, oy + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                self.neighbor_count[ny, nx] += 1

        # Correctness check: original position of this TV
        orig_x = int(self.wall._perm_x[src_y, src_x])
        orig_y = int(self.wall._perm_y[src_y, src_x])
        if orig_x == ox and orig_y == oy:
            self.stats["correct_placements"] += 1

    def _expand_frontier(self, output_pos):
        ox, oy = output_pos
        for dx, dy in self._deltas:
            nx, ny = ox + dx, oy + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                if self.placed_at[ny, nx] < 0 and not self.in_frontier[ny, nx]:
                    placed_count = int(self.neighbor_count[ny, nx])
                    heapq.heappush(
                        self.frontier,
                        (-placed_count, self.insertion_counter, (nx, ny)),
                    )
                    self.insertion_counter += 1
                    self.in_frontier[ny, nx] = True

    def _get_output_neighbors(self, x, y):
        result = []
        for dx, dy in self._deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                result.append((nx, ny))
        return result

    def _get_placed_neighbors(self, output_pos):
        ox, oy = output_pos
        result = []
        for dx, dy in self._deltas:
            nx, ny = ox + dx, oy + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                if self.placed_at[ny, nx] >= 0:
                    result.append((nx, ny))
        return result

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _save_snapshot(self, placements):
        """Save current output grid state as a PNG."""
        os.makedirs(self.snapshot_dir, exist_ok=True)
        # Build image from placed pixels using frame 0 colors
        img = np.full((self.H, self.W, 3), 40, dtype=np.uint8)  # dark gray for unplaced
        placed_mask = self.placed_at >= 0
        if placed_mask.any() and self.all_series is not None:
            ys, xs = np.where(placed_mask)
            pix_indices = self.placed_at[ys, xs]
            src_xs = pix_indices % self.W
            src_ys = pix_indices // self.W
            img[ys, xs] = self.all_series[src_ys, src_xs, :, 0]
        path = os.path.join(self.snapshot_dir, f"snapshot_{placements:07d}.png")
        Image.fromarray(img).save(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, callback=None):
        """
        Run the full BFS solver.

        Args:
            callback: Optional callable(step_info: dict) invoked after each placement.

        Returns:
            output_grid: np.ndarray shape (H, W, 2), output_grid[y, x] = (src_x, src_y)
        """
        self._stop = False
        start = time.time()

        print("Phase 0: Precomputation...")
        self._precompute()

        print("Phase 1: Seed placement...")
        self._place_seed()
        if callback:
            callback({
                "placements": self.stats["placements"],
                "total_pixels": self.total_pixels,
                "percent": 100.0 * self.stats["placements"] / self.total_pixels,
                "position": (self.W // 2, self.H // 2),
                "pixel_idx": self.placed_at[self.H // 2, self.W // 2],
                "score": 0.0,
                "frontier_size": len(self.frontier),
                "unplaced_count": len(self.unplaced),
                "correct_placements": self.stats["correct_placements"],
            })

        print("Phase 2: Initial neighborhood...")
        self._initial_neighborhood()

        print("Phase 3: BFS expansion...")
        self._bfs_expand(callback=callback)

        elapsed = time.time() - start
        self.stats["elapsed_time"] = elapsed

        # Save final snapshot
        if self.snapshot_interval:
            self._save_snapshot(self.stats["placements"])

        return self.output_grid

    def apply_solution(self):
        """Apply the computed output_grid as the TVWall's permutation."""
        old_perm_x = self.wall._perm_x.copy()
        old_perm_y = self.wall._perm_y.copy()

        new_perm_x = np.zeros_like(old_perm_x)
        new_perm_y = np.zeros_like(old_perm_y)

        scrambled_xs = self.output_grid[:, :, 0]
        scrambled_ys = self.output_grid[:, :, 1]
        valid = (scrambled_xs >= 0) & (scrambled_ys >= 0)

        new_perm_x[valid] = old_perm_x[scrambled_ys[valid], scrambled_xs[valid]]
        new_perm_y[valid] = old_perm_y[scrambled_ys[valid], scrambled_xs[valid]]

        # Identity for any unplaced positions
        identity_y, identity_x = np.mgrid[0:self.H, 0:self.W]
        new_perm_x[~valid] = identity_x[~valid]
        new_perm_y[~valid] = identity_y[~valid]

        self.wall._perm_x = new_perm_x
        self.wall._perm_y = new_perm_y

        if self.wall._gpu is not None:
            self.wall._gpu.cache_permutation(new_perm_x, new_perm_y)
            self.wall._gpu.invalidate_series_cache()

    def get_stats(self):
        """Return solve statistics."""
        placements = self.stats["placements"]
        elapsed = self.stats["elapsed_time"]
        return {
            "total_pixels": self.total_pixels,
            "placements": placements,
            "correct_placements": self.stats["correct_placements"],
            "accuracy": 100.0 * self.stats["correct_placements"] / max(1, placements),
            "elapsed_time": elapsed,
            "placements_per_sec": placements / max(0.001, elapsed),
        }

    def stop(self):
        """Signal the solver to stop."""
        self._stop = True


# ----------------------------------------------------------------------
# Pygame GUI
# ----------------------------------------------------------------------

def run_pygame_gui(solver, wall, args):
    """Run BFS solver with pygame visualization."""
    if not PYGAME_AVAILABLE:
        print("pygame not installed, running headless.")
        print("Install with: pip install pygame")
        solver.solve()
        return

    scale = args.gui_scale
    update_interval = args.gui_update_interval

    H, W = wall.height, wall.width
    panel_w = W * scale
    panel_h = H * scale
    info_w = 300
    window_w = panel_w + info_w
    window_h = max(panel_h, 360)

    pygame.init()
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("BFS Solver — Unscramble Video")
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    # Pre-render frame 0 colors from the scrambled grid
    frame0 = solver.all_series[:, :, :, 0].astype(np.uint8) if solver.all_series is not None else None

    output_surface = pygame.Surface((W, H))
    output_surface.fill((40, 40, 40))

    running = True
    start_time = time.time()

    def gui_callback(step_info):
        nonlocal running, frame0

        # Lazy init frame0 if precompute hadn't run when we set it up
        if frame0 is None and solver.all_series is not None:
            frame0 = solver.all_series[:, :, :, 0].astype(np.uint8)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                solver.stop()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                solver.stop()
                return

        if not running:
            return

        # Paint the placed pixel
        ox, oy = step_info["position"]
        pix_idx = step_info["pixel_idx"]
        src_x = pix_idx % W
        src_y = pix_idx // W
        if frame0 is not None:
            color = tuple(int(c) for c in frame0[src_y, src_x])
            output_surface.set_at((ox, oy), color)

        placements = step_info["placements"]
        if placements % update_interval == 0 or placements == solver.total_pixels:
            scaled = pygame.transform.scale(output_surface, (panel_w, panel_h))
            screen.fill((30, 30, 30))
            screen.blit(scaled, (0, 0))

            elapsed = time.time() - start_time
            pps = placements / max(0.001, elapsed)
            correct = step_info.get("correct_placements", 0)
            acc = 100.0 * correct / max(1, placements)

            lines = [
                "BFS Solver",
                "",
                f"Placed: {placements} / {solver.total_pixels}",
                f"Progress: {step_info['percent']:.1f}%",
                "",
                f"Correct: {correct}",
                f"Accuracy: {acc:.1f}%",
                "",
                f"Frontier: {step_info['frontier_size']}",
                f"Unplaced: {step_info['unplaced_count']}",
                "",
                f"Score: {step_info['score']:.2f}",
                "",
                f"Time: {elapsed:.1f}s",
                f"Speed: {pps:.0f} px/s",
                "",
                f"Metric: {solver.distance_metric}",
                f"Shortlist: {'off' if not solver.shortlist_size else solver.shortlist_size}",
                f"Mode: {solver.neighbor_mode} ({len(solver._deltas)} neighbors)",
            ]

            y_off = 10
            for line in lines:
                if line:
                    surf = font.render(line, True, (200, 200, 200))
                    screen.blit(surf, (panel_w + 10, y_off))
                y_off += 18

            pygame.display.flip()
            clock.tick(60)

    solver.solve(callback=gui_callback)

    # Final display
    if running:
        scaled = pygame.transform.scale(output_surface, (panel_w, panel_h))
        screen.fill((30, 30, 30))
        screen.blit(scaled, (0, 0))

        done_surf = font.render("DONE — Press any key to close", True, (0, 255, 0))
        screen.blit(done_surf, (panel_w + 10, 10))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False

    pygame.quit()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="BFS pixel unscrambling solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-v", "--video", type=str, default="cab_ride_trimmed.mkv",
                        help="Input video file")
    parser.add_argument("-f", "--frames", type=int, default=100,
                        help="Number of frames to load")
    parser.add_argument("-s", "--stride", type=int, default=100,
                        help="Frame skip interval")
    parser.add_argument("--crop", type=float, default=100,
                        help="Crop percentage (1-100)")
    parser.add_argument("--min-entropy", type=float, default=0.0,
                        help="Minimum frame entropy filter")

    parser.add_argument("--scramble-seed", type=int, default=42,
                        help="Random seed for scrambling")

    parser.add_argument("--seed-position", type=str, default="random",
                        help='Seed pixel position: "random" or "x,y"')
    parser.add_argument("--neighbor-mode", type=str, default="all",
                        choices=["cardinal", "all", "5x5"],
                        help="Neighbor connectivity: cardinal (4), all (8), or 5x5 (24)")
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "squared", "manhattan", "cosine", "dtw", "mahalanobis"],
                        help="Distance metric")
    parser.add_argument("--dtw-window", type=float, default=0.1,
                        help="DTW Sakoe-Chiba window (0.0-1.0)")
    parser.add_argument("--initial-candidates", type=int, default=8,
                        help="Candidates for initial permutation search (C)")
    parser.add_argument("--permutation-limit", type=int, default=100_000,
                        help="Max permutations before greedy fallback")
    parser.add_argument("--shortlist", type=int, default=0,
                        help="Shortlist size for BFS expansion (S). 0 = disabled (evaluate all unplaced)")
    parser.add_argument("--rng-seed", type=int, default=None,
                        help="RNG seed for solver")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save result video to this path")
    parser.add_argument("--save-frames", action="store_true", default=True,
                        help="Save before/after frame images (default: on)")
    parser.add_argument("--no-save-frames", action="store_false", dest="save_frames",
                        help="Disable saving before/after frame images")
    parser.add_argument("--save-result-video", type=str, nargs="?",
                        const="bfs_result.mp4", default=None,
                        help="Save solved frames as video (default: bfs_result.mp4)")
    parser.add_argument("--result-fps", type=int, default=15,
                        help="Frames per second for result video")

    parser.add_argument("--snapshot-interval", type=int, default=10_000,
                        help="Save a progress PNG every N placements (0 to disable)")
    parser.add_argument("--snapshot-dir", type=str, default="bfs_output",
                        help="Directory to save progress snapshots")

    parser.add_argument("--no-gui", action="store_true",
                        help="Run without pygame visualization")
    parser.add_argument("--gui-scale", type=int, default=2,
                        help="Display scale factor for pygame window")
    parser.add_argument("--gui-update-interval", type=int, default=10,
                        help="Update pygame display every N placements")

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse seed position
    if args.seed_position != "random":
        try:
            parts = args.seed_position.split(",")
            seed_pos = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            print(f"Invalid seed position: {args.seed_position}. Use 'random' or 'x,y'.")
            sys.exit(1)
    else:
        seed_pos = "random"

    # Load video
    print(f"Loading video: {args.video}")
    wall = TVWall(args.video, num_frames=args.frames, stride=args.stride,
                  crop_percent=args.crop, min_entropy=args.min_entropy)
    print(f"Grid size: {wall.width}x{wall.height} = {wall.num_tvs} pixels")
    print(f"Frames: {wall.num_frames}")

    if args.save_frames:
        wall.save_frame(0, "bfs_original.png")
        print("Saved bfs_original.png")

    # Scramble
    print(f"Scrambling with seed={args.scramble_seed}...")
    wall.scramble(seed=args.scramble_seed)
    print(f"Swapped pixels: {wall.num_swapped}")

    if args.save_frames:
        wall.save_frame(0, "bfs_scrambled.png")
        print("Saved bfs_scrambled.png")

    # Create solver
    solver = BFSSolver(
        wall,
        seed_position=seed_pos,
        neighbor_mode=args.neighbor_mode,
        distance_metric=args.metric,
        dtw_window=args.dtw_window,
        initial_candidates=args.initial_candidates,
        permutation_limit=args.permutation_limit,
        shortlist_size=args.shortlist,
        rng_seed=args.rng_seed,
        snapshot_interval=args.snapshot_interval or 0,
        snapshot_dir=args.snapshot_dir,
    )

    # Solve
    if args.no_gui:
        print("Solving (no GUI)...")
        solver.solve()
    else:
        # Precompute before GUI so frame0 colors are available
        solver._precompute()
        run_pygame_gui(solver, wall, args)

    # Apply solution
    solver.apply_solution()

    # Stats
    stats = solver.get_stats()
    print()
    print("=" * 50)
    print("BFS SOLVER RESULTS")
    print("=" * 50)
    print(f"Total pixels:      {stats['total_pixels']}")
    print(f"Correct placements: {stats['correct_placements']} ({stats['accuracy']:.1f}%)")
    print(f"Solve time:         {stats['elapsed_time']:.2f}s")
    print(f"Speed:              {stats['placements_per_sec']:.0f} px/s")
    print("=" * 50)

    remaining = wall.num_swapped
    print(f"Final misplaced pixels (TVWall): {remaining}")

    if args.save_frames:
        wall.save_frame(0, "bfs_result.png")
        print("Saved bfs_result.png")

    if args.save_result_video:
        import subprocess as _sp
        import cv2

        frames_dir = os.path.join(args.snapshot_dir, "result_frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Total dense frames = num_frames * stride (e.g. 100 * 100 = 10,000)
        total_dense = args.frames * args.stride
        batch_size = 100

        # Compute crop params to match what TVWall applied
        cap = cv2.VideoCapture(args.video)
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        crop_pct = max(1, min(100, args.crop))
        if crop_pct < 100:
            scale = crop_pct / 100.0
            crop_w = int(raw_w * scale)
            crop_h = int(raw_h * scale)
            crop_x = (raw_w - crop_w) // 2
            crop_y = (raw_h - crop_h) // 2
        else:
            crop_w, crop_h = raw_w, raw_h
            crop_x, crop_y = 0, 0

        # Get the solved permutation from the wall
        perm_y = wall._perm_y
        perm_x = wall._perm_x

        print(f"Rendering {total_dense} dense frames (stride=1) to {frames_dir}/ ...")
        print(f"  Source video: {raw_w}x{raw_h}, crop: {crop_w}x{crop_h}")
        print(f"  Loading in batches of {batch_size}")

        frame_counter = 0
        start_frame_idx = wall.start_frame  # same starting point as the solver used

        for batch_start in tqdm(range(0, total_dense, batch_size),
                                desc="Batches", unit="batch"):
            batch_end = min(batch_start + batch_size, total_dense)
            batch_count = batch_end - batch_start

            # Open video and seek to batch start
            cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx + batch_start)

            for i in range(batch_count):
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply crop
                if crop_pct < 100:
                    frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                # Apply solved permutation
                output = frame[perm_y, perm_x]
                img = Image.fromarray(output, mode='RGB')
                img.save(os.path.join(frames_dir, f"frame_{frame_counter:06d}.png"))
                frame_counter += 1

            cap.release()

        # Encode with ffmpeg
        input_pattern = os.path.join(frames_dir, "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.result_fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            args.save_result_video,
        ]
        print(f"Encoding {frame_counter} frames with ffmpeg -> {args.save_result_video}")
        result = _sp.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Saved {args.save_result_video} ({frame_counter} frames, {args.result_fps} fps)")
        else:
            print(f"ffmpeg failed (exit {result.returncode}):")
            print(result.stderr[:500])

    if args.output:
        print(f"Saving result video to {args.output}...")
        wall.save_video(args.output)
        print("Done.")


if __name__ == "__main__":
    main()
