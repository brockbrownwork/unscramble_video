#!/usr/bin/env python
"""
Pinwheel Experiment — Compare reconstruction methods against the ground truth
circular patch from the original video.

Methods:
  - pinwheel: Concentric ring assignment + TSP ordering + rotation optimisation
  - greedy:   BFS-style expansion from center, placing one pixel at a time

Measures total neighbor dissonance using the "summed color distance" metric:
  distance(A, B) = Σ_t √(ΔR² + ΔG² + ΔB²)

Usage:
    python pinwheel_experiment.py -v video.mkv --method pinwheel --radius 5
    python pinwheel_experiment.py -v video.mkv --method greedy --radius 5 --shortlist 50
"""

import argparse
import heapq
import sys

import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QSlider, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from tv_wall import TVWall
from pinwheel_solver_gui import build_lattice_rings, Pinwheel, solve_ring_tsp


# ──────────────────────────────────────────────────────────────────────
# Distance metric
# ──────────────────────────────────────────────────────────────────────

def summed_color_distance(series_a, series_b):
    """Summed per-frame Euclidean color distance.

    Parameters
    ----------
    series_a, series_b : ndarray, shape (3, T)

    Returns
    -------
    float
    """
    diff = series_a - series_b  # (3, T)
    return float(np.sum(np.sqrt(np.sum(diff ** 2, axis=0))))


def compute_patch_dissonance(positions, series_at):
    """Total neighbor dissonance of a circular patch.

    For each position, computes mean summed-color-distance to all
    8-connected neighbors within the patch, then sums across positions.

    Parameters
    ----------
    positions : list of (gx, gy)
    series_at : dict mapping (gx, gy) -> ndarray (3, T)

    Returns
    -------
    float
    """
    total = 0.0
    pos_set = set(positions)
    for pos in positions:
        neighbors = [
            (pos[0] + dx, pos[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)]
        ]
        neighbors = [n for n in neighbors if n in pos_set]
        if not neighbors:
            continue
        dists = [summed_color_distance(series_at[pos], series_at[n])
                 for n in neighbors]
        total += np.mean(dists)
    return total


# ──────────────────────────────────────────────────────────────────────
# Greedy BFS expansion
# ──────────────────────────────────────────────────────────────────────

ALL_8_DELTAS = [(-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1)]


def greedy_expand_patch(center_pixel_idx, patch_positions, series_flat,
                        mean_colors, shortlist_size=50):
    """Build a circular patch by greedy BFS expansion from the center.

    Starting from the center pixel, maintains a frontier of empty patch
    positions adjacent to already-placed positions.  At each step, pops the
    frontier position with the most placed neighbours, finds the best
    unplaced pixel (by flattened Euclidean distance to placed neighbours),
    and places it.

    Parameters
    ----------
    center_pixel_idx : int
        Flat index of the pixel to place at the center.
    patch_positions : set of (gx, gy)
        All grid positions that belong to the circular patch.
    series_flat : ndarray, shape (N, 3, T)
        All pixel time-series (float64).
    mean_colors : ndarray, shape (N, 3)
        Mean RGB per pixel for fast shortlisting.
    shortlist_size : int
        How many candidates to shortlist by mean-RGB before full evaluation.

    Returns
    -------
    pixel_grid : dict mapping (gx, gy) -> flat pixel index
    """
    N = len(series_flat)
    patch_set = set(patch_positions)

    # Placed tracking
    pixel_grid = {}          # (gx, gy) -> flat pixel idx
    placed_series = {}       # (gx, gy) -> (3, T) array
    used_pixels = set()

    # Pick the center position (the one at offset (0,0) in the patch)
    # We need to find which position in patch_positions is the center.
    # Caller should ensure the center is in the patch.
    # Find the center as the position closest to the mean of all positions.
    positions_list = list(patch_positions)
    xs = [p[0] for p in positions_list]
    ys = [p[1] for p in positions_list]
    cx = round(np.mean(xs))
    cy = round(np.mean(ys))
    center_pos = (cx, cy)
    if center_pos not in patch_set:
        # Fallback: pick the position closest to centroid
        best = min(positions_list, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        center_pos = best

    # Place center
    pixel_grid[center_pos] = center_pixel_idx
    placed_series[center_pos] = series_flat[center_pixel_idx]
    used_pixels.add(center_pixel_idx)

    # Frontier: priority queue of (-num_placed_neighbors, insertion_order, pos)
    # More placed neighbors = higher priority (popped first due to negation)
    frontier = []
    in_frontier = set()
    counter = 0

    def _add_to_frontier(pos):
        nonlocal counter
        if pos in patch_set and pos not in pixel_grid and pos not in in_frontier:
            # Count placed neighbors
            n_placed = sum(1 for dx, dy in ALL_8_DELTAS
                           if (pos[0]+dx, pos[1]+dy) in pixel_grid)
            heapq.heappush(frontier, (-n_placed, counter, pos))
            counter += 1
            in_frontier.add(pos)

    # Seed frontier from center's neighbors
    for dx, dy in ALL_8_DELTAS:
        _add_to_frontier((center_pos[0] + dx, center_pos[1] + dy))

    # Precompute flat series for distance computation
    flat_all = series_flat.reshape(N, -1)  # (N, 3*T)
    sq_norms = np.sum(flat_all ** 2, axis=1)  # (N,)

    # Used mask for fast exclusion
    is_used = np.zeros(N, dtype=bool)
    is_used[center_pixel_idx] = True

    while frontier:
        neg_n, _, pos = heapq.heappop(frontier)
        in_frontier.discard(pos)

        if pos in pixel_grid:
            continue

        # Gather placed neighbor series
        neighbor_series = []
        for dx, dy in ALL_8_DELTAS:
            npos = (pos[0] + dx, pos[1] + dy)
            if npos in placed_series:
                neighbor_series.append(placed_series[npos])

        if not neighbor_series:
            # No placed neighbors — shouldn't happen in BFS, but handle anyway
            continue

        # Mean reference series from placed neighbors
        ref_series = np.mean(neighbor_series, axis=0)  # (3, T)
        ref_flat = ref_series.reshape(1, -1)  # (1, 3*T)
        ref_mean_color = ref_series.mean(axis=1)  # (3,)

        # Stage 1: Coarse shortlist by mean-RGB distance
        color_dists = np.sum((mean_colors - ref_mean_color) ** 2, axis=1)
        color_dists[is_used] = np.inf  # exclude already-used pixels
        S = min(shortlist_size, int(np.sum(~is_used)))
        if S <= 0:
            break
        top_idx = np.argpartition(color_dists, S)[:S]

        # Stage 2: Full time-series Euclidean distance on shortlist
        ref_sq = float(np.sum(ref_flat ** 2))
        cand_flat = flat_all[top_idx]  # (S, 3*T)
        dists = sq_norms[top_idx] + ref_sq - 2.0 * (cand_flat @ ref_flat.T).ravel()
        best_local = np.argmin(dists)
        best_pix = int(top_idx[best_local])

        # Place it
        pixel_grid[pos] = best_pix
        placed_series[pos] = series_flat[best_pix]
        used_pixels.add(best_pix)
        is_used[best_pix] = True

        # Expand frontier
        for dx, dy in ALL_8_DELTAS:
            _add_to_frontier((pos[0] + dx, pos[1] + dy))

    return pixel_grid


# ──────────────────────────────────────────────────────────────────────
# Patch rendering
# ──────────────────────────────────────────────────────────────────────

def render_patch_image(positions, series_at, frame_idx, padding=2):
    """Render a circular patch as a PIL Image.

    Parameters
    ----------
    positions : list of (gx, gy)
    series_at : dict mapping (gx, gy) -> ndarray (3, T)
    frame_idx : int
    padding : int

    Returns
    -------
    PIL.Image (RGB)
    """
    if not positions:
        return Image.new('RGB', (1, 1), (40, 40, 40))

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    w = max_x - min_x + 1 + 2 * padding
    h = max_y - min_y + 1 + 2 * padding

    img = np.full((h, w, 3), 40, dtype=np.uint8)

    for gx, gy in positions:
        series = series_at[(gx, gy)]  # (3, T)
        color = np.clip(series[:, frame_idx], 0, 255).astype(np.uint8)
        ix = gx - min_x + padding
        iy = gy - min_y + padding
        img[iy, ix] = color

    return Image.fromarray(img, 'RGB')


# ──────────────────────────────────────────────────────────────────────
# PyQt5 GUI
# ──────────────────────────────────────────────────────────────────────

class PinwheelExperimentGUI(QMainWindow):
    def __init__(self, args):
        super().__init__()
        method_name = args.method.capitalize()
        self.setWindowTitle(f"{method_name} Experiment — Ground Truth vs Reconstruction")
        self.setGeometry(100, 100, 900, 700)

        self.args = args

        # State filled by _run_experiment
        self.wall = None
        self.num_frames = 0
        self.gt_positions = []
        self.pw_positions = []
        self.gt_series = {}      # (gx,gy) -> (3, T)
        self.pw_series = {}      # (gx,gy) -> (3, T)
        self.gt_dissonance = 0.0
        self.pw_dissonance = 0.0
        self.correct = 0
        self.total_in_patch = 0

        # Keep byte buffers alive for QImage
        self._left_data = None
        self._right_data = None

        self._setup_ui()
        self._run_experiment()

    # ── UI ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Frame slider ──
        frame_group = QGroupBox("Frame")
        frame_lay = QHBoxLayout(frame_group)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setMinimumWidth(120)
        frame_lay.addWidget(self.frame_label)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        frame_lay.addWidget(self.frame_slider)
        root.addWidget(frame_group)

        # ── Metrics ──
        metrics_group = QGroupBox("Dissonance  (Summed Color Distance)")
        metrics_lay = QVBoxLayout(metrics_group)
        self.gt_diss_label = QLabel("Ground truth:  —")
        self.pw_diss_label = QLabel("Pinwheel:      —")
        self.diff_label = QLabel("Difference:    —")
        self.acc_label = QLabel("Accuracy:      —")
        self.frames_label = QLabel("Frames:        —")
        for lbl in (self.gt_diss_label, self.pw_diss_label,
                    self.diff_label, self.acc_label, self.frames_label):
            lbl.setStyleSheet(
                "font-family: 'Consolas', 'Courier New', monospace; font-size: 12pt;"
            )
            metrics_lay.addWidget(lbl)
        root.addWidget(metrics_group)

        # ── Side-by-side panels ──
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(10)

        # Left: Ground Truth
        left_col = QVBoxLayout()
        left_title = QLabel("Ground Truth")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #d63384;"
        )
        left_col.addWidget(left_title)
        self.left_panel = QLabel()
        self.left_panel.setAlignment(Qt.AlignCenter)
        self.left_panel.setStyleSheet(
            "background-color: #f8d7e3; border: 2px solid #ffb6c1;"
            "border-radius: 8px;"
        )
        self.left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_panel.setMinimumSize(300, 300)
        left_col.addWidget(self.left_panel)
        panels_layout.addLayout(left_col)

        # Right: Reconstruction
        right_col = QVBoxLayout()
        method_name = self.args.method.capitalize()
        right_title = QLabel(f"{method_name} Reconstruction")
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #d63384;"
        )
        right_col.addWidget(right_title)
        self.right_panel = QLabel()
        self.right_panel.setAlignment(Qt.AlignCenter)
        self.right_panel.setStyleSheet(
            "background-color: #f8d7e3; border: 2px solid #ffb6c1;"
            "border-radius: 8px;"
        )
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_panel.setMinimumSize(300, 300)
        right_col.addWidget(self.right_panel)
        panels_layout.addLayout(right_col)

        root.addLayout(panels_layout, stretch=1)

    # ── Experiment ──────────────────────────────────────────────────

    def _run_experiment(self):
        a = self.args

        # 1. Load video with the FINE frame count (used for TSP, rotation,
        #    dissonance evaluation, and display).  The coarse subset for
        #    initial candidate selection is derived by subsampling.
        self.wall = TVWall(a.video, num_frames=a.fine_frames, stride=a.stride,
                           crop_percent=a.crop)
        H, W, T = self.wall.height, self.wall.width, self.wall.num_frames
        self.num_frames = T

        # 2. Build lattice positions
        rings = build_lattice_rings(a.radius)
        cx = a.center_x if a.center_x is not None else W // 2
        cy = a.center_y if a.center_y is not None else H // 2

        all_offsets = []
        for ring in rings:
            all_offsets.extend(ring)

        gt_positions = []
        for dx, dy in all_offsets:
            gx, gy = cx + dx, cy + dy
            if 0 <= gx < W and 0 <= gy < H:
                gt_positions.append((gx, gy))
        self.gt_positions = gt_positions

        # 3. Extract ground truth series BEFORE scrambling (full fine frames)
        all_series_orig = self.wall.get_all_series(force_cpu=True)  # (H, W, 3, T)
        self.gt_series = {}
        for gx, gy in gt_positions:
            self.gt_series[(gx, gy)] = all_series_orig[gy, gx].astype(np.float64)

        # 4. Compute ground truth dissonance (on fine frames)
        self.gt_dissonance = compute_patch_dissonance(gt_positions, self.gt_series)

        # 5. Scramble
        self.wall.scramble(seed=a.scramble_seed)

        # 6. Build scrambled series (fine = all T frames)
        all_series_scr = self.wall.get_all_series(force_cpu=True)  # (H, W, 3, T)
        series_flat_fine = all_series_scr.reshape(H * W, 3, T).astype(np.float64)

        # 7. Build coarse series by subsampling fine frames
        coarse_t = min(a.coarse_frames, T)
        if coarse_t < T:
            coarse_idx = np.linspace(0, T - 1, coarse_t, dtype=int)
            series_flat_coarse = series_flat_fine[:, :, coarse_idx]
        else:
            series_flat_coarse = series_flat_fine
        mean_colors = series_flat_coarse.mean(axis=2)  # (H*W, 3)

        print(f"  Frames: {T} fine, {coarse_t} coarse", flush=True)

        # 8. Find center pixel (the pixel originally at (cx, cy))
        cur_x, cur_y = self.wall.get_current_position(cx, cy)
        center_pixel_idx = cur_y * W + cur_x

        # ── Dispatch to selected construction method ──────────────────
        method = a.method
        patch_set = set(gt_positions)

        if method == 'greedy':
            pixel_grid = greedy_expand_patch(
                center_pixel_idx, patch_set, series_flat_fine,
                mean_colors, shortlist_size=a.shortlist,
            )
        else:
            # ── Pinwheel method ──────────────────────────────────────
            # Phase A: Candidate selection using COARSE frames
            pw = Pinwheel(
                center_grid_pos=(cx, cy),
                lattice_rings=rings,
                series_flat=series_flat_coarse,
                mean_colors=mean_colors,
                distance_metric='euclidean',
                use_2opt=False,
            )
            pw.build(center_pixel_idx)

            # Phase B: Re-do TSP ordering using FINE frames
            for i in range(1, len(pw.rings)):
                if len(pw.rings[i]) > 2:
                    pw.rings[i] = solve_ring_tsp(
                        pw.rings[i], series_flat_fine, True
                    )

            # Phase C: Re-do rotation optimization using FINE frames
            pw._series_flat = series_flat_fine
            pw.ring_rotations = [0] * len(pw.rings)
            pw._optimise_rotations(placed_series_at=None)
            pw._assign_grid_positions()

            pixel_grid = {
                (gx, gy): pix_idx
                for (gx, gy), pix_idx in pw.pixel_grid.items()
                if 0 <= gx < W and 0 <= gy < H
            }

        # 9. Extract reconstruction series (from fine frames)
        self.pw_series = {}
        pw_positions = []
        for (gx, gy), pix_idx in pixel_grid.items():
            if 0 <= gx < W and 0 <= gy < H:
                self.pw_series[(gx, gy)] = series_flat_fine[pix_idx]
                pw_positions.append((gx, gy))
        self.pw_positions = pw_positions

        # 10. Compute reconstruction dissonance (on fine frames)
        self.pw_dissonance = compute_patch_dissonance(pw_positions, self.pw_series)

        # 11. Compute accuracy
        correct = 0
        total = 0
        for (gx, gy), pix_idx in pixel_grid.items():
            if not (0 <= gx < W and 0 <= gy < H):
                continue
            total += 1
            src_x = pix_idx % W
            src_y = pix_idx // W
            orig_x = int(self.wall._perm_x[src_y, src_x])
            orig_y = int(self.wall._perm_y[src_y, src_x])
            if orig_x == gx and orig_y == gy:
                correct += 1
        self.correct = correct
        self.total_in_patch = total

        # 12. Print results to console
        method_label = method.upper()
        diff = self.pw_dissonance - self.gt_dissonance
        pct = 100.0 * correct / max(1, total)
        print(f"\n{'=' * 50}")
        print(f"  {method_label} EXPERIMENT  (radius={a.radius})")
        print(f"{'=' * 50}")
        print(f"  Method:          {method}")
        print(f"  Grid:            {W}x{H}")
        if method == 'pinwheel':
            print(f"  Coarse frames:   {coarse_t}  (candidate selection)")
            print(f"  Fine frames:     {T}  (TSP, rotation, dissonance)")
        else:
            print(f"  Frames:          {T}")
            print(f"  Shortlist:       {a.shortlist}")
        print(f"  Center:          ({cx}, {cy})")
        print(f"  Patch pixels:    {total}")
        print(f"  GT dissonance:   {self.gt_dissonance:,.2f}")
        print(f"  Recon dissonance:{self.pw_dissonance:,.2f}")
        print(f"  Difference:      {diff:+,.2f}")
        print(f"  Accuracy:        {correct}/{total} ({pct:.1f}%)")
        print(f"{'=' * 50}\n", flush=True)

        # 13. Update UI
        self.frame_slider.setRange(0, T - 1)
        self.frame_slider.setValue(0)

        self.gt_diss_label.setText(f"Ground truth:  {self.gt_dissonance:,.2f}")
        self.pw_diss_label.setText(f"{method_label}:".ljust(15) + f"{self.pw_dissonance:,.2f}")
        self.diff_label.setText(f"Difference:    {diff:+,.2f}")
        self.acc_label.setText(
            f"Accuracy:      {correct} / {total} correct ({pct:.1f}%)"
        )
        if method == 'pinwheel':
            self.frames_label.setText(
                f"Frames:        {coarse_t} coarse / {T} fine"
            )
        else:
            self.frames_label.setText(
                f"Frames:        {T}  |  Shortlist: {a.shortlist}"
            )

        self._update_display(0)

    # ── Display ─────────────────────────────────────────────────────

    def _on_frame_changed(self, frame_idx):
        self.frame_label.setText(
            f"Frame: {frame_idx} / {max(0, self.num_frames - 1)}"
        )
        self._update_display(frame_idx)

    def _update_display(self, frame_idx):
        # Render patches
        gt_img = render_patch_image(self.gt_positions, self.gt_series, frame_idx)
        pw_img = render_patch_image(self.pw_positions, self.pw_series, frame_idx)

        # Compute zoom scale: fit the patch into ~350px
        patch_w, patch_h = gt_img.size
        target = 350
        scale = max(4, target // max(patch_w, patch_h, 1))

        gt_img = gt_img.resize(
            (patch_w * scale, patch_h * scale), Image.Resampling.NEAREST
        )
        pw_img = pw_img.resize(
            (patch_w * scale, patch_h * scale), Image.Resampling.NEAREST
        )

        self._set_pixmap(self.left_panel, gt_img, '_left_data')
        self._set_pixmap(self.right_panel, pw_img, '_right_data')

    def _set_pixmap(self, label, pil_img, data_attr):
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        data = pil_img.tobytes("raw", "RGB")
        setattr(self, data_attr, data)  # prevent GC
        qimg = QImage(data, pil_img.width, pil_img.height,
                      pil_img.width * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))


# ──────────────────────────────────────────────────────────────────────
# Pink stylesheet
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
QLabel { background-color: transparent; color: #8b4563; }
QSlider::groove:horizontal { height: 8px; background-color: #ffe4ec;
                              border-radius: 4px; border: 1px solid #ffb6c1; }
QSlider::handle:horizontal { background-color: #ff85a2; width: 18px;
                              height: 18px; margin: -6px 0; border-radius: 9px;
                              border: 2px solid #fff; }
QSlider::handle:horizontal:hover { background-color: #ff6b8a; }
QSlider::sub-page:horizontal { background-color: #ffb6c1; border-radius: 4px; }
QScrollBar:vertical { background-color: #ffe4ec; width: 12px; border-radius: 6px; }
QScrollBar::handle:vertical { background-color: #ffb6c1; border-radius: 5px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background-color: #ff85a2; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
"""


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pinwheel Experiment — compare single pinwheel vs ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-v", "--video", type=str, required=True, help="Input video file")
    p.add_argument("--coarse-frames", type=int, default=100,
                   help="Frames for initial candidate selection (coarse, fast)")
    p.add_argument("--fine-frames", type=int, default=500,
                   help="Frames for TSP ordering, rotation, and dissonance (fine, thorough)")
    p.add_argument("-s", "--stride", type=int, default=1, help="Frame stride")
    p.add_argument("--crop", type=float, default=100, help="Crop percent")
    p.add_argument("--scramble-seed", type=int, default=42, help="Scramble seed")
    p.add_argument("--radius", type=int, default=5, help="Pinwheel radius (rings)")
    p.add_argument("--center-x", type=int, default=None,
                   help="Center X position (default: grid center)")
    p.add_argument("--center-y", type=int, default=None,
                   help="Center Y position (default: grid center)")
    p.add_argument("--method", choices=["pinwheel", "greedy"], default="pinwheel",
                   help="Construction method: pinwheel (ring+TSP) or greedy (BFS expansion)")
    p.add_argument("--shortlist", type=int, default=50,
                   help="Shortlist size for greedy method's candidate search per step")
    return p.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyleSheet(PINK_STYLESHEET)
    gui = PinwheelExperimentGUI(args)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
