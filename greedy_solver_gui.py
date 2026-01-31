#!/usr/bin/env python
# coding: utf-8
"""
Greedy Solver GUI

Interactive GUI to solve scrambled TV walls using a two-phase approach:
1. Identify high-dissonance positions (likely misplaced TVs)
2. Optimize by rearranging only those positions to minimize total dissonance

Features:
- Multiple scrambling methods (pair swaps, short swaps, full scramble)
- Multiple solving strategies (greedy, top-K permutation, simulated annealing)
- Real-time visualization with animation
- Metrics display showing progress and accuracy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import threading
import time
from itertools import permutations
import os
import subprocess
import sys

from tv_wall import TVWall


class GreedySolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Greedy Solver - Unscramble Video")
        self.root.geometry("1500x950")

        self.wall = None
        self.all_series = None  # Precomputed color series (original positions)
        self.current_series = None  # Series with current permutation
        self.dissonance_map = None

        # Scramble tracking
        self.swapped_positions = set()
        self.swap_pairs = []

        # Solver state
        self.high_dissonance_positions = set()
        self.solver_running = False
        self.solver_paused = False
        self.solver_thread = None
        self.iteration = 0
        self.total_dissonance_history = []
        self.correct_count_history = []

        # Display settings
        self.display_scale = 1
        self.current_frame = 0

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - controls
        left_panel = ttk.Frame(main_frame, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left_panel.pack_propagate(False)

        # Video loading
        load_frame = ttk.LabelFrame(left_panel, text="Load Video", padding="5")
        load_frame.pack(fill=tk.X, pady=5)

        ttk.Button(load_frame, text="Open Video...", command=self.load_video).pack(fill=tk.X)

        self.video_label = ttk.Label(load_frame, text="No video loaded", wraplength=300)
        self.video_label.pack(fill=tk.X, pady=5)

        # Parameters
        param_frame = ttk.LabelFrame(left_panel, text="Parameters", padding="5")
        param_frame.pack(fill=tk.X, pady=5)

        params = [
            ("Frames:", "frames_var", "50"),
            ("Stride:", "stride_var", "10"),
            ("Crop % (1-100):", "crop_var", "100"),
            ("DTW Window (0-1):", "window_var", "0.1"),
            ("Kernel Size (odd):", "kernel_var", "3"),
        ]

        for label_text, var_name, default in params:
            row = ttk.Frame(param_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label_text, width=18).pack(side=tk.LEFT)
            setattr(self, var_name, tk.StringVar(value=default))
            ttk.Entry(row, textvariable=getattr(self, var_name), width=8).pack(side=tk.LEFT)

        # Distance metric selection
        metric_row = ttk.Frame(param_frame)
        metric_row.pack(fill=tk.X, pady=2)
        ttk.Label(metric_row, text="Distance Metric:", width=18).pack(side=tk.LEFT)
        self.metric_var = tk.StringVar(value="euclidean")
        metric_combo = ttk.Combobox(metric_row, textvariable=self.metric_var,
                                    values=["dtw", "euclidean", "manhattan", "cosine"],
                                    width=10, state="readonly")
        metric_combo.pack(side=tk.LEFT)

        # Scramble controls
        scramble_frame = ttk.LabelFrame(left_panel, text="Scramble", padding="5")
        scramble_frame.pack(fill=tk.X, pady=5)

        # Number of swaps
        swap_row = ttk.Frame(scramble_frame)
        swap_row.pack(fill=tk.X, pady=1)
        ttk.Label(swap_row, text="Num swaps:", width=12).pack(side=tk.LEFT)
        self.num_swaps_var = tk.StringVar(value="10")
        ttk.Entry(swap_row, textvariable=self.num_swaps_var, width=6).pack(side=tk.LEFT)

        # Max distance for short swaps
        dist_row = ttk.Frame(scramble_frame)
        dist_row.pack(fill=tk.X, pady=1)
        ttk.Label(dist_row, text="Max dist:", width=12).pack(side=tk.LEFT)
        self.max_dist_var = tk.StringVar(value="10")
        ttk.Entry(dist_row, textvariable=self.max_dist_var, width=6).pack(side=tk.LEFT)
        ttk.Label(dist_row, text="px").pack(side=tk.LEFT)

        # Scramble buttons
        btn_frame = ttk.Frame(scramble_frame)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Pair Swap", command=self.pair_swap, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Short Swap", command=self.short_swap, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Full Scramble", command=self.full_scramble, width=11).pack(side=tk.LEFT, padx=1)

        ttk.Button(scramble_frame, text="Reset All", command=self.reset_all).pack(fill=tk.X, pady=2)

        self.scramble_info = ttk.Label(scramble_frame, text="Swapped: 0 positions", wraplength=300)
        self.scramble_info.pack(fill=tk.X)

        # Solver controls
        solver_frame = ttk.LabelFrame(left_panel, text="Solver", padding="5")
        solver_frame.pack(fill=tk.X, pady=5)

        # Strategy selection
        strat_row = ttk.Frame(solver_frame)
        strat_row.pack(fill=tk.X, pady=2)
        ttk.Label(strat_row, text="Strategy:", width=10).pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="greedy")
        strat_combo = ttk.Combobox(strat_row, textvariable=self.strategy_var,
                                   values=["greedy", "top_k_best", "simulated_annealing"],
                                   width=16, state="readonly")
        strat_combo.pack(side=tk.LEFT)

        # Top-N parameter (for identifying high-dissonance positions)
        topn_row = ttk.Frame(solver_frame)
        topn_row.pack(fill=tk.X, pady=1)
        ttk.Label(topn_row, text="Top-N:", width=10).pack(side=tk.LEFT)
        self.topn_var = tk.StringVar(value="20")
        ttk.Entry(topn_row, textvariable=self.topn_var, width=6).pack(side=tk.LEFT)
        ttk.Label(topn_row, text="(# high-diss to identify)", font=("TkDefaultFont", 8)).pack(side=tk.LEFT)

        # Top-K parameter (for solver strategies)
        topk_row = ttk.Frame(solver_frame)
        topk_row.pack(fill=tk.X, pady=1)
        ttk.Label(topk_row, text="Top-K:", width=10).pack(side=tk.LEFT)
        self.topk_var = tk.StringVar(value="5")
        ttk.Entry(topk_row, textvariable=self.topk_var, width=6).pack(side=tk.LEFT)
        ttk.Label(topk_row, text="(for top_k and SA)", font=("TkDefaultFont", 8)).pack(side=tk.LEFT)

        # Max iterations
        iter_row = ttk.Frame(solver_frame)
        iter_row.pack(fill=tk.X, pady=1)
        ttk.Label(iter_row, text="Max iters:", width=10).pack(side=tk.LEFT)
        self.max_iters_var = tk.StringVar(value="100")
        ttk.Entry(iter_row, textvariable=self.max_iters_var, width=6).pack(side=tk.LEFT)

        # Animation delay
        delay_row = ttk.Frame(solver_frame)
        delay_row.pack(fill=tk.X, pady=1)
        ttk.Label(delay_row, text="Delay (ms):", width=10).pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="100")
        ttk.Entry(delay_row, textvariable=self.delay_var, width=6).pack(side=tk.LEFT)

        # SA parameters
        sa_frame = ttk.Frame(solver_frame)
        sa_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sa_frame, text="SA Temp:", width=10).pack(side=tk.LEFT)
        self.temp_var = tk.StringVar(value="1.0")
        ttk.Entry(sa_frame, textvariable=self.temp_var, width=5).pack(side=tk.LEFT)
        ttk.Label(sa_frame, text="Cool:", width=4).pack(side=tk.LEFT)
        self.cooling_var = tk.StringVar(value="0.95")
        ttk.Entry(sa_frame, textvariable=self.cooling_var, width=5).pack(side=tk.LEFT)

        # Solver action buttons
        btn_frame2 = ttk.Frame(solver_frame)
        btn_frame2.pack(fill=tk.X, pady=3)

        self.identify_btn = ttk.Button(btn_frame2, text="1. Identify", command=self.identify_high_dissonance, width=10)
        self.identify_btn.pack(side=tk.LEFT, padx=1)

        self.solve_btn = ttk.Button(btn_frame2, text="2. Solve", command=self.start_solver, width=8)
        self.solve_btn.pack(side=tk.LEFT, padx=1)

        self.pause_btn = ttk.Button(btn_frame2, text="Pause", command=self.toggle_pause, width=6, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=1)

        self.stop_btn = ttk.Button(btn_frame2, text="Stop", command=self.stop_solver, width=5, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=1)

        ttk.Button(solver_frame, text="Step (one iteration)", command=self.step_solver).pack(fill=tk.X, pady=2)

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(solver_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(solver_frame, text="Ready", wraplength=300)
        self.status_label.pack(fill=tk.X)

        # Results / Metrics
        results_frame = ttk.LabelFrame(left_panel, text="Metrics", padding="5")
        results_frame.pack(fill=tk.X, pady=5, expand=True)

        self.results_text = tk.Text(results_frame, height=14, width=38, font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # View controls
        view_frame = ttk.LabelFrame(left_panel, text="View", padding="5")
        view_frame.pack(fill=tk.X, pady=5)

        self.view_mode = tk.StringVar(value="video")
        views = [("Video Frame", "video"), ("Heatmap Overlay", "heatmap_overlay"),
                 ("Correctness Overlay", "correctness_overlay"), ("Heatmap Only", "heatmap")]
        for text, val in views:
            ttk.Radiobutton(view_frame, text=text, variable=self.view_mode,
                           value=val, command=self.update_display).pack(anchor=tk.W)

        self.overlay_alpha_var = tk.StringVar(value="0.4")
        alpha_row = ttk.Frame(view_frame)
        alpha_row.pack(fill=tk.X, pady=2)
        ttk.Label(alpha_row, text="Overlay alpha:").pack(side=tk.LEFT)
        ttk.Entry(alpha_row, textvariable=self.overlay_alpha_var, width=5).pack(side=tk.LEFT)

        # Frame slider
        ttk.Label(view_frame, text="Frame:").pack(anchor=tk.W)
        self.frame_slider = ttk.Scale(view_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                      command=self.on_frame_change)
        self.frame_slider.pack(fill=tk.X)

        # Right panel - canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Canvas for main display
        self.canvas = tk.Canvas(right_panel, bg='#333333')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse motion for position info
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # Position info
        self.position_label = ttk.Label(right_panel, text="Position: -")
        self.position_label.pack(anchor=tk.W)

    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mkv *.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            num_frames = int(self.frames_var.get())
            stride = int(self.stride_var.get())
            crop_percent = float(self.crop_var.get())

            self.status_label.config(text="Loading video...")
            self.root.update()

            self.wall = TVWall(filepath, num_frames=num_frames, stride=stride, crop_percent=crop_percent)

            filename = os.path.basename(filepath)
            self.video_label.config(text=f"{filename}\n"
                                        f"Size: {self.wall.width}x{self.wall.height}\n"
                                        f"Frames: {self.wall.num_frames}")

            # Precompute color series for original positions
            self.precompute_series()

            # Reset state
            self.reset_state()

            # Update frame slider
            self.frame_slider.config(to=self.wall.num_frames - 1)

            # Calculate display scale
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 600
            scale_x = canvas_width / self.wall.width
            scale_y = canvas_height / self.wall.height
            self.display_scale = max(1, min(scale_x, scale_y, 10))

            self.update_display()
            self.status_label.config(text="Video loaded")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

    def precompute_series(self):
        """Precompute all TV color series for original positions."""
        if self.wall is None:
            return

        height, width = self.wall.height, self.wall.width
        n_frames = self.wall.num_frames

        self.all_series = np.zeros((height, width, 3, n_frames), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                self.all_series[y, x] = self.wall.get_tv_color_series(x, y).T

    def reset_state(self):
        """Reset all solver and scramble state."""
        self.swapped_positions = set()
        self.swap_pairs = []
        self.high_dissonance_positions = set()
        self.dissonance_map = None
        self.current_series = None
        self.iteration = 0
        self.total_dissonance_history = []
        self.correct_count_history = []
        self.solver_running = False
        self.solver_paused = False
        self.update_scramble_info()
        self.results_text.delete(1.0, tk.END)

    def update_scramble_info(self):
        """Update the scramble info label."""
        if self.wall is None:
            self.scramble_info.config(text="Swapped: 0 positions")
            return

        n_swapped = len(self.swapped_positions)
        n_pairs = len(self.swap_pairs)
        correct = self.count_correct_positions()
        total = self.wall.num_tvs
        self.scramble_info.config(text=f"Swapped: {n_swapped} | Pairs: {n_pairs} | "
                                       f"Correct: {correct}/{total}")

    def count_correct_positions(self):
        """Count positions where the TV is in its original location."""
        if self.wall is None:
            return 0
        count = 0
        for y in range(self.wall.height):
            for x in range(self.wall.width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                if orig_x == x and orig_y == y:
                    count += 1
        return count

    def pair_swap(self):
        """Perform random pair swaps (unlimited distance)."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        try:
            num_swaps = int(self.num_swaps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of swaps")
            return

        swap_pairs = self.wall.pair_swaps(num_swaps)
        for pos1, pos2 in swap_pairs:
            self.swapped_positions.add(pos1)
            self.swapped_positions.add(pos2)
            self.swap_pairs.append((pos1, pos2))

        self.dissonance_map = None
        self.high_dissonance_positions = set()
        self.update_scramble_info()
        self.update_display()

    def short_swap(self):
        """Perform random swaps within max distance."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        try:
            num_swaps = int(self.num_swaps_var.get())
            max_dist = int(self.max_dist_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number")
            return

        swap_pairs = self.wall.short_swaps(num_swaps, max_dist)
        for pos1, pos2 in swap_pairs:
            self.swapped_positions.add(pos1)
            self.swapped_positions.add(pos2)
            self.swap_pairs.append((pos1, pos2))

        if len(swap_pairs) < num_swaps:
            messagebox.showinfo("Info", f"Made {len(swap_pairs)}/{num_swaps} swaps")

        self.dissonance_map = None
        self.high_dissonance_positions = set()
        self.update_scramble_info()
        self.update_display()

    def full_scramble(self):
        """Fully scramble all positions."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        self.wall.scramble()
        self.swapped_positions = set((x, y) for y in range(self.wall.height)
                                      for x in range(self.wall.width))
        self.swap_pairs = []  # Can't track individual pairs for full scramble

        self.dissonance_map = None
        self.high_dissonance_positions = set()
        self.update_scramble_info()
        self.update_display()

    def reset_all(self):
        """Reset all swaps and solver state."""
        if self.wall is None:
            return

        self.stop_solver()
        self.wall.reset_swaps()
        self.reset_state()
        self.update_display()

    def get_current_series(self):
        """Get color series array with current permutation (vectorized)."""
        if self.wall is None:
            return None

        # Use the optimized TVWall method which uses NumPy advanced indexing
        return self.wall.get_all_series()

    def identify_high_dissonance(self):
        """Phase 1: Identify high-dissonance positions using clustering."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        if len(self.swapped_positions) == 0:
            messagebox.showinfo("Info", "No swapped positions to identify")
            return

        self.status_label.config(text="Computing dissonance map...")
        self.root.update()

        # Run in background thread
        thread = threading.Thread(target=self._identify_worker)
        thread.start()

    def _identify_worker(self):
        """Background worker for Phase 1."""
        try:
            kernel_size = int(self.kernel_var.get())
            if kernel_size < 3:
                kernel_size = 3
            if kernel_size % 2 == 0:
                kernel_size += 1

            window = float(self.window_var.get())
            metric = self.metric_var.get()

            # Get current series
            self.current_series = self.get_current_series()

            # Compute full dissonance map
            height, width = self.wall.height, self.wall.width
            self.dissonance_map = np.zeros((height, width))

            total = height * width
            for idx, (y, x) in enumerate([(y, x) for y in range(height) for x in range(width)]):
                self.dissonance_map[y, x] = self.wall.compute_position_dissonance(
                    x, y, self.current_series, kernel_size, metric, window
                )
                if idx % 100 == 0:
                    progress = (idx + 1) / total * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))

            # Get top-N most dissonant positions
            self.high_dissonance_positions = self._get_top_n_dissonance()

            self.root.after(0, self._on_identify_complete)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Identification failed: {e}"))
            self.root.after(0, lambda: self.status_label.config(text="Error"))

    def _get_top_n_dissonance(self):
        """Get the top-N most dissonant positions."""
        if self.dissonance_map is None:
            return set()

        try:
            top_n = int(self.topn_var.get())
        except ValueError:
            top_n = 20

        # Flatten and sort dissonance values with positions (descending)
        height, width = self.wall.height, self.wall.width
        values_with_pos = []
        for y in range(height):
            for x in range(width):
                values_with_pos.append((self.dissonance_map[y, x], (x, y)))

        values_with_pos.sort(key=lambda x: x[0], reverse=True)

        # Take top-N positions
        top_n = min(top_n, len(values_with_pos))
        high_positions = set()
        for i in range(top_n):
            _, pos = values_with_pos[i]
            high_positions.add(pos)

        return high_positions

    def _on_identify_complete(self):
        """Called when Phase 1 is complete."""
        self.progress_var.set(100)
        self.status_label.config(text=f"Identified {len(self.high_dissonance_positions)} high-dissonance positions")
        self.update_metrics()
        self.update_display()

    def update_metrics(self):
        """Update the metrics display."""
        self.results_text.delete(1.0, tk.END)

        if self.wall is None:
            return

        lines = []
        lines.append("=" * 38)
        lines.append("         SOLVER METRICS")
        lines.append("=" * 38)

        # Position counts
        total = self.wall.num_tvs
        correct = self.count_correct_positions()
        incorrect = total - correct
        lines.append(f"Total positions: {total}")
        lines.append(f"Currently correct: {correct} ({100*correct/total:.1f}%)")
        lines.append(f"Currently wrong: {incorrect}")
        lines.append("")

        # Ground truth (swapped positions)
        n_swapped = len(self.swapped_positions)
        lines.append(f"Originally swapped: {n_swapped}")

        # High dissonance detection
        n_high = len(self.high_dissonance_positions)
        lines.append(f"Detected high-diss: {n_high}")

        if n_swapped > 0 and n_high > 0:
            # Precision and recall
            true_positives = len(self.high_dissonance_positions & self.swapped_positions)
            precision = true_positives / n_high if n_high > 0 else 0
            recall = true_positives / n_swapped if n_swapped > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            lines.append(f"True positives: {true_positives}")
            lines.append(f"Precision: {precision:.2f}")
            lines.append(f"Recall: {recall:.2f}")
            lines.append(f"F1 Score: {f1:.2f}")

        lines.append("")

        # Dissonance stats
        if self.dissonance_map is not None:
            d_mean = self.dissonance_map.mean()
            d_std = self.dissonance_map.std()
            d_min = self.dissonance_map.min()
            d_max = self.dissonance_map.max()
            lines.append(f"Dissonance mean: {d_mean:.1f}")
            lines.append(f"Dissonance std: {d_std:.1f}")
            lines.append(f"Dissonance range: [{d_min:.1f}, {d_max:.1f}]")

            # High vs low dissonance comparison
            if self.high_dissonance_positions:
                high_vals = [self.dissonance_map[y, x] for x, y in self.high_dissonance_positions]
                low_vals = [self.dissonance_map[y, x] for y in range(self.wall.height)
                           for x in range(self.wall.width)
                           if (x, y) not in self.high_dissonance_positions]

                if high_vals and low_vals:
                    lines.append(f"High-diss mean: {np.mean(high_vals):.1f}")
                    lines.append(f"Low-diss mean: {np.mean(low_vals):.1f}")

        lines.append("")

        # Solver progress
        if self.iteration > 0:
            lines.append("-" * 38)
            lines.append(f"Solver iteration: {self.iteration}")
            if self.total_dissonance_history:
                lines.append(f"Initial total diss: {self.total_dissonance_history[0]:.1f}")
                lines.append(f"Current total diss: {self.total_dissonance_history[-1]:.1f}")
                improvement = self.total_dissonance_history[0] - self.total_dissonance_history[-1]
                lines.append(f"Improvement: {improvement:.1f}")

        lines.append("=" * 38)
        self.results_text.insert(tk.END, "\n".join(lines))

    def start_solver(self):
        """Start the solver in a background thread."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        if len(self.high_dissonance_positions) == 0:
            messagebox.showinfo("Info", "Run 'Identify' first to detect high-dissonance positions")
            return

        if self.solver_running:
            return

        self.solver_running = True
        self.solver_paused = False
        self.solve_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause")
        self.stop_btn.config(state=tk.NORMAL)

        self.solver_thread = threading.Thread(target=self._solver_loop)
        self.solver_thread.start()

    def _solver_loop(self):
        """Main solver loop."""
        try:
            max_iters = int(self.max_iters_var.get())
            delay_ms = int(self.delay_var.get())
            strategy = self.strategy_var.get()

            # Initialize history if empty
            if not self.total_dissonance_history:
                total_d = self._compute_high_diss_total()
                self.total_dissonance_history.append(total_d)
                correct = self.count_correct_positions()
                self.correct_count_history.append(correct)

            while self.solver_running and self.iteration < max_iters:
                # Check for pause
                while self.solver_paused and self.solver_running:
                    time.sleep(0.1)

                if not self.solver_running:
                    break

                # Run one iteration
                improved = self._solver_step(strategy)

                self.iteration += 1

                # Record metrics
                total_d = self._compute_high_diss_total()
                self.total_dissonance_history.append(total_d)
                correct = self.count_correct_positions()
                self.correct_count_history.append(correct)

                # Update UI
                progress = (self.iteration / max_iters) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, self.update_metrics)
                self.root.after(0, self.update_display)
                self.root.after(0, self.update_scramble_info)

                # Check for convergence
                if not improved and strategy == "greedy":
                    self.root.after(0, lambda: self.status_label.config(text="Converged (no improvement)"))
                    break

                # Delay for animation
                time.sleep(delay_ms / 1000.0)

            self.root.after(0, self._on_solver_done)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Solver failed: {e}"))
            self.root.after(0, self._on_solver_done)

    def _solver_step(self, strategy):
        """Execute one solver iteration. Returns True if improvement was made."""
        if strategy == "greedy":
            return self._greedy_step()
        elif strategy == "top_k_best":
            return self._top_k_best_step()
        elif strategy == "simulated_annealing":
            return self._sa_step()
        return False

    def _greedy_step(self):
        """Greedy strategy: swap highest-dissonance position with its best neighbor.

        Uses local dissonance optimization: only computes dissonance for the two
        swapped positions rather than the total dissonance of all high-dissonance
        positions.

        Optimized to avoid recomputing full series array on each tentative swap.
        """
        if not self.high_dissonance_positions:
            return False

        kernel_size = int(self.kernel_var.get())
        window = float(self.window_var.get())
        metric = self.metric_var.get()

        # Compute current series once
        self.current_series = self.get_current_series()

        # Batch compute dissonance for all high-dissonance positions at once
        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        # Find position with highest dissonance among high-dissonance set
        best_pos = max(self.high_dissonance_positions, key=lambda p: all_diss[p])
        best_diss = all_diss[best_pos]
        x1, y1 = best_pos
        diss_best_before = best_diss

        best_swap = None
        best_improvement = 0

        # Create local copy for swap evaluation
        series_copy = self.current_series.copy()

        for other_pos in self.high_dissonance_positions:
            if other_pos == best_pos:
                continue

            x2, y2 = other_pos
            diss_other_before = all_diss[other_pos]

            # Swap series values locally (not in the wall)
            series_copy[y1, x1], series_copy[y2, x2] = \
                series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

            # Compute dissonance for both positions after swap
            diss_best_after = self.wall.compute_position_dissonance(
                x1, y1, series_copy, kernel_size, metric, window)
            diss_other_after = self.wall.compute_position_dissonance(
                x2, y2, series_copy, kernel_size, metric, window)

            # Improvement is reduction in combined dissonance of the two positions
            before_sum = diss_best_before + diss_other_before
            after_sum = diss_best_after + diss_other_after
            improvement = before_sum - after_sum

            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = other_pos

            # Revert the local swap
            series_copy[y1, x1], series_copy[y2, x2] = \
                series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

        if best_swap is not None and best_improvement > 0:
            self.wall.swap_positions(best_pos, best_swap)
            self.current_series = self.get_current_series()
            return True

        return False

    def _top_k_best_step(self):
        """Top-K strategy: try all pairwise swaps among top-K highest dissonance.

        Uses local dissonance optimization: only computes dissonance for the two
        swapped positions rather than the total dissonance of all high-dissonance
        positions.

        Optimized to avoid recomputing full series array on each tentative swap.
        Instead, we compute dissonance by swapping series values locally.
        """
        if not self.high_dissonance_positions:
            return False

        topk = int(self.topk_var.get())
        kernel_size = int(self.kernel_var.get())
        window = float(self.window_var.get())
        metric = self.metric_var.get()

        # Compute current series once
        self.current_series = self.get_current_series()

        # Batch compute dissonance for all high-dissonance positions at once
        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        # Get top-K highest dissonance positions
        diss_list = [(all_diss[pos], pos) for pos in self.high_dissonance_positions]
        diss_list.sort(reverse=True)
        top_k_with_diss = diss_list[:topk]
        top_positions = [pos for _, pos in top_k_with_diss]
        diss_before = {pos: d for d, pos in top_k_with_diss}

        if len(top_positions) < 2:
            return False

        best_swap = None
        best_improvement = 0

        # For evaluating swaps, we create a local copy of the series that we can modify
        # Instead of swapping in the wall and recomputing, swap in the series array
        series_copy = self.current_series.copy()

        # Try all pairs - use vectorized swap evaluation
        n_positions = len(top_positions)

        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                pos1 = top_positions[i]
                pos2 = top_positions[j]
                x1, y1 = pos1
                x2, y2 = pos2

                # Get dissonance before swap
                diss1_before = diss_before[pos1]
                diss2_before = diss_before[pos2]

                # Swap series values locally (not in the wall)
                series_copy[y1, x1], series_copy[y2, x2] = \
                    series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

                # Compute dissonance after swap for both positions
                diss1_after = self.wall.compute_position_dissonance(
                    x1, y1, series_copy, kernel_size, metric, window)
                diss2_after = self.wall.compute_position_dissonance(
                    x2, y2, series_copy, kernel_size, metric, window)

                # Improvement is reduction in combined dissonance
                before_sum = diss1_before + diss2_before
                after_sum = diss1_after + diss2_after
                improvement = before_sum - after_sum

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (pos1, pos2)

                # Revert the local swap
                series_copy[y1, x1], series_copy[y2, x2] = \
                    series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

        if best_swap is not None and best_improvement > 0:
            self.wall.swap_positions(best_swap[0], best_swap[1])
            self.current_series = self.get_current_series()
            return True

        return False

    def _sa_step(self):
        """Simulated annealing step: random swap among top-K, accept with probability.

        Uses local dissonance optimization: only computes dissonance for the two
        swapped positions rather than the total dissonance of all high-dissonance
        positions.

        Optimized to avoid recomputing full series array for evaluation.
        """
        if not self.high_dissonance_positions:
            return False

        topk = int(self.topk_var.get())
        temp = float(self.temp_var.get())
        cooling = float(self.cooling_var.get())
        kernel_size = int(self.kernel_var.get())
        window = float(self.window_var.get())
        metric = self.metric_var.get()

        # Compute current series once
        self.current_series = self.get_current_series()

        # Batch compute dissonance for all high-dissonance positions at once
        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        # Get top-K highest dissonance positions
        diss_list = [(all_diss[pos], pos) for pos in self.high_dissonance_positions]
        diss_list.sort(reverse=True)
        top_k_with_diss = diss_list[:topk]
        top_positions = [pos for _, pos in top_k_with_diss]
        diss_before = {pos: d for d, pos in top_k_with_diss}

        if len(top_positions) < 2:
            return False

        # Random swap among top-K
        idx1, idx2 = np.random.choice(len(top_positions), size=2, replace=False)
        pos1, pos2 = top_positions[idx1], top_positions[idx2]
        x1, y1 = pos1
        x2, y2 = pos2

        # Get dissonance before swap
        diss1_before = diss_before[pos1]
        diss2_before = diss_before[pos2]

        # Evaluate swap using local series copy first
        series_copy = self.current_series.copy()
        series_copy[y1, x1], series_copy[y2, x2] = \
            series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

        # Compute dissonance after swap
        diss1_after = self.wall.compute_position_dissonance(
            x1, y1, series_copy, kernel_size, metric, window)
        diss2_after = self.wall.compute_position_dissonance(
            x2, y2, series_copy, kernel_size, metric, window)

        # Delta is increase in combined dissonance (positive = worse)
        before_sum = diss1_before + diss2_before
        after_sum = diss1_after + diss2_after
        delta = after_sum - before_sum

        # Accept if improvement or with probability based on temperature
        accept = False
        if delta < 0:
            accept = True
        else:
            prob = np.exp(-delta / temp) if temp > 0 else 0
            if np.random.random() < prob:
                accept = True

        if accept:
            # Actually perform the swap in the wall
            self.wall.swap_positions(pos1, pos2)
            self.current_series = self.get_current_series()

        # Cool down
        self.temp_var.set(f"{temp * cooling:.4f}")

        return accept

    def _compute_high_diss_total(self):
        """Compute total dissonance over high-dissonance positions only.

        Note: This is only used for metrics/display purposes, not for swap evaluation.
        The solver strategies use local dissonance (comparing just the two swapped
        positions) for efficiency.

        Uses batch computation for better performance.
        """
        if not self.high_dissonance_positions:
            return 0

        kernel_size = int(self.kernel_var.get())
        window = float(self.window_var.get())
        metric = self.metric_var.get()

        if self.current_series is None:
            self.current_series = self.get_current_series()

        # Use batch computation
        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        return sum(all_diss.values())

    def step_solver(self):
        """Execute a single solver step."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        if len(self.high_dissonance_positions) == 0:
            messagebox.showinfo("Info", "Run 'Identify' first")
            return

        strategy = self.strategy_var.get()

        # Initialize if needed
        if not self.total_dissonance_history:
            self.current_series = self.get_current_series()
            total_d = self._compute_high_diss_total()
            self.total_dissonance_history.append(total_d)
            correct = self.count_correct_positions()
            self.correct_count_history.append(correct)

        improved = self._solver_step(strategy)
        self.iteration += 1

        total_d = self._compute_high_diss_total()
        self.total_dissonance_history.append(total_d)
        correct = self.count_correct_positions()
        self.correct_count_history.append(correct)

        status = "Improved" if improved else "No improvement"
        self.status_label.config(text=f"Step {self.iteration}: {status}")

        self.update_metrics()
        self.update_display()
        self.update_scramble_info()
        self.root.update()  # Force UI refresh

    def toggle_pause(self):
        """Toggle solver pause state."""
        if not self.solver_running:
            return

        self.solver_paused = not self.solver_paused
        if self.solver_paused:
            self.pause_btn.config(text="Resume")
            self.status_label.config(text="Paused")
        else:
            self.pause_btn.config(text="Pause")
            self.status_label.config(text="Running...")

    def stop_solver(self):
        """Stop the solver."""
        self.solver_running = False
        self.solver_paused = False
        if self.solver_thread and self.solver_thread.is_alive():
            self.solver_thread.join(timeout=1.0)

        self._on_solver_done()

    def _on_solver_done(self):
        """Called when solver finishes."""
        self.solver_running = False
        self.solver_paused = False
        self.solve_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text=f"Solver stopped at iteration {self.iteration}")

    def update_display(self):
        """Update the canvas display."""
        if self.wall is None:
            return

        mode = self.view_mode.get()

        # Always get the current video frame (shows swapped TVs)
        video_img = self.wall.get_frame_image(self.current_frame)

        if mode == "video":
            img = video_img
        elif mode == "heatmap":
            img = self.get_heatmap_image()
        elif mode == "heatmap_overlay":
            img = self.get_overlay_image(video_img, self.get_heatmap_image())
        elif mode == "correctness_overlay":
            img = self.get_overlay_image(video_img, self.get_correctness_image())
        else:
            img = video_img

        if img is None:
            return

        # Scale image
        new_width = int(self.wall.width * self.display_scale)
        new_height = int(self.wall.height * self.display_scale)
        img = img.resize((new_width, new_height), Image.Resampling.NEAREST)

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def get_overlay_image(self, base_img, overlay_img):
        """Blend base image with overlay using alpha."""
        if base_img is None or overlay_img is None:
            return base_img

        try:
            alpha = float(self.overlay_alpha_var.get())
            alpha = max(0.0, min(1.0, alpha))
        except ValueError:
            alpha = 0.4

        # Convert to same mode if needed
        base = base_img.convert('RGB')
        overlay = overlay_img.convert('RGB')

        # Blend images
        blended = Image.blend(base, overlay, alpha)
        return blended

    def get_heatmap_image(self):
        """Create heatmap image from dissonance map."""
        if self.dissonance_map is None:
            return Image.new('RGB', (self.wall.width, self.wall.height), (50, 50, 50))

        d_min = self.dissonance_map.min()
        d_max = self.dissonance_map.max()

        if d_max > d_min:
            d_norm = (self.dissonance_map - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(self.dissonance_map)

        # Colormap: black -> red -> yellow -> white
        rgb = np.zeros((self.wall.height, self.wall.width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(d_norm * 2 * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((d_norm - 0.5) * 2 * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip((d_norm - 0.8) * 5 * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(rgb, 'RGB')

    def get_correctness_image(self):
        """Create correctness map: green = correct, red = wrong, with intensity based on distance."""
        if self.wall is None:
            return None

        rgb = np.zeros((self.wall.height, self.wall.width, 3), dtype=np.uint8)

        # Find max distance for normalization
        max_dist = np.sqrt(self.wall.width**2 + self.wall.height**2)

        for y in range(self.wall.height):
            for x in range(self.wall.width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                if orig_x == x and orig_y == y:
                    rgb[y, x] = [0, 255, 0]  # Bright green = correct
                else:
                    # Red intensity based on how far from correct position
                    dist = np.sqrt((orig_x - x)**2 + (orig_y - y)**2)
                    intensity = int(100 + 155 * (dist / max_dist))
                    rgb[y, x] = [intensity, 0, 0]  # Red = wrong, brighter = further

        return Image.fromarray(rgb, 'RGB')

    def on_frame_change(self, value):
        """Handle frame slider change."""
        self.current_frame = int(float(value))
        self.update_display()

    def on_canvas_motion(self, event):
        """Handle mouse motion for position info."""
        if self.wall is None:
            return

        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)

        if x < 0 or x >= self.wall.width or y < 0 or y >= self.wall.height:
            self.position_label.config(text="Position: -")
            return

        info = f"Position: ({x}, {y})"

        # Show original position
        orig_x, orig_y = self.wall.get_original_position(x, y)
        if orig_x != x or orig_y != y:
            info += f" [orig: ({orig_x}, {orig_y})]"
        else:
            info += " [correct]"

        # Show dissonance
        if self.dissonance_map is not None:
            d = self.dissonance_map[y, x]
            info += f" | Diss: {d:.1f}"

        # Show if in high-dissonance set
        if (x, y) in self.high_dissonance_positions:
            info += " | HIGH-DISS"

        self.position_label.config(text=info)


def main():
    root = tk.Tk()
    app = GreedySolverGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
