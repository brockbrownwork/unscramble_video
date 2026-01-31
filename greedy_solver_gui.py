#!/usr/bin/env python
# coding: utf-8
"""
Greedy Solver GUI Experiment

Interactive GUI to test greedy swap descent for unscrambling:
1. Load a video and scramble TV positions
2. Run greedy solver that minimizes total neighbor dissonance
3. Watch the solving process in real-time
4. Analyze success rate at different scramble levels
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import math
from collections import deque

from tv_wall import TVWall


class GreedySolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Greedy Solver Experiment")
        self.root.geometry("1500x950")

        self.wall = None
        self.original_series = None  # Original (unscrambled) color series
        self.current_series = None   # Current series with swaps applied
        self.dissonance_map = None   # Per-position dissonance

        self.display_scale = 1
        self.current_frame = 0

        # Solver state
        self.solving = False
        self.solve_thread = None
        self.solve_history = []  # List of (total_dissonance, num_correct) tuples
        self.ground_truth_positions = {}  # Maps current_pos -> original_pos at scramble time

        # Animation
        self.animation_speed = 100  # ms between updates

        self.setup_ui()

    def setup_ui(self):
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

        ttk.Label(param_frame, text="Frames:").pack(anchor=tk.W)
        self.frames_var = tk.StringVar(value="50")
        ttk.Entry(param_frame, textvariable=self.frames_var, width=10).pack(anchor=tk.W)

        ttk.Label(param_frame, text="Stride:").pack(anchor=tk.W)
        self.stride_var = tk.StringVar(value="10")
        ttk.Entry(param_frame, textvariable=self.stride_var, width=10).pack(anchor=tk.W)

        ttk.Label(param_frame, text="DTW Window (0-1):").pack(anchor=tk.W)
        self.window_var = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.window_var, width=10).pack(anchor=tk.W)

        ttk.Label(param_frame, text="Kernel Size (odd):").pack(anchor=tk.W)
        self.kernel_var = tk.StringVar(value="3")
        ttk.Entry(param_frame, textvariable=self.kernel_var, width=10).pack(anchor=tk.W)

        ttk.Label(param_frame, text="Distance Metric:").pack(anchor=tk.W)
        self.distance_metric_var = tk.StringVar(value="dtw")
        metric_combo = ttk.Combobox(param_frame, textvariable=self.distance_metric_var,
                                    values=["dtw", "euclidean", "squared"], width=10, state="readonly")
        metric_combo.pack(anchor=tk.W)

        # Scramble controls
        scramble_frame = ttk.LabelFrame(left_panel, text="Scramble", padding="5")
        scramble_frame.pack(fill=tk.X, pady=5)

        ttk.Label(scramble_frame, text="Number of swaps:").pack(anchor=tk.W)
        self.num_swaps_var = tk.StringVar(value="10")
        ttk.Entry(scramble_frame, textvariable=self.num_swaps_var, width=10).pack(anchor=tk.W)

        ttk.Label(scramble_frame, text="Max swap distance:").pack(anchor=tk.W)
        self.max_dist_var = tk.StringVar(value="10")
        dist_frame = ttk.Frame(scramble_frame)
        dist_frame.pack(anchor=tk.W)
        ttk.Entry(dist_frame, textvariable=self.max_dist_var, width=5).pack(side=tk.LEFT)
        ttk.Label(dist_frame, text=" px (0=unlimited)").pack(side=tk.LEFT)

        btn_frame = ttk.Frame(scramble_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Scramble", command=self.scramble).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=2)

        self.scramble_info = ttk.Label(scramble_frame, text="Status: Unscrambled")
        self.scramble_info.pack(fill=tk.X)

        # Solver controls
        solver_frame = ttk.LabelFrame(left_panel, text="Solver", padding="5")
        solver_frame.pack(fill=tk.X, pady=5)

        ttk.Label(solver_frame, text="Strategy:").pack(anchor=tk.W)
        self.strategy_var = tk.StringVar(value="best_of_topk")
        ttk.Radiobutton(solver_frame, text="Greedy: highest dissonance",
                       variable=self.strategy_var, value="highest_dissonance").pack(anchor=tk.W)
        ttk.Radiobutton(solver_frame, text="Greedy: best of top-K",
                       variable=self.strategy_var, value="best_of_topk").pack(anchor=tk.W)
        ttk.Radiobutton(solver_frame, text="Simulated Annealing",
                       variable=self.strategy_var, value="simulated_annealing").pack(anchor=tk.W)

        # Parameters in a grid
        param_grid = ttk.Frame(solver_frame)
        param_grid.pack(fill=tk.X, pady=2)

        ttk.Label(param_grid, text="Top-K:").grid(row=0, column=0, sticky=tk.W)
        self.topk_var = tk.StringVar(value="20")
        ttk.Entry(param_grid, textvariable=self.topk_var, width=6).grid(row=0, column=1, padx=2)

        ttk.Label(param_grid, text="Max iter:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.max_iter_var = tk.StringVar(value="200")
        ttk.Entry(param_grid, textvariable=self.max_iter_var, width=6).grid(row=0, column=3, padx=2)

        ttk.Label(param_grid, text="Temp:").grid(row=1, column=0, sticky=tk.W)
        self.temp_var = tk.StringVar(value="100")
        ttk.Entry(param_grid, textvariable=self.temp_var, width=6).grid(row=1, column=1, padx=2)

        ttk.Label(param_grid, text="Cooling:").grid(row=1, column=2, sticky=tk.W, padx=(10, 0))
        self.cooling_var = tk.StringVar(value="0.98")
        ttk.Entry(param_grid, textvariable=self.cooling_var, width=6).grid(row=1, column=3, padx=2)

        ttk.Label(solver_frame, text="Animation speed (ms):").pack(anchor=tk.W)
        self.speed_var = tk.StringVar(value="50")
        ttk.Entry(solver_frame, textvariable=self.speed_var, width=10).pack(anchor=tk.W)

        btn_frame2 = ttk.Frame(solver_frame)
        btn_frame2.pack(fill=tk.X, pady=5)
        self.solve_btn = ttk.Button(btn_frame2, text="Start Solving", command=self.toggle_solving)
        self.solve_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="Step Once", command=self.step_once).pack(side=tk.LEFT, padx=2)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(solver_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)

        self.solver_status = ttk.Label(solver_frame, text="Ready")
        self.solver_status.pack(fill=tk.X)

        # Batch experiment
        batch_frame = ttk.LabelFrame(left_panel, text="Batch Experiment", padding="5")
        batch_frame.pack(fill=tk.X, pady=5)

        ttk.Label(batch_frame, text="Test swap counts (comma-sep):").pack(anchor=tk.W)
        self.batch_swaps_var = tk.StringVar(value="2,5,10,20,50")
        ttk.Entry(batch_frame, textvariable=self.batch_swaps_var, width=20).pack(anchor=tk.W)

        ttk.Label(batch_frame, text="Trials per level:").pack(anchor=tk.W)
        self.batch_trials_var = tk.StringVar(value="3")
        ttk.Entry(batch_frame, textvariable=self.batch_trials_var, width=10).pack(anchor=tk.W)

        self.batch_btn = ttk.Button(batch_frame, text="Run Batch", command=self.run_batch)
        self.batch_btn.pack(fill=tk.X, pady=2)

        # Results
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding="5")
        results_frame.pack(fill=tk.X, pady=5, expand=True)

        self.results_text = tk.Text(results_frame, height=14, width=38, font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # View controls
        view_frame = ttk.LabelFrame(left_panel, text="View", padding="5")
        view_frame.pack(fill=tk.X, pady=5)

        self.view_mode = tk.StringVar(value="video")
        ttk.Radiobutton(view_frame, text="Video Frame", variable=self.view_mode,
                       value="video", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="Dissonance Heatmap", variable=self.view_mode,
                       value="heatmap", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="Correctness Map", variable=self.view_mode,
                       value="correctness", command=self.update_display).pack(anchor=tk.W)

        self.show_markers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="Show markers",
                       variable=self.show_markers_var, command=self.update_display).pack(anchor=tk.W)

        ttk.Label(view_frame, text="Frame:").pack(anchor=tk.W)
        self.frame_slider = ttk.Scale(view_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                      command=self.on_frame_change)
        self.frame_slider.pack(fill=tk.X)

        # Right panel - canvas and chart
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Canvas for video/heatmap
        canvas_frame = ttk.Frame(right_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='#333333', height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Chart canvas for solve history
        chart_frame = ttk.LabelFrame(right_panel, text="Solve Progress", padding="5")
        chart_frame.pack(fill=tk.X, pady=5)

        self.chart_canvas = tk.Canvas(chart_frame, bg='#1a1a1a', height=150)
        self.chart_canvas.pack(fill=tk.X)

        # Position info
        self.position_label = ttk.Label(right_panel, text="Position: -")
        self.position_label.pack(anchor=tk.W)

        # Bind mouse events
        self.canvas.bind("<Motion>", self.on_canvas_motion)

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

            self.solver_status.config(text="Loading video...")
            self.root.update()

            self.wall = TVWall(filepath, num_frames=num_frames, stride=stride)

            self.video_label.config(text=f"{filepath.split('/')[-1].split(chr(92))[-1]}\n"
                                        f"Size: {self.wall.width}x{self.wall.height}\n"
                                        f"Frames: {self.wall.num_frames}")

            # Precompute original color series
            self.precompute_series()

            # Reset state
            self.ground_truth_positions = {}
            self.solve_history = []
            self.dissonance_map = None

            # Update frame slider
            self.frame_slider.config(to=self.wall.num_frames - 1)

            # Calculate display scale
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 500
            scale_x = canvas_width / self.wall.width
            scale_y = canvas_height / self.wall.height
            self.display_scale = max(1, min(scale_x, scale_y, 8))

            self.update_display()
            self.update_scramble_info()
            self.solver_status.config(text="Video loaded")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

    def precompute_series(self):
        """Precompute all TV color series."""
        if self.wall is None:
            return

        height, width = self.wall.height, self.wall.width
        n_frames = self.wall.num_frames

        self.original_series = np.zeros((height, width, 3, n_frames), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                self.original_series[y, x] = self.wall.get_tv_color_series(x, y).T

        # Current series starts as copy of original
        self.current_series = self.original_series.copy()

    def build_current_series(self):
        """Rebuild current_series from wall's current permutation."""
        if self.wall is None or self.original_series is None:
            return

        height, width = self.wall.height, self.wall.width
        self.current_series = np.zeros_like(self.original_series)

        for y in range(height):
            for x in range(width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                self.current_series[y, x] = self.original_series[orig_y, orig_x]

    def scramble(self):
        """Scramble TV positions."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        try:
            num_swaps = int(self.num_swaps_var.get())
            max_dist = int(self.max_dist_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number")
            return

        # Reset first
        self.wall.reset_swaps()
        self.ground_truth_positions = {}
        self.solve_history = []

        if max_dist == 0:
            # Unlimited distance - random swaps
            self.do_random_swaps(num_swaps)
        else:
            # Limited distance swaps
            self.do_short_swaps(num_swaps, max_dist)

        # Record ground truth (what position each TV should be in)
        for y in range(self.wall.height):
            for x in range(self.wall.width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                if orig_x != x or orig_y != y:
                    self.ground_truth_positions[(x, y)] = (orig_x, orig_y)

        self.build_current_series()
        self.compute_dissonance_map()
        self.update_scramble_info()
        self.update_display()
        self.update_results()

    def do_random_swaps(self, num_swaps):
        """Perform random swaps with no distance limit."""
        all_positions = [(x, y) for y in range(self.wall.height) for x in range(self.wall.width)]
        np.random.shuffle(all_positions)

        for i in range(min(num_swaps, len(all_positions) // 2)):
            pos1 = all_positions[i * 2]
            pos2 = all_positions[i * 2 + 1]
            self.wall.swap_positions(pos1, pos2)

    def do_short_swaps(self, num_swaps, max_dist):
        """Perform swaps within a maximum distance."""
        used = set()
        swaps_made = 0
        attempts = 0

        while swaps_made < num_swaps and attempts < num_swaps * 100:
            attempts += 1
            x1 = np.random.randint(0, self.wall.width)
            y1 = np.random.randint(0, self.wall.height)

            if (x1, y1) in used:
                continue

            # Find candidates within distance
            candidates = []
            for dx in range(-max_dist, max_dist + 1):
                for dy in range(-max_dist, max_dist + 1):
                    if dx == 0 and dy == 0:
                        continue
                    x2, y2 = x1 + dx, y1 + dy
                    if 0 <= x2 < self.wall.width and 0 <= y2 < self.wall.height:
                        if np.sqrt(dx**2 + dy**2) <= max_dist and (x2, y2) not in used:
                            candidates.append((x2, y2))

            if not candidates:
                continue

            x2, y2 = candidates[np.random.randint(len(candidates))]
            self.wall.swap_positions((x1, y1), (x2, y2))
            used.add((x1, y1))
            used.add((x2, y2))
            swaps_made += 1

    def reset(self):
        """Reset to unscrambled state."""
        if self.wall is None:
            return

        self.wall.reset_swaps()
        self.ground_truth_positions = {}
        self.solve_history = []
        self.current_series = self.original_series.copy()
        self.dissonance_map = None

        self.update_scramble_info()
        self.update_display()
        self.results_text.delete(1.0, tk.END)

    def update_scramble_info(self):
        """Update scramble status label."""
        if self.wall is None:
            self.scramble_info.config(text="Status: No video")
            return

        num_wrong = len(self.ground_truth_positions)
        total = self.wall.num_tvs
        pct = 100 * num_wrong / total if total > 0 else 0
        self.scramble_info.config(text=f"Misplaced: {num_wrong}/{total} ({pct:.1f}%)")

    def compute_dissonance_map(self):
        """Compute dissonance for all positions."""
        if self.current_series is None:
            return

        window = float(self.window_var.get())
        kernel_size = int(self.kernel_var.get())
        if kernel_size % 2 == 0:
            kernel_size += 1
        distance_metric = self.distance_metric_var.get()

        self.dissonance_map = self.wall.compute_dissonance_map(
            all_series=self.current_series,
            kernel_size=kernel_size,
            distance_metric=distance_metric,
            window=window
        )

    def count_correct_positions(self):
        """Count how many TVs are in their correct position."""
        if self.wall is None:
            return 0, 0

        correct = 0
        total = self.wall.num_tvs

        for y in range(self.wall.height):
            for x in range(self.wall.width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                if orig_x == x and orig_y == y:
                    correct += 1

        return correct, total

    def toggle_solving(self):
        """Start/stop the solver."""
        if self.solving:
            self.solving = False
            self.solve_btn.config(text="Start Solving")
        else:
            self.solving = True
            self.solve_btn.config(text="Stop")
            self.solve_thread = threading.Thread(target=self.solve_loop)
            self.solve_thread.start()

    def solve_loop(self):
        """Main solving loop running in background thread."""
        max_iter = int(self.max_iter_var.get())
        iteration = 0

        while self.solving and iteration < max_iter:
            improved = self.solve_step()

            # Record history
            total_d = self.dissonance_map.sum() if self.dissonance_map is not None else 0
            correct, total = self.count_correct_positions()
            self.solve_history.append((total_d, correct, total))

            # Update UI
            self.root.after(0, self.update_display)
            self.root.after(0, self.update_results)
            self.root.after(0, self.draw_chart)
            self.root.after(0, lambda i=iteration, m=max_iter: self.progress_var.set(100 * i / m))

            iteration += 1

            if not improved:
                self.root.after(0, lambda: self.solver_status.config(text="Converged (no improvement)"))
                break

            # Animation delay
            time.sleep(int(self.speed_var.get()) / 1000)

        self.solving = False
        self.root.after(0, lambda: self.solve_btn.config(text="Start Solving"))
        self.root.after(0, lambda: self.solver_status.config(
            text=f"Done: {iteration} iterations"))

    def step_once(self):
        """Perform a single solve step."""
        if self.wall is None or self.dissonance_map is None:
            self.compute_dissonance_map()
            if self.dissonance_map is None:
                return

        improved = self.solve_step()

        total_d = self.dissonance_map.sum()
        correct, total = self.count_correct_positions()
        self.solve_history.append((total_d, correct, total))

        self.update_display()
        self.update_results()
        self.draw_chart()

        if not improved:
            self.solver_status.config(text="No improvement found")
        else:
            self.solver_status.config(text="Step completed")

    def solve_step(self):
        """
        Perform one greedy solve step.
        Returns True if an improvement was made.
        """
        if self.dissonance_map is None:
            self.compute_dissonance_map()

        strategy = self.strategy_var.get()
        window = float(self.window_var.get())
        kernel_size = int(self.kernel_var.get())
        if kernel_size % 2 == 0:
            kernel_size += 1
        distance_metric = self.distance_metric_var.get()

        if strategy == "highest_dissonance":
            return self.step_highest_dissonance(window, kernel_size, distance_metric)
        elif strategy == "best_of_topk":
            return self.step_best_of_topk(window, kernel_size, distance_metric)
        elif strategy == "simulated_annealing":
            return self.step_simulated_annealing(window, kernel_size, distance_metric)
        else:
            return self.step_best_of_topk(window, kernel_size, distance_metric)

    def step_highest_dissonance(self, window, kernel_size, distance_metric):
        """Swap the highest dissonance position with its best neighbor."""
        height, width = self.wall.height, self.wall.width

        # Find position with highest dissonance
        flat_idx = np.argmax(self.dissonance_map)
        max_y, max_x = divmod(flat_idx, width)

        # Try swapping with each neighbor and find best improvement
        neighbors = self.wall.get_neighbors(max_x, max_y, kernel_size)
        if not neighbors:
            return False

        current_total = self.dissonance_map.sum()
        best_swap = None
        best_improvement = 0

        for nx, ny in neighbors:
            # Trial swap
            self.wall.swap_positions((max_x, max_y), (nx, ny))
            self.build_current_series()

            # Compute new dissonance for affected region
            affected = set([(max_x, max_y), (nx, ny)])
            affected.update(self.wall.get_neighbors(max_x, max_y, kernel_size))
            affected.update(self.wall.get_neighbors(nx, ny, kernel_size))

            new_total = 0
            for ay in range(height):
                for ax in range(width):
                    if (ax, ay) in affected:
                        new_total += self.wall.compute_position_dissonance(
                            ax, ay, self.current_series, kernel_size, distance_metric, window
                        )
                    else:
                        new_total += self.dissonance_map[ay, ax]

            improvement = current_total - new_total

            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = (nx, ny)

            # Undo swap
            self.wall.swap_positions((max_x, max_y), (nx, ny))

        if best_swap is not None and best_improvement > 0:
            # Commit the best swap
            self.wall.swap_positions((max_x, max_y), best_swap)
            self.build_current_series()
            self.compute_dissonance_map()
            return True

        return False

    def step_best_of_topk(self, window, kernel_size, distance_metric):
        """Find the best swap among top-K highest dissonance positions."""
        height, width = self.wall.height, self.wall.width
        topk = int(self.topk_var.get())

        # Get top-K positions by dissonance
        flat_dissonance = self.dissonance_map.ravel()
        top_indices = np.argsort(flat_dissonance)[-topk:]

        candidates = []
        for idx in top_indices:
            y, x = divmod(idx, width)
            candidates.append((x, y, flat_dissonance[idx]))

        current_total = self.dissonance_map.sum()
        best_swap = None
        best_improvement = 0

        # Try all pairs among candidates
        for i, (x1, y1, d1) in enumerate(candidates):
            for x2, y2, d2 in candidates[i+1:]:
                # Trial swap
                self.wall.swap_positions((x1, y1), (x2, y2))
                self.build_current_series()

                # Compute new dissonance for affected region
                affected = set([(x1, y1), (x2, y2)])
                affected.update(self.wall.get_neighbors(x1, y1, kernel_size))
                affected.update(self.wall.get_neighbors(x2, y2, kernel_size))

                new_total = 0
                for ay in range(height):
                    for ax in range(width):
                        if (ax, ay) in affected:
                            new_total += self.wall.compute_position_dissonance(
                                ax, ay, self.current_series, kernel_size, distance_metric, window
                            )
                        else:
                            new_total += self.dissonance_map[ay, ax]

                improvement = current_total - new_total

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = ((x1, y1), (x2, y2))

                # Undo swap
                self.wall.swap_positions((x1, y1), (x2, y2))

        if best_swap is not None and best_improvement > 0:
            # Commit the best swap
            self.wall.swap_positions(best_swap[0], best_swap[1])
            self.build_current_series()
            self.compute_dissonance_map()
            return True

        return False

    def step_simulated_annealing(self, window, kernel_size, distance_metric):
        """
        Simulated annealing step: accept worse moves probabilistically.
        Temperature decreases over iterations, making bad moves less likely over time.
        """
        height, width = self.wall.height, self.wall.width
        topk = int(self.topk_var.get())

        # Get temperature (decays over iterations)
        base_temp = float(self.temp_var.get())
        cooling = float(self.cooling_var.get())
        iteration = len(self.solve_history)
        temperature = base_temp * (cooling ** iteration)

        if temperature < 0.01:
            # Too cold, fall back to greedy
            return self.step_best_of_topk(window, kernel_size, distance_metric)

        # Get top-K positions by dissonance as candidates
        flat_dissonance = self.dissonance_map.ravel()
        top_indices = np.argsort(flat_dissonance)[-topk:]

        candidates = []
        for idx in top_indices:
            y, x = divmod(idx, width)
            candidates.append((x, y))

        if len(candidates) < 2:
            return False

        # Pick a random pair from candidates
        idx1, idx2 = np.random.choice(len(candidates), size=2, replace=False)
        pos1 = candidates[idx1]
        pos2 = candidates[idx2]

        current_total = self.dissonance_map.sum()

        # Trial swap
        self.wall.swap_positions(pos1, pos2)
        self.build_current_series()

        # Compute new dissonance for affected region
        affected = set([pos1, pos2])
        affected.update(self.wall.get_neighbors(pos1[0], pos1[1], kernel_size))
        affected.update(self.wall.get_neighbors(pos2[0], pos2[1], kernel_size))

        new_total = 0
        for ay in range(height):
            for ax in range(width):
                if (ax, ay) in affected:
                    new_total += self.wall.compute_position_dissonance(
                        ax, ay, self.current_series, kernel_size, distance_metric, window
                    )
                else:
                    new_total += self.dissonance_map[ay, ax]

        delta = new_total - current_total  # positive = worse

        # Accept or reject
        if delta < 0:
            # Improvement - always accept
            self.compute_dissonance_map()
            return True
        else:
            # Worse - accept with probability exp(-delta/T)
            prob = math.exp(-delta / temperature)
            if np.random.random() < prob:
                # Accept worse move
                self.compute_dissonance_map()
                return True
            else:
                # Reject - undo swap
                self.wall.swap_positions(pos1, pos2)
                self.build_current_series()
                return False

    def run_batch(self):
        """Run batch experiment across multiple scramble levels."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        try:
            swap_counts = [int(x.strip()) for x in self.batch_swaps_var.get().split(",")]
            num_trials = int(self.batch_trials_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid batch parameters")
            return

        # Disable batch button during run
        self.batch_btn.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Running batch experiment...\n")

        # Run in background thread
        thread = threading.Thread(target=self._batch_worker, args=(swap_counts, num_trials))
        thread.start()

    def _batch_worker(self, swap_counts, num_trials):
        """Background worker for batch experiment."""
        results = []
        max_iter = int(self.max_iter_var.get())
        max_dist = int(self.max_dist_var.get())

        total_runs = len(swap_counts) * num_trials
        run_idx = 0

        for num_swaps in swap_counts:
            level_results = []

            for trial in range(num_trials):
                run_idx += 1
                self.root.after(0, lambda r=run_idx, t=total_runs:
                    self.solver_status.config(text=f"Batch: {r}/{t}"))
                self.root.after(0, lambda r=run_idx, t=total_runs:
                    self.progress_var.set(100 * r / t))

                # Reset and scramble
                self.wall.reset_swaps()
                self.ground_truth_positions = {}

                if max_dist == 0:
                    self.do_random_swaps(num_swaps)
                else:
                    self.do_short_swaps(num_swaps, max_dist)

                # Record ground truth
                for y in range(self.wall.height):
                    for x in range(self.wall.width):
                        orig_x, orig_y = self.wall.get_original_position(x, y)
                        if orig_x != x or orig_y != y:
                            self.ground_truth_positions[(x, y)] = (orig_x, orig_y)

                self.build_current_series()
                self.compute_dissonance_map()
                self.solve_history = []

                initial_correct, total = self.count_correct_positions()
                initial_d = self.dissonance_map.sum()

                # Run solver (without animation)
                for i in range(max_iter):
                    improved = self.solve_step()
                    if not improved:
                        break

                final_correct, _ = self.count_correct_positions()
                final_d = self.dissonance_map.sum()

                # Record: (initial_wrong, final_wrong, iterations, d_reduction)
                initial_wrong = total - initial_correct
                final_wrong = total - final_correct
                solved_pct = 100 * (initial_wrong - final_wrong) / initial_wrong if initial_wrong > 0 else 100
                level_results.append({
                    'initial_wrong': initial_wrong,
                    'final_wrong': final_wrong,
                    'solved_pct': solved_pct,
                    'iterations': i + 1,
                    'd_reduction': initial_d - final_d
                })

            results.append((num_swaps, level_results))

        # Display results
        self.root.after(0, lambda: self._display_batch_results(results))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.solver_status.config(text="Batch complete"))

    def _display_batch_results(self, results):
        """Display batch experiment results."""
        self.results_text.delete(1.0, tk.END)

        lines = []
        lines.append("=" * 38)
        lines.append("      BATCH EXPERIMENT RESULTS")
        lines.append("=" * 38)
        lines.append(f"Strategy: {self.strategy_var.get()}")
        lines.append("")

        for num_swaps, level_results in results:
            avg_solved = np.mean([r['solved_pct'] for r in level_results])
            avg_final = np.mean([r['final_wrong'] for r in level_results])
            avg_iter = np.mean([r['iterations'] for r in level_results])

            lines.append(f"--- {num_swaps} swaps ---")
            lines.append(f"  Avg solved: {avg_solved:.1f}%")
            lines.append(f"  Avg remaining wrong: {avg_final:.1f}")
            lines.append(f"  Avg iterations: {avg_iter:.1f}")

            # Individual trials
            for i, r in enumerate(level_results):
                lines.append(f"    Trial {i+1}: {r['solved_pct']:.0f}% solved, "
                           f"{r['final_wrong']} wrong")
            lines.append("")

        lines.append("=" * 38)
        self.results_text.insert(tk.END, "\n".join(lines))

        # Update display to show final state
        self.update_display()

    def update_results(self):
        """Update results text."""
        self.results_text.delete(1.0, tk.END)

        if self.wall is None:
            return

        lines = []
        lines.append("=" * 38)
        lines.append("          SOLVER RESULTS")
        lines.append("=" * 38)
        lines.append(f"Strategy: {self.strategy_var.get()}")

        correct, total = self.count_correct_positions()
        pct = 100 * correct / total if total > 0 else 0
        lines.append(f"Correct positions: {correct}/{total} ({pct:.1f}%)")

        if self.dissonance_map is not None:
            total_d = self.dissonance_map.sum()
            mean_d = self.dissonance_map.mean()
            lines.append(f"Total dissonance: {total_d:.1f}")
            lines.append(f"Mean dissonance: {mean_d:.2f}")

        lines.append(f"Iterations: {len(self.solve_history)}")

        # Show temperature for simulated annealing
        if self.strategy_var.get() == "simulated_annealing":
            base_temp = float(self.temp_var.get())
            cooling = float(self.cooling_var.get())
            iteration = len(self.solve_history)
            current_temp = base_temp * (cooling ** iteration)
            lines.append(f"Temperature: {current_temp:.2f}")
        lines.append("")

        if len(self.solve_history) > 0:
            lines.append("--- History ---")
            # Show last 10 entries
            start = max(0, len(self.solve_history) - 10)
            for i, (d, c, t) in enumerate(self.solve_history[start:], start=start):
                pct = 100 * c / t if t > 0 else 0
                lines.append(f"{i:3d}: D={d:8.1f}  Correct={c:4d} ({pct:5.1f}%)")

        lines.append("")
        lines.append("=" * 38)

        self.results_text.insert(tk.END, "\n".join(lines))

    def draw_chart(self):
        """Draw the solve progress chart."""
        self.chart_canvas.delete("all")

        if len(self.solve_history) < 2:
            return

        w = self.chart_canvas.winfo_width() or 400
        h = self.chart_canvas.winfo_height() or 150
        margin = 40

        # Extract data
        dissonances = [d for d, c, t in self.solve_history]
        correctness = [100 * c / t if t > 0 else 0 for d, c, t in self.solve_history]

        n = len(self.solve_history)

        # Scale factors
        x_scale = (w - 2 * margin) / max(1, n - 1)

        d_min, d_max = min(dissonances), max(dissonances)
        if d_max == d_min:
            d_max = d_min + 1
        d_scale = (h - 2 * margin) / (d_max - d_min)

        # Draw axes
        self.chart_canvas.create_line(margin, h - margin, w - margin, h - margin, fill='#555')
        self.chart_canvas.create_line(margin, margin, margin, h - margin, fill='#555')

        # Draw dissonance line (red)
        points = []
        for i, d in enumerate(dissonances):
            x = margin + i * x_scale
            y = h - margin - (d - d_min) * d_scale
            points.extend([x, y])

        if len(points) >= 4:
            self.chart_canvas.create_line(points, fill='#ff6666', width=2)

        # Draw correctness line (green) - scaled to 0-100%
        c_scale = (h - 2 * margin) / 100
        points = []
        for i, c in enumerate(correctness):
            x = margin + i * x_scale
            y = h - margin - c * c_scale
            points.extend([x, y])

        if len(points) >= 4:
            self.chart_canvas.create_line(points, fill='#66ff66', width=2)

        # Labels
        self.chart_canvas.create_text(margin + 5, margin, anchor=tk.NW, fill='#ff6666',
                                     text=f"Dissonance: {dissonances[-1]:.0f}", font=("TkDefaultFont", 8))
        self.chart_canvas.create_text(margin + 5, margin + 12, anchor=tk.NW, fill='#66ff66',
                                     text=f"Correct: {correctness[-1]:.1f}%", font=("TkDefaultFont", 8))
        self.chart_canvas.create_text(w - margin, h - margin + 5, anchor=tk.NE, fill='#888',
                                     text=f"Iter: {n}", font=("TkDefaultFont", 8))

    def update_display(self):
        """Update the canvas display."""
        if self.wall is None:
            return

        mode = self.view_mode.get()

        if mode == "video":
            img = self.wall.get_frame_image(self.current_frame)
        elif mode == "heatmap":
            img = self.get_heatmap_image()
        elif mode == "correctness":
            img = self.get_correctness_image()
        else:
            img = self.wall.get_frame_image(self.current_frame)

        if img is None:
            return

        # Scale image
        new_width = int(self.wall.width * self.display_scale)
        new_height = int(self.wall.height * self.display_scale)
        img = img.resize((new_width, new_height), Image.Resampling.NEAREST)

        # Draw markers
        if self.show_markers_var.get():
            img = self.draw_markers(img)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def get_heatmap_image(self):
        """Create heatmap from dissonance values."""
        if self.dissonance_map is None:
            return Image.new('RGB', (self.wall.width, self.wall.height), (50, 50, 50))

        d = self.dissonance_map
        d_min, d_max = d.min(), d.max()

        if d_max > d_min:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)

        # Colormap: black -> red -> yellow -> white
        rgb = np.zeros((self.wall.height, self.wall.width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(d_norm * 2 * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((d_norm - 0.5) * 2 * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip((d_norm - 0.8) * 5 * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(rgb, 'RGB')

    def get_correctness_image(self):
        """Create image showing correct (green) vs incorrect (red) positions."""
        if self.wall is None:
            return None

        rgb = np.zeros((self.wall.height, self.wall.width, 3), dtype=np.uint8)

        for y in range(self.wall.height):
            for x in range(self.wall.width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                if orig_x == x and orig_y == y:
                    rgb[y, x] = [0, 150, 0]  # Green = correct
                else:
                    rgb[y, x] = [150, 0, 0]  # Red = incorrect

        return Image.fromarray(rgb, 'RGB')

    def draw_markers(self, img):
        """Draw markers on misplaced positions."""
        if not self.ground_truth_positions:
            return img

        img = img.copy()
        draw = ImageDraw.Draw(img)
        scale = self.display_scale

        # Draw circles on positions that should be swapped
        for (x, y), (orig_x, orig_y) in self.ground_truth_positions.items():
            # Check if this position is still wrong
            curr_orig_x, curr_orig_y = self.wall.get_original_position(x, y)
            if curr_orig_x != x or curr_orig_y != y:
                # Still misplaced - draw red circle
                cx = int((x + 0.5) * scale)
                cy = int((y + 0.5) * scale)
                r = max(2, int(scale * 0.3))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline='red', width=1)

        return img

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

        orig_x, orig_y = self.wall.get_original_position(x, y)
        if orig_x != x or orig_y != y:
            info += f" [has TV from ({orig_x}, {orig_y})]"

        if self.dissonance_map is not None:
            info += f" | Dissonance: {self.dissonance_map[y, x]:.1f}"

        self.position_label.config(text=info)


def main():
    root = tk.Tk()
    app = GreedySolverGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
