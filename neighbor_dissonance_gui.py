#!/usr/bin/env python
# coding: utf-8
"""
Neighbor Dissonance GUI Experiment

Interactive GUI to test the neighbor dissonance metric:
1. Load a video and display the TV wall
2. Click to swap TV positions
3. Compute and visualize neighbor dissonance heatmap
4. Compare dissonance of swapped vs non-swapped positions
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import threading
from collections import deque

from tv_wall import TVWall

# Import distance functions from aeon
from aeon.distances import dtw_pairwise_distance, euclidean_pairwise_distance


def get_neighbors(x, y, width, height):
    """Get valid neighbor positions for a given (x, y) coordinate."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    return neighbors


def compute_neighbor_dissonance_at_position(wall, x, y, all_series, window=0.1, metric='dtw'):
    """Compute neighbor dissonance for a single position using specified metric."""
    height, width = wall.height, wall.width
    neighbors = get_neighbors(x, y, width, height)

    if not neighbors:
        return 0.0

    center_series = all_series[y, x]

    # Stack center + neighbors
    series_list = [center_series]
    for nx, ny in neighbors:
        series_list.append(all_series[ny, nx])

    stacked = np.array(series_list)  # (n+1, 3, n_frames)

    if metric == 'dtw':
        distances = dtw_pairwise_distance(stacked, window=window)
    else:  # euclidean
        distances = euclidean_pairwise_distance(stacked)

    return distances[0, 1:].mean()


class NeighborDissonanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neighbor Dissonance Experiment")
        self.root.geometry("1400x900")

        self.wall = None
        self.all_series = None  # Precomputed color series
        self.dissonance_dtw = None  # DTW dissonance heatmap
        self.dissonance_euc = None  # Euclidean dissonance heatmap
        self.swapped_positions = set()  # Track which positions have been swapped
        self.swap_pairs = []  # List of (pos1, pos2) swap pairs
        self.comparison_positions = set()  # Random positions for comparison

        self.display_scale = 1
        self.current_frame = 0
        self.selected_position = None

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left_panel.pack_propagate(False)

        # Video loading
        load_frame = ttk.LabelFrame(left_panel, text="Load Video", padding="5")
        load_frame.pack(fill=tk.X, pady=5)

        ttk.Button(load_frame, text="Open Video...", command=self.load_video).pack(fill=tk.X)

        self.video_label = ttk.Label(load_frame, text="No video loaded", wraplength=280)
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

        # Swap controls
        swap_frame = ttk.LabelFrame(left_panel, text="Swap Controls", padding="5")
        swap_frame.pack(fill=tk.X, pady=5)

        ttk.Label(swap_frame, text="Random swaps:").pack(anchor=tk.W)
        self.num_swaps_var = tk.StringVar(value="5")
        ttk.Entry(swap_frame, textvariable=self.num_swaps_var, width=10).pack(anchor=tk.W)

        ttk.Button(swap_frame, text="Random Swap", command=self.random_swap).pack(fill=tk.X, pady=2)
        ttk.Button(swap_frame, text="Reset All Swaps", command=self.reset_swaps).pack(fill=tk.X, pady=2)

        ttk.Separator(swap_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        ttk.Label(swap_frame, text="Comparison positions:").pack(anchor=tk.W)
        self.num_compare_var = tk.StringVar(value="50")
        ttk.Entry(swap_frame, textvariable=self.num_compare_var, width=10).pack(anchor=tk.W)
        ttk.Label(swap_frame, text="(random non-swapped positions\nto compare against)", font=("TkDefaultFont", 8)).pack(anchor=tk.W)

        self.swap_info_label = ttk.Label(swap_frame, text="Swaps: 0", wraplength=280)
        self.swap_info_label.pack(fill=tk.X, pady=5)

        # Dissonance computation
        compute_frame = ttk.LabelFrame(left_panel, text="Dissonance", padding="5")
        compute_frame.pack(fill=tk.X, pady=5)

        ttk.Button(compute_frame, text="Compute Dissonance", command=self.compute_dissonance).pack(fill=tk.X, pady=2)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(compute_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(compute_frame, text="Ready", wraplength=280)
        self.status_label.pack(fill=tk.X, pady=5)

        # Results
        results_frame = ttk.LabelFrame(left_panel, text="Results", padding="5")
        results_frame.pack(fill=tk.X, pady=5)

        self.results_text = tk.Text(results_frame, height=12, width=35, font=("Courier", 9))
        self.results_text.pack(fill=tk.X)

        # View controls
        view_frame = ttk.LabelFrame(left_panel, text="View", padding="5")
        view_frame.pack(fill=tk.X, pady=5)

        self.view_mode = tk.StringVar(value="video")
        ttk.Radiobutton(view_frame, text="Video Frame", variable=self.view_mode,
                       value="video", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="DTW Heatmap", variable=self.view_mode,
                       value="heatmap_dtw", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="Euclidean Heatmap", variable=self.view_mode,
                       value="heatmap_euc", command=self.update_display).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="Side-by-Side", variable=self.view_mode,
                       value="sidebyside", command=self.update_display).pack(anchor=tk.W)

        self.show_swaps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="Show swap markers",
                       variable=self.show_swaps_var, command=self.update_display).pack(anchor=tk.W)

        # Frame slider
        ttk.Label(view_frame, text="Frame:").pack(anchor=tk.W)
        self.frame_slider = ttk.Scale(view_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                                      command=self.on_frame_change)
        self.frame_slider.pack(fill=tk.X)

        # Right panel - canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Canvas with scrollbars
        canvas_frame = ttk.Frame(right_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='#333333')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # Position info label
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

            self.status_label.config(text="Loading video...")
            self.root.update()

            self.wall = TVWall(filepath, num_frames=num_frames, stride=stride)

            self.video_label.config(text=f"{filepath.split('/')[-1]}\n"
                                        f"Size: {self.wall.width}x{self.wall.height}\n"
                                        f"Frames: {self.wall.num_frames}")

            # Precompute color series
            self.precompute_series()

            # Reset state
            self.dissonance_dtw = None
            self.dissonance_euc = None
            self.swapped_positions = set()
            self.swap_pairs = []
            self.comparison_positions = set()
            self.selected_position = None

            # Update frame slider
            self.frame_slider.config(to=self.wall.num_frames - 1)

            # Calculate display scale
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 600
            scale_x = canvas_width / self.wall.width
            scale_y = canvas_height / self.wall.height
            self.display_scale = max(1, min(scale_x, scale_y, 8))

            self.update_display()
            self.update_swap_info()
            self.status_label.config(text="Video loaded")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

    def precompute_series(self):
        """Precompute all TV color series."""
        if self.wall is None:
            return

        height, width = self.wall.height, self.wall.width
        n_frames = self.wall.num_frames

        self.all_series = np.zeros((height, width, 3, n_frames), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                self.all_series[y, x] = self.wall.get_tv_color_series(x, y).T

    def random_swap(self):
        """Perform random swaps."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        try:
            num_swaps = int(self.num_swaps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of swaps")
            return

        # Select random positions to swap
        all_positions = [(x, y) for y in range(self.wall.height) for x in range(self.wall.width)]
        np.random.shuffle(all_positions)

        # Swap in pairs
        num_pairs = min(num_swaps, len(all_positions) // 2)

        for i in range(num_pairs):
            pos1 = all_positions[i * 2]
            pos2 = all_positions[i * 2 + 1]

            self.wall.swap_positions(pos1, pos2)
            self.swapped_positions.add(pos1)
            self.swapped_positions.add(pos2)
            self.swap_pairs.append((pos1, pos2))

        self.update_swap_info()
        self.update_display()

    def reset_swaps(self):
        """Reset all swaps."""
        if self.wall is None:
            return

        self.wall.reset_swaps()
        self.swapped_positions = set()
        self.swap_pairs = []
        self.comparison_positions = set()
        self.dissonance_dtw = None
        self.dissonance_euc = None

        self.update_swap_info()
        self.update_display()
        self.results_text.delete(1.0, tk.END)

    def update_swap_info(self):
        """Update the swap info label."""
        if self.wall is None:
            self.swap_info_label.config(text="Swaps: 0")
            return

        n_swapped = len(self.swapped_positions)
        self.swap_info_label.config(text=f"Swapped positions: {n_swapped}\n"
                                        f"Swap pairs: {len(self.swap_pairs)}")

    def compute_dissonance(self):
        """Compute neighbor dissonance in a background thread."""
        if self.wall is None:
            messagebox.showwarning("Warning", "Load a video first")
            return

        self.status_label.config(text="Computing dissonance...")
        self.progress_var.set(0)

        # Run in background thread
        thread = threading.Thread(target=self._compute_dissonance_worker)
        thread.start()

    def _compute_dissonance_worker(self):
        """Background worker to compute dissonance for selected positions only."""
        try:
            height, width = self.wall.height, self.wall.width
            window = float(self.window_var.get())
            num_compare = int(self.num_compare_var.get())

            # Build series array with current swap configuration
            current_series = np.zeros_like(self.all_series)
            for y in range(height):
                for x in range(width):
                    orig_x, orig_y = self.wall.get_original_position(x, y)
                    current_series[y, x] = self.all_series[orig_y, orig_x]

            # Only compute for swapped positions + random comparison positions
            positions_to_compute = list(self.swapped_positions)

            # Get non-swapped positions for comparison
            all_positions = [(x, y) for y in range(height) for x in range(width)]
            non_swapped = [p for p in all_positions if p not in self.swapped_positions]

            # Sample random comparison positions
            num_compare = min(num_compare, len(non_swapped))
            if num_compare > 0:
                compare_indices = np.random.choice(len(non_swapped), size=num_compare, replace=False)
                self.comparison_positions = set([non_swapped[i] for i in compare_indices])
                positions_to_compute.extend(self.comparison_positions)
            else:
                self.comparison_positions = set()

            # Initialize dissonance arrays with NaN (uncomputed)
            dissonance_dtw = np.full((height, width), np.nan)
            dissonance_euc = np.full((height, width), np.nan)

            total = len(positions_to_compute)
            for i, (x, y) in enumerate(positions_to_compute):
                # Compute both metrics
                dissonance_dtw[y, x] = compute_neighbor_dissonance_at_position(
                    self.wall, x, y, current_series, window, metric='dtw'
                )
                dissonance_euc[y, x] = compute_neighbor_dissonance_at_position(
                    self.wall, x, y, current_series, window, metric='euclidean'
                )

                # Update progress
                progress = (i + 1) / total * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))

            self.dissonance_dtw = dissonance_dtw
            self.dissonance_euc = dissonance_euc

            # Update UI on main thread
            self.root.after(0, self._on_dissonance_complete)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Computation failed: {e}"))
            self.root.after(0, lambda: self.status_label.config(text="Error"))

    def _on_dissonance_complete(self):
        """Called when dissonance computation is complete."""
        self.status_label.config(text="Dissonance computed")
        self.progress_var.set(100)

        # Analyze results
        self.analyze_results()
        self.update_display()

    def analyze_results(self):
        """Analyze and display results for both DTW and Euclidean."""
        if self.dissonance_dtw is None:
            return

        self.results_text.delete(1.0, tk.END)

        results = []
        results.append("=" * 35)
        results.append("  DTW vs EUCLIDEAN COMPARISON")
        results.append("=" * 35)
        results.append(f"Swapped: {len(self.swapped_positions)}")
        results.append(f"Comparison: {len(self.comparison_positions)}")
        results.append("")

        # Analyze both metrics
        for metric_name, dissonance in [("DTW", self.dissonance_dtw), ("EUCLIDEAN", self.dissonance_euc)]:
            swapped_d = []
            comparison_d = []

            for x, y in self.swapped_positions:
                d = dissonance[y, x]
                if not np.isnan(d):
                    swapped_d.append(d)

            for x, y in self.comparison_positions:
                d = dissonance[y, x]
                if not np.isnan(d):
                    comparison_d.append(d)

            swapped_d = np.array(swapped_d)
            comparison_d = np.array(comparison_d)

            results.append(f"--- {metric_name} ---")

            if len(swapped_d) > 0 and len(comparison_d) > 0:
                results.append(f"Swapped mean:  {swapped_d.mean():.1f}")
                results.append(f"Compare mean:  {comparison_d.mean():.1f}")

                if comparison_d.std() > 0:
                    separation = (swapped_d.mean() - comparison_d.mean()) / comparison_d.std()
                    results.append(f"Z-score: {separation:.2f}", )
                    if separation > 2:
                        results.append("  -> STRONG")
                    elif separation > 1:
                        results.append("  -> Moderate")
                    elif separation > 0:
                        results.append("  -> Weak")
                    else:
                        results.append("  -> FAILING")

                # Detection accuracy
                all_computed = list(zip(swapped_d, ['S'] * len(swapped_d))) + \
                              list(zip(comparison_d, ['C'] * len(comparison_d)))
                all_computed.sort(key=lambda x: -x[0])

                n_swapped = len(swapped_d)
                k = n_swapped
                top_k = all_computed[:k]
                hits = sum(1 for d, label in top_k if label == 'S')
                pct = 100 * hits / n_swapped if n_swapped > 0 else 0
                results.append(f"Top-{k} recall: {hits}/{n_swapped} ({pct:.0f}%)")

            results.append("")

        results.append("=" * 35)
        self.results_text.insert(tk.END, "\n".join(results))

    def update_display(self):
        """Update the canvas display."""
        if self.wall is None:
            return

        mode = self.view_mode.get()

        if mode == "video":
            img = self.wall.get_frame_image(self.current_frame)
        elif mode == "heatmap_dtw":
            img = self.get_heatmap_image(self.dissonance_dtw)
        elif mode == "heatmap_euc":
            img = self.get_heatmap_image(self.dissonance_euc)
        elif mode == "sidebyside":
            img = self.get_sidebyside_image()
        else:
            img = self.wall.get_frame_image(self.current_frame)

        if img is None:
            return

        # Scale image
        if mode == "sidebyside":
            new_width = int(self.wall.width * 2 * self.display_scale)
        else:
            new_width = int(self.wall.width * self.display_scale)
        new_height = int(self.wall.height * self.display_scale)
        img = img.resize((new_width, new_height), Image.Resampling.NEAREST)

        # Draw swap markers
        if self.show_swaps_var.get() and self.swapped_positions:
            img = self.draw_swap_markers(img, sidebyside=(mode == "sidebyside"))

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def get_heatmap_image(self, dissonance):
        """Create a heatmap image from dissonance values."""
        if dissonance is None:
            # Return gray image
            return Image.new('RGB', (self.wall.width, self.wall.height), (50, 50, 50))

        # Get valid (non-NaN) values for normalization
        valid_mask = ~np.isnan(dissonance)
        if not valid_mask.any():
            return Image.new('RGB', (self.wall.width, self.wall.height), (50, 50, 50))

        valid_values = dissonance[valid_mask]
        d_min, d_max = valid_values.min(), valid_values.max()

        # Normalize dissonance (NaN stays as NaN)
        if d_max > d_min:
            d_norm = (dissonance - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(dissonance)

        # Create colormap (black -> red -> yellow -> white for computed, dark gray for uncomputed)
        rgb = np.full((self.wall.height, self.wall.width, 3), 50, dtype=np.uint8)  # Dark gray background

        # Only color computed positions
        computed_mask = ~np.isnan(dissonance)

        # R channel: increases first
        r_vals = np.clip(d_norm * 2 * 255, 0, 255)
        rgb[:, :, 0] = np.where(computed_mask, r_vals, 50).astype(np.uint8)

        # G channel: increases second
        g_vals = np.clip((d_norm - 0.5) * 2 * 255, 0, 255)
        rgb[:, :, 1] = np.where(computed_mask, g_vals, 50).astype(np.uint8)

        # B channel: only at very high values
        b_vals = np.clip((d_norm - 0.8) * 5 * 255, 0, 255)
        rgb[:, :, 2] = np.where(computed_mask, b_vals, 50).astype(np.uint8)

        return Image.fromarray(rgb, 'RGB')

    def get_sidebyside_image(self):
        """Create side-by-side DTW and Euclidean heatmaps."""
        dtw_img = self.get_heatmap_image(self.dissonance_dtw)
        euc_img = self.get_heatmap_image(self.dissonance_euc)

        # Combine side by side
        combined = Image.new('RGB', (self.wall.width * 2, self.wall.height), (50, 50, 50))
        combined.paste(dtw_img, (0, 0))
        combined.paste(euc_img, (self.wall.width, 0))

        return combined


    def draw_swap_markers(self, img, sidebyside=False):
        """Draw markers on swapped and comparison positions."""
        from PIL import ImageDraw, ImageFont

        img = img.copy()
        draw = ImageDraw.Draw(img)

        scale = self.display_scale
        offsets = [0] if not sidebyside else [0, self.wall.width * scale]

        for offset in offsets:
            # Draw comparison positions (blue squares)
            for x, y in self.comparison_positions:
                cx = int((x + 0.5) * scale) + int(offset)
                cy = int((y + 0.5) * scale)
                r = max(2, int(scale * 0.3))
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline='#4488ff', width=1)

            # Draw swapped positions (green circles)
            for x, y in self.swapped_positions:
                cx = int((x + 0.5) * scale) + int(offset)
                cy = int((y + 0.5) * scale)
                r = max(2, int(scale * 0.4))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline='lime', width=2)

            # Draw lines connecting swap pairs
            for pos1, pos2 in self.swap_pairs:
                x1 = int((pos1[0] + 0.5) * scale) + int(offset)
                y1 = int((pos1[1] + 0.5) * scale)
                x2 = int((pos2[0] + 0.5) * scale) + int(offset)
                y2 = int((pos2[1] + 0.5) * scale)
                draw.line([x1, y1, x2, y2], fill='cyan', width=1)

        # Add labels for side-by-side view
        if sidebyside:
            try:
                draw.text((5, 5), "DTW", fill='white')
                draw.text((int(self.wall.width * scale) + 5, 5), "Euclidean", fill='white')
            except Exception:
                pass  # Font issues on some systems

        return img

    def on_frame_change(self, value):
        """Handle frame slider change."""
        self.current_frame = int(float(value))
        self.update_display()

    def on_canvas_click(self, event):
        """Handle canvas click for manual swapping."""
        if self.wall is None:
            return

        # Convert canvas coordinates to wall position
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)

        if x < 0 or x >= self.wall.width or y < 0 or y >= self.wall.height:
            return

        if self.selected_position is None:
            # First click - select position
            self.selected_position = (x, y)
            self.status_label.config(text=f"Selected ({x}, {y}). Click another position to swap.")
        else:
            # Second click - perform swap
            pos1 = self.selected_position
            pos2 = (x, y)

            if pos1 != pos2:
                self.wall.swap_positions(pos1, pos2)
                self.swapped_positions.add(pos1)
                self.swapped_positions.add(pos2)
                self.swap_pairs.append((pos1, pos2))

                self.update_swap_info()
                self.update_display()
                self.status_label.config(text=f"Swapped ({pos1[0]}, {pos1[1]}) <-> ({pos2[0]}, {pos2[1]})")
            else:
                self.status_label.config(text="Swap cancelled")

            self.selected_position = None

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

        # Show original position if swapped
        orig_x, orig_y = self.wall.get_original_position(x, y)
        if orig_x != x or orig_y != y:
            info += f" [originally ({orig_x}, {orig_y})]"

        # Show dissonance if computed
        if self.dissonance_dtw is not None and not np.isnan(self.dissonance_dtw[y, x]):
            info += f" | DTW: {self.dissonance_dtw[y, x]:.1f}"
        if self.dissonance_euc is not None and not np.isnan(self.dissonance_euc[y, x]):
            info += f" | Euc: {self.dissonance_euc[y, x]:.1f}"

        self.position_label.config(text=info)


def main():
    root = tk.Tk()
    app = NeighborDissonanceGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
