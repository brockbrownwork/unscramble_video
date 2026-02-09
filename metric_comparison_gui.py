"""
Metric Comparison GUI — Compare flattened Euclidean vs sum-of-per-frame Euclidean distance.

Click a pixel to see the N least dissonant pixels for each metric,
overlaid as semi-transparent grey on the original frame.

Usage:
    python metric_comparison_gui.py
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QSpinBox, QLineEdit, QSlider,
    QFileDialog, QMessageBox, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from tv_wall import TVWall


# ---------------------------------------------------------------------------
# Clickable image panel
# ---------------------------------------------------------------------------

class ClickableImagePanel(QLabel):
    """QLabel that emits pixel coordinates on click and hover.

    Automatically scales the image to fit the available widget size while
    preserving the aspect ratio (nearest-neighbor for crisp pixels).
    """

    pixel_clicked = pyqtSignal(int, int)
    mouse_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet(
            "background-color: #f8d7e3; border: 2px solid #ffb6c1; border-radius: 8px;"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)
        self.display_scale = 1.0
        self.wall_width = 0
        self.wall_height = 0
        self._pil_image = None   # original PIL image for re-scaling on resize
        self._image_data = None  # prevent GC of QImage buffer

    def set_image(self, pil_image, _scale=None):
        """Store the image and scale it to fit the current widget size."""
        self._pil_image = pil_image
        self.wall_width = pil_image.width
        self.wall_height = pil_image.height
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        """Rescale stored image to fit current widget dimensions."""
        if self._pil_image is None:
            return

        # Compute scale to fit within widget, preserving aspect ratio
        w = self.width() - 4   # account for border
        h = self.height() - 4
        if w <= 0 or h <= 0:
            return
        scale_x = w / self._pil_image.width
        scale_y = h / self._pil_image.height
        self.display_scale = min(scale_x, scale_y)

        new_w = max(1, int(self._pil_image.width * self.display_scale))
        new_h = max(1, int(self._pil_image.height * self.display_scale))

        scaled = self._pil_image.resize((new_w, new_h), Image.Resampling.NEAREST)
        rgb = scaled.convert("RGB")
        self._image_data = rgb.tobytes("raw", "RGB")
        qimg = QImage(
            self._image_data, rgb.width, rgb.height,
            rgb.width * 3, QImage.Format_RGB888,
        )
        self.setPixmap(QPixmap.fromImage(qimg))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _to_wall_coords(self, event):
        px = int(event.x() / self.display_scale)
        py = int(event.y() / self.display_scale)
        if 0 <= px < self.wall_width and 0 <= py < self.wall_height:
            return px, py
        return None

    def mousePressEvent(self, event):
        coords = self._to_wall_coords(event)
        if coords:
            self.pixel_clicked.emit(*coords)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        coords = self._to_wall_coords(event)
        if coords:
            self.mouse_moved.emit(*coords)
        super().mouseMoveEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MetricComparisonGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metric Comparison — Flattened vs Sum-of-Per-Frame Euclidean")

        # State
        self.wall = None
        self.all_series = None          # (H, W, 3, T) float32
        self.base_frame_image = None    # PIL Image
        self.clicked_pixel = None       # (x, y)
        self.flat_dists = None          # (H, W) float32
        self.per_frame_dists = None     # (H, W) float32

        self._setup_ui()

    # -----------------------------------------------------------------------
    # UI setup
    # -----------------------------------------------------------------------

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)

        # --- Controls row ---------------------------------------------------
        ctrl_group = QGroupBox("Video")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to video file…")
        ctrl_layout.addWidget(self.path_edit, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_video)
        ctrl_layout.addWidget(browse_btn)

        ctrl_layout.addWidget(QLabel("Frames:"))
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(10, 10000)
        self.frames_spin.setValue(100)
        ctrl_layout.addWidget(self.frames_spin)

        ctrl_layout.addWidget(QLabel("Stride:"))
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 100)
        self.stride_spin.setValue(1)
        ctrl_layout.addWidget(self.stride_spin)

        load_btn = QPushButton("Load Video")
        load_btn.clicked.connect(self._load_video)
        ctrl_layout.addWidget(load_btn)

        root_layout.addWidget(ctrl_group)

        # --- Top-N row -------------------------------------------------------
        topn_group = QGroupBox("Selection")
        topn_layout = QHBoxLayout(topn_group)

        topn_layout.addWidget(QLabel("Top-N least dissonant:"))

        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(1, 100000)
        self.topn_spin.setValue(100)
        self.topn_spin.valueChanged.connect(self._on_topn_changed)
        topn_layout.addWidget(self.topn_spin)

        topn_layout.addStretch(1)

        plot_btn = QPushButton("Plot Avg Distance")
        plot_btn.clicked.connect(self._plot_avg_distance)
        topn_layout.addWidget(plot_btn)

        save_btn = QPushButton("Save PNG")
        save_btn.clicked.connect(self._save_figure)
        topn_layout.addWidget(save_btn)

        root_layout.addWidget(topn_group)

        # --- Frame slider row ------------------------------------------------
        frame_group = QGroupBox("Frame")
        frame_layout = QHBoxLayout(frame_group)

        self.frame_label = QLabel("Frame: 0")
        self.frame_label.setMinimumWidth(80)
        frame_layout.addWidget(self.frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        frame_layout.addWidget(self.frame_slider, stretch=1)

        root_layout.addWidget(frame_group)

        # --- Info bar --------------------------------------------------------
        self.info_label = QLabel("Click a pixel to compare metrics.")
        self.info_label.setStyleSheet(
            "padding: 4px 8px; font-style: italic;"
        )
        root_layout.addWidget(self.info_label)

        # --- Image panels ----------------------------------------------------
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(12)

        # Left panel — Flattened Euclidean
        left_col = QVBoxLayout()
        left_title = QLabel("Flattened Euclidean")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        left_col.addWidget(left_title)

        self.left_panel = ClickableImagePanel()
        self.left_panel.pixel_clicked.connect(self._on_pixel_clicked)
        self.left_panel.mouse_moved.connect(self._on_mouse_hover)
        left_col.addWidget(self.left_panel)

        self.left_count_label = QLabel("")
        self.left_count_label.setAlignment(Qt.AlignCenter)
        left_col.addWidget(self.left_count_label)

        panels_layout.addLayout(left_col)

        # Right panel — Sum-of-Per-Frame Euclidean
        right_col = QVBoxLayout()
        right_title = QLabel("Sum-of-Per-Frame Euclidean")
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        right_col.addWidget(right_title)

        self.right_panel = ClickableImagePanel()
        self.right_panel.pixel_clicked.connect(self._on_pixel_clicked)
        self.right_panel.mouse_moved.connect(self._on_mouse_hover)
        right_col.addWidget(self.right_panel)

        self.right_count_label = QLabel("")
        self.right_count_label.setAlignment(Qt.AlignCenter)
        right_col.addWidget(self.right_count_label)

        panels_layout.addLayout(right_col)

        root_layout.addLayout(panels_layout, stretch=1)

    # -----------------------------------------------------------------------
    # Video loading
    # -----------------------------------------------------------------------

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Video files (*.mkv *.mp4 *.avi *.mov);;All files (*.*)",
        )
        if path:
            self.path_edit.setText(path)

    def _load_video(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Error", "Please select a valid video file.")
            return

        self.info_label.setText("Loading video…")
        QApplication.processEvents()

        self.wall = TVWall(
            path,
            num_frames=self.frames_spin.value(),
            stride=self.stride_spin.value(),
        )
        self.all_series = self.wall.get_all_series(force_cpu=True)  # (H, W, 3, T)
        self.base_frame_image = self.wall.get_frame_image(0)

        # Reset click state
        self.clicked_pixel = None
        self.flat_dists = None
        self.per_frame_dists = None

        # Configure frame slider
        self.frame_slider.setRange(0, self.wall.num_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(True)
        self.frame_label.setText("Frame: 0")

        # Display bare frame in both panels
        self.left_panel.set_image(self.base_frame_image)
        self.right_panel.set_image(self.base_frame_image)
        self.left_count_label.setText("")
        self.right_count_label.setText("")

        total = self.wall.width * self.wall.height
        self.info_label.setText(
            f"Loaded {self.wall.width}×{self.wall.height} · "
            f"{self.wall.num_frames} frames · {total:,} pixels. Click a pixel!"
        )

    # -----------------------------------------------------------------------
    # Click / hover
    # -----------------------------------------------------------------------

    def _on_pixel_clicked(self, px, py):
        if self.all_series is None:
            return

        self.clicked_pixel = (px, py)
        self.info_label.setText(f"Computing distances from ({px}, {py})…")
        QApplication.processEvents()

        # clicked series shape: (3, T)
        clicked_series = self.all_series[py, px]

        # diff shape: (H, W, 3, T)
        diff = self.all_series - clicked_series[np.newaxis, np.newaxis, :, :]
        sq_diff = diff ** 2

        # Metric 1 — Flattened Euclidean
        self.flat_dists = np.sqrt(np.sum(sq_diff, axis=(2, 3)))  # (H, W)

        # Metric 2 — Sum-of-Per-Frame Euclidean
        per_frame_sq = np.sum(sq_diff, axis=2)  # (H, W, T)
        self.per_frame_dists = np.sum(np.sqrt(per_frame_sq), axis=2)  # (H, W)

        del diff, sq_diff, per_frame_sq

        # Adjust top-N max to total pixel count
        total = self.wall.width * self.wall.height
        self.topn_spin.setMaximum(total)

        self.info_label.setText(
            f"Clicked ({px}, {py}) — "
            f"Flat max: {float(self.flat_dists.max()):.0f}, "
            f"Per-frame max: {float(self.per_frame_dists.max()):.0f}"
        )

        self._update_overlay()

    def _on_mouse_hover(self, px, py):
        if self.flat_dists is None or self.per_frame_dists is None:
            return
        fd = self.flat_dists[py, px]
        pfd = self.per_frame_dists[py, px]
        self.info_label.setText(
            f"Hover ({px}, {py}) — "
            f"Flat Euc: {fd:.1f}  |  Sum-Per-Frame Euc: {pfd:.1f}"
        )

    # -----------------------------------------------------------------------
    # Overlay rendering
    # -----------------------------------------------------------------------

    def _get_topn_mask(self, distances, top_n):
        """Return boolean mask (H, W) of the top-N least dissonant pixels."""
        flat = distances.ravel()
        if top_n >= flat.size:
            return np.ones(distances.shape, dtype=bool)
        nth_dist = np.partition(flat, top_n)[top_n]
        return distances <= nth_dist

    def _compute_circle_coverage(self, distances, top_n):
        """Compute inscribed circle coverage for a metric.

        Returns (radius, circle_pixel_count, topn_in_circle, coverage_pct)
        or None if no clicked pixel.
        """
        if self.clicked_pixel is None:
            return None
        cx, cy = self.clicked_pixel
        H, W = distances.shape
        mask = self._get_topn_mask(distances, top_n)

        # Find the furthest top-N pixel from the clicked pixel
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        spatial_dists = np.sqrt((xs.astype(np.float64) - cx) ** 2 +
                                (ys.astype(np.float64) - cy) ** 2)
        radius = float(spatial_dists.max())

        if radius < 0.5:
            return (radius, 1, 1, 100.0)

        # Build distance map from clicked pixel to every pixel in the grid
        yy, xx = np.mgrid[0:H, 0:W]
        dist_from_center = np.sqrt((xx.astype(np.float64) - cx) ** 2 +
                                   (yy.astype(np.float64) - cy) ** 2)
        circle_mask = dist_from_center <= radius

        circle_pixel_count = int(circle_mask.sum())
        topn_in_circle = int((mask & circle_mask).sum())
        coverage_pct = (topn_in_circle / circle_pixel_count * 100.0
                        if circle_pixel_count > 0 else 0.0)

        return (radius, circle_pixel_count, topn_in_circle, coverage_pct)

    def _compute_circle_quality(self, distances, top_n):
        """Combined metric: geometric mean of compactness and fill rate.

        Compactness = 1 - radius / max_diagonal  (0 = spread across image, 1 = tight)
        Fill rate   = coverage_pct / 100          (0 = sparse circle, 1 = solid disc)

        Circle Quality = sqrt(compactness * fill_rate) * 100

        The geometric mean ensures both components must be good for a high
        score — a tiny radius with sparse fill, or a huge radius with dense
        fill, both get penalised.
        """
        info = self._compute_circle_coverage(distances, top_n)
        if info is None:
            return None
        radius, _circle_px, _topn_in, coverage_pct = info
        H, W = distances.shape
        max_diag = np.sqrt((W - 1) ** 2 + (H - 1) ** 2)
        if max_diag == 0:
            return None
        compactness = 1.0 - radius / max_diag
        fill_rate = coverage_pct / 100.0
        # Clamp to avoid negative values from edge cases
        compactness = max(0.0, min(1.0, compactness))
        fill_rate = max(0.0, min(1.0, fill_rate))
        quality = np.sqrt(compactness * fill_rate) * 100.0
        return quality

    def _create_overlay_image(self, distances, top_n):
        """Return PIL Image with semi-transparent grey on the top-N least dissonant pixels."""
        base = np.array(self.base_frame_image).astype(np.float32)  # (H, W, 3)
        mask = self._get_topn_mask(distances, top_n)

        alpha = 0.5
        grey = np.array([200, 200, 200], dtype=np.float32)
        base[mask] = base[mask] * (1.0 - alpha) + grey * alpha

        # Mark clicked pixel with a cyan cross
        if self.clicked_pixel is not None:
            cx, cy = self.clicked_pixel
            for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.wall.width and 0 <= ny < self.wall.height:
                    base[ny, nx] = [0, 255, 255]

        img = Image.fromarray(base.astype(np.uint8), "RGB")

        # Draw inscribed circle outline
        if self.clicked_pixel is not None:
            info = self._compute_circle_coverage(distances, top_n)
            if info is not None:
                radius = info[0]
                if radius >= 0.5:
                    draw = ImageDraw.Draw(img)
                    x0 = cx - radius
                    y0 = cy - radius
                    x1 = cx + radius
                    y1 = cy + radius
                    draw.ellipse([x0, y0, x1, y1], outline=(0, 255, 255), width=1)

        return img

    def _avg_spatial_distance(self, distances, top_n):
        """Average Euclidean pixel distance from clicked pixel to the top-N closest."""
        if self.clicked_pixel is None:
            return 0.0
        cx, cy = self.clicked_pixel
        flat = distances.ravel()
        n = min(top_n, flat.size)
        # Indices of the top-N least dissonant pixels (flat index)
        top_indices = np.argpartition(flat, n)[:n]
        ys, xs = np.unravel_index(top_indices, distances.shape)
        spatial_dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        return float(spatial_dists.mean())

    def _format_label(self, distances, top_n, total):
        """Build the label string including avg distance, circle coverage, and quality."""
        avg = self._avg_spatial_distance(distances, top_n)
        text = f"Top {top_n:,} / {total:,}  |  Avg dist: {avg:.1f} px"

        info = self._compute_circle_coverage(distances, top_n)
        if info is not None:
            radius, circle_px, topn_in, pct = info
            text += (f"  |  Circle r={radius:.1f}: "
                     f"{topn_in:,}/{circle_px:,} = {pct:.1f}%")

        quality = self._compute_circle_quality(distances, top_n)
        if quality is not None:
            text += f"  |  Quality: {quality:.1f}"
        return text

    def _update_overlay(self):
        if self.flat_dists is None:
            return

        top_n = self.topn_spin.value()
        total = self.wall.width * self.wall.height

        left_img = self._create_overlay_image(self.flat_dists, top_n)
        self.left_panel.set_image(left_img)
        self.left_count_label.setText(
            self._format_label(self.flat_dists, top_n, total)
        )

        right_img = self._create_overlay_image(self.per_frame_dists, top_n)
        self.right_panel.set_image(right_img)
        self.right_count_label.setText(
            self._format_label(self.per_frame_dists, top_n, total)
        )

    def _on_topn_changed(self, value):
        self._update_overlay()

    def _on_frame_changed(self, frame_idx):
        if self.wall is None:
            return
        self.frame_label.setText(f"Frame: {frame_idx}")
        self.base_frame_image = self.wall.get_frame_image(frame_idx)
        if self.flat_dists is not None:
            self._update_overlay()
        else:
            self.left_panel.set_image(self.base_frame_image)
            self.right_panel.set_image(self.base_frame_image)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    def _cumulative_avg_spatial_dist(self, distances):
        """Return array where element i = avg spatial dist for top-(i+1) pixels.

        Sorts pixels by dissonance, then computes cumulative mean of their
        spatial distances to the clicked pixel. O(N log N) from the sort.
        """
        cx, cy = self.clicked_pixel
        flat = distances.ravel()
        order = np.argsort(flat)
        ys, xs = np.unravel_index(order, distances.shape)
        spatial = np.sqrt((xs.astype(np.float64) - cx) ** 2 +
                          (ys.astype(np.float64) - cy) ** 2)
        cumavg = np.cumsum(spatial) / np.arange(1, len(spatial) + 1)
        return cumavg

    def _cumulative_circle_quality(self, distances):
        """Return array where element i = circle quality for top-(i+1) pixels.

        For each top-N, computes the geometric mean of compactness and fill rate.
        Vectorised: O(N log N) from the sort, O(N) for the running computations.
        """
        cx, cy = self.clicked_pixel
        H, W = distances.shape
        max_diag = np.sqrt((W - 1) ** 2 + (H - 1) ** 2)
        if max_diag == 0:
            return np.zeros(H * W)

        flat = distances.ravel()
        order = np.argsort(flat)
        ys, xs = np.unravel_index(order, distances.shape)
        spatial = np.sqrt((xs.astype(np.float64) - cx) ** 2 +
                          (ys.astype(np.float64) - cy) ** 2)

        # Running max distance = radius for each top-N
        running_radius = np.maximum.accumulate(spatial)

        # Compactness: 1 - radius / max_diag
        compactness = np.clip(1.0 - running_radius / max_diag, 0.0, 1.0)

        # Fill rate: top_n / (pi * radius^2), clamped to [0, 1]
        ns = np.arange(1, len(spatial) + 1, dtype=np.float64)
        circle_area = np.pi * running_radius ** 2
        # Avoid division by zero for radius=0 (first pixel)
        circle_area = np.maximum(circle_area, 1.0)
        fill_rate = np.clip(ns / circle_area, 0.0, 1.0)

        quality = np.sqrt(compactness * fill_rate) * 100.0
        return quality

    def _plot_avg_distance(self):
        if self.flat_dists is None or self.clicked_pixel is None:
            QMessageBox.warning(self, "Warning", "Click a pixel first.")
            return

        self.info_label.setText("Computing plot…")
        QApplication.processEvents()

        max_n = 10000
        flat_cumavg = self._cumulative_avg_spatial_dist(self.flat_dists)[:max_n]
        pf_cumavg = self._cumulative_avg_spatial_dist(self.per_frame_dists)[:max_n]
        flat_quality = self._cumulative_circle_quality(self.flat_dists)[:max_n]
        pf_quality = self._cumulative_circle_quality(self.per_frame_dists)[:max_n]
        ns = np.arange(1, len(flat_cumavg) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

        # Top: Avg spatial distance
        ax1.plot(ns, flat_cumavg, label="Flattened Euclidean", color="#ff85a2", linewidth=1.5)
        ax1.plot(ns, pf_cumavg, label="Sum-of-Per-Frame Euclidean", color="#8b4563", linewidth=1.5)
        ax1.set_ylabel("Avg Spatial Distance (px)")
        ax1.set_title(
            f"Pixel ({self.clicked_pixel[0]}, {self.clicked_pixel[1]}) — Metric Comparison"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Circle quality
        ax2.plot(ns, flat_quality, label="Flattened Euclidean", color="#ff85a2", linewidth=1.5)
        ax2.plot(ns, pf_quality, label="Sum-of-Per-Frame Euclidean", color="#8b4563", linewidth=1.5)
        ax2.set_xlabel("Top-N")
        ax2.set_ylabel("Circle Quality")
        ax2.set_title("Circle Quality = √(compactness × fill rate) × 100")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

        self.info_label.setText("Plot shown.")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def _save_figure(self):
        if self.flat_dists is None:
            QMessageBox.warning(self, "Warning", "Click a pixel first.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "metric_comparison.png",
            "PNG files (*.png);;All files (*.*)",
        )
        if not filepath:
            return

        top_n = self.topn_spin.value()
        left = self._create_overlay_image(self.flat_dists, top_n)
        right = self._create_overlay_image(self.per_frame_dists, top_n)

        s = 2.0  # fixed scale for saved image
        lw = int(left.width * s)
        lh = int(left.height * s)
        left_s = left.resize((lw, lh), Image.Resampling.NEAREST)
        right_s = right.resize((lw, lh), Image.Resampling.NEAREST)

        gap = 20
        title_h = 30
        combined = Image.new("RGB", (lw * 2 + gap, lh + title_h), (255, 240, 245))
        combined.paste(left_s, (0, title_h))
        combined.paste(right_s, (lw + gap, title_h))

        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((lw // 2, 5), "Flattened Euclidean",
                  fill=(139, 69, 99), anchor="mt", font=font)
        draw.text((lw + gap + lw // 2, 5), "Sum-of-Per-Frame Euclidean",
                  fill=(139, 69, 99), anchor="mt", font=font)

        combined.save(filepath)
        self.info_label.setText(f"Saved: {filepath}")


# ---------------------------------------------------------------------------
# Pink theme (from greedy_solver_gui_pyqt.py)
# ---------------------------------------------------------------------------

def get_pink_stylesheet():
    return """
    QMainWindow { background-color: #fff0f5; }
    QWidget { background-color: #fff0f5; color: #8b4563;
              font-family: 'Segoe UI', Arial, sans-serif; }
    QGroupBox { background-color: #ffe4ec; border: 2px solid #ffb6c1;
                border-radius: 10px; margin-top: 12px; padding-top: 10px;
                font-weight: bold; color: #d63384; }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                       padding: 2px 10px; background-color: #ffb6c1;
                       border-radius: 5px; color: #fff; }
    QPushButton { background-color: #ff85a2; color: white; border: none;
                  border-radius: 8px; padding: 6px 12px; font-weight: bold;
                  min-height: 24px; }
    QPushButton:hover { background-color: #ff6b8a; }
    QPushButton:pressed { background-color: #e55a7b; }
    QPushButton:disabled { background-color: #ddb8c4; color: #f5e6ea; }
    QLineEdit { background-color: #fff; border: 2px solid #ffb6c1;
                border-radius: 6px; padding: 4px 8px; color: #8b4563;
                selection-background-color: #ff85a2; }
    QLineEdit:focus { border: 2px solid #ff85a2; }
    QLabel { background-color: transparent; color: #8b4563; }
    QSpinBox { background-color: #fff; border: 2px solid #ffb6c1;
               border-radius: 6px; padding: 4px 8px; color: #8b4563; }
    QSpinBox:focus { border: 2px solid #ff85a2; }
    QSlider::groove:horizontal { height: 8px; background-color: #ffe4ec;
                                  border-radius: 4px; border: 1px solid #ffb6c1; }
    QSlider::handle:horizontal { background-color: #ff85a2; width: 18px;
                                  height: 18px; margin: -6px 0; border-radius: 9px;
                                  border: 2px solid #fff; }
    QSlider::handle:horizontal:hover { background-color: #ff6b8a; }
    QSlider::sub-page:horizontal { background-color: #ffb6c1; border-radius: 4px; }
    QScrollBar:vertical { background-color: #ffe4ec; width: 12px; border-radius: 6px; }
    QScrollBar::handle:vertical { background-color: #ffb6c1; border-radius: 5px;
                                   min-height: 20px; }
    QScrollBar::handle:vertical:hover { background-color: #ff85a2; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QMessageBox { background-color: #fff0f5; }
    QMessageBox QLabel { color: #8b4563; }
    QMessageBox QPushButton { min-width: 80px; }
    QFileDialog { background-color: #fff0f5; }
    """


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(get_pink_stylesheet())
    window = MetricComparisonGUI()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
