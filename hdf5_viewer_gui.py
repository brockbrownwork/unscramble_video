"""HDF5 video viewer GUI — browse frames and test pixel time-series read speed.

Usage:
    python hdf5_viewer_gui.py videos.h5 --dataset "videos/cab_ride_trimmed"
"""

import argparse
import sys
import time

import h5py
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QGroupBox, QSizePolicy,
)


PINK_STYLE = """
QMainWindow { background-color: #1a1a2e; }
QGroupBox {
    background-color: #16213e;
    border: 2px solid #e94560;
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px;
    padding-top: 20px;
    color: #eee;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #e94560;
}
QLabel { color: #eee; font-size: 13px; }
QSlider::groove:horizontal {
    height: 8px;
    background: #16213e;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #e94560;
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::sub-page:horizontal { background: #e94560; border-radius: 4px; }
QPushButton {
    background-color: #e94560;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover { background-color: #ff6b81; }
QPushButton:pressed { background-color: #c0392b; }
"""


class FrameLabel(QLabel):
    """Clickable label that displays a video frame and crosshair."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 180)
        self._pixmap = None
        self._click_pos = None  # (x, y) in image coords
        self._img_w = 0
        self._img_h = 0
        self.click_callback = None

    def set_frame(self, pixmap, img_w, img_h):
        self._pixmap = pixmap
        self._img_w = img_w
        self._img_h = img_h
        self._update_display()

    def _update_display(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if self._click_pos is not None:
            # Draw crosshair
            painter = QPainter(scaled)
            sx = scaled.width() / self._img_w
            sy = scaled.height() / self._img_h
            cx = int(self._click_pos[0] * sx)
            cy = int(self._click_pos[1] * sy)
            pen = QPen(QColor("#e94560"), 2)
            painter.setPen(pen)
            painter.drawLine(cx - 12, cy, cx + 12, cy)
            painter.drawLine(cx, cy - 12, cx, cy + 12)
            painter.drawEllipse(cx - 8, cy - 8, 16, 16)
            painter.end()
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event):
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        # Map widget coords to image coords
        pm = self.pixmap()
        if pm is None:
            return
        # Pixmap is centered in the label
        lx = (self.width() - pm.width()) // 2
        ly = (self.height() - pm.height()) // 2
        px = event.x() - lx
        py = event.y() - ly
        if 0 <= px < pm.width() and 0 <= py < pm.height():
            ix = int(px * self._img_w / pm.width())
            iy = int(py * self._img_h / pm.height())
            ix = max(0, min(ix, self._img_w - 1))
            iy = max(0, min(iy, self._img_h - 1))
            self._click_pos = (ix, iy)
            self._update_display()
            if self.click_callback:
                self.click_callback(ix, iy)


class HDF5ViewerWindow(QMainWindow):
    def __init__(self, hdf5_path, dataset_path):
        super().__init__()
        self.setWindowTitle("HDF5 Video Viewer")
        self.setMinimumSize(900, 620)
        self.setStyleSheet(PINK_STYLE)

        self.h5file = h5py.File(hdf5_path, "r")
        self.dataset = self.h5file[dataset_path]
        self.T, self.C, self.H, self.W = self.dataset.shape
        self.current_frame = 0
        self.selected_pixel = None

        self._build_ui()
        self._show_frame(0)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)

        # Info bar
        info_box = QGroupBox("Dataset Info")
        info_layout = QHBoxLayout(info_box)
        self.info_label = QLabel(
            f"Shape: ({self.T}, {self.C}, {self.H}, {self.W})  |  "
            f"Chunks: {self.dataset.chunks}  |  "
            f"Dtype: {self.dataset.dtype}"
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_box)

        # Frame display
        frame_box = QGroupBox("Frame Viewer — click a pixel to test time-series read")
        frame_layout = QVBoxLayout(frame_box)

        self.frame_label = FrameLabel()
        self.frame_label.click_callback = self._on_pixel_click
        frame_layout.addWidget(self.frame_label, 1)

        # Slider row
        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.T - 1)
        self.slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self.slider, 1)

        self.frame_num_label = QLabel("0 / 0")
        self.frame_num_label.setFixedWidth(140)
        self.frame_num_label.setAlignment(Qt.AlignCenter)
        slider_row.addWidget(self.frame_num_label)

        frame_layout.addLayout(slider_row)

        # Frame read time
        self.frame_time_label = QLabel("Frame read: —")
        frame_layout.addWidget(self.frame_time_label)

        layout.addWidget(frame_box, 1)

        # Pixel info box
        pixel_box = QGroupBox("Pixel Time-Series")
        pixel_layout = QVBoxLayout(pixel_box)

        self.pixel_info_label = QLabel("Click a pixel on the frame to read its full time-series")
        self.pixel_info_label.setAlignment(Qt.AlignCenter)
        pixel_layout.addWidget(self.pixel_info_label)

        self.pixel_stats_label = QLabel("")
        self.pixel_stats_label.setAlignment(Qt.AlignCenter)
        pixel_layout.addWidget(self.pixel_stats_label)

        # Benchmark button
        btn_row = QHBoxLayout()
        self.bench_btn = QPushButton("Benchmark 100 random pixels")
        self.bench_btn.clicked.connect(self._run_benchmark)
        btn_row.addStretch()
        btn_row.addWidget(self.bench_btn)
        btn_row.addStretch()
        pixel_layout.addLayout(btn_row)

        self.bench_label = QLabel("")
        self.bench_label.setAlignment(Qt.AlignCenter)
        pixel_layout.addWidget(self.bench_label)

        layout.addWidget(pixel_box)

    def _show_frame(self, idx):
        self.current_frame = idx
        t0 = time.perf_counter()
        frame = self.dataset[idx]  # (C, H, W)
        t1 = time.perf_counter()

        # C, H, W -> H, W, C for display
        rgb = frame.transpose(1, 2, 0).copy()
        qimg = QImage(rgb.data, self.W, self.H, 3 * self.W, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.frame_label.set_frame(pixmap, self.W, self.H)

        self.frame_num_label.setText(f"{idx} / {self.T - 1}")
        self.frame_time_label.setText(f"Frame read: {(t1 - t0) * 1000:.1f} ms")

    def _on_slider(self, value):
        self._show_frame(value)

    def _on_pixel_click(self, x, y):
        self.selected_pixel = (x, y)

        t0 = time.perf_counter()
        ts = self.dataset[:, :, y, x]  # (T, C)
        t1 = time.perf_counter()
        read_ms = (t1 - t0) * 1000

        color_now = ts[self.current_frame]
        mean_color = ts.mean(axis=0).astype(int)
        std_color = ts.std(axis=0)

        self.pixel_info_label.setText(
            f"Pixel ({x}, {y})  —  time-series read: {read_ms:.1f} ms  "
            f"({self.T} frames × 3 channels = {ts.nbytes / 1024:.1f} KB)"
        )
        self.pixel_stats_label.setText(
            f"Current RGB: ({color_now[0]}, {color_now[1]}, {color_now[2]})  |  "
            f"Mean: ({mean_color[0]}, {mean_color[1]}, {mean_color[2]})  |  "
            f"Std: ({std_color[0]:.1f}, {std_color[1]:.1f}, {std_color[2]:.1f})"
        )

    def _run_benchmark(self):
        self.bench_btn.setEnabled(False)
        self.bench_label.setText("Running benchmark...")
        QApplication.processEvents()

        rng = np.random.default_rng(42)
        xs = rng.integers(0, self.W, 100)
        ys = rng.integers(0, self.H, 100)

        times = []
        for x, y in zip(xs, ys):
            t0 = time.perf_counter()
            _ = self.dataset[:, :, y, x]
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times = np.array(times)
        self.bench_label.setText(
            f"100 random pixel reads:  "
            f"mean {times.mean():.1f} ms  |  "
            f"median {np.median(times):.1f} ms  |  "
            f"min {times.min():.1f} ms  |  "
            f"max {times.max():.1f} ms  |  "
            f"total {times.sum() / 1000:.2f} s"
        )
        self.bench_btn.setEnabled(True)

    def closeEvent(self, event):
        self.h5file.close()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.slider.setValue(max(0, self.current_frame - 1))
        elif event.key() == Qt.Key_Right:
            self.slider.setValue(min(self.T - 1, self.current_frame + 1))
        elif event.key() == Qt.Key_Home:
            self.slider.setValue(0)
        elif event.key() == Qt.Key_End:
            self.slider.setValue(self.T - 1)
        else:
            super().keyPressEvent(event)


def main():
    parser = argparse.ArgumentParser(description="HDF5 video frame viewer")
    parser.add_argument("hdf5_file", help="Path to HDF5 file")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset path inside HDF5")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = HDF5ViewerWindow(args.hdf5_file, args.dataset)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
