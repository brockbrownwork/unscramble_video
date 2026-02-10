#!/usr/bin/env python
# coding: utf-8
"""
Animated Pixel Shuffle GUI (PyQt5)

Interactive GUI for creating and previewing pixel shuffle animations.
Wraps the animation logic from animated_shuffle_visual.py with a cute pink theme.
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QGroupBox, QFileDialog,
    QSlider, QProgressBar, QCheckBox, QSizePolicy, QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QFontDatabase

from animated_shuffle_visual import (
    load_video_frames, ease_in_out_cubic, create_swap_animation,
    generate_swap_schedule
)


# =============================================================================
# RENDER WORKER (background thread for GIF export)
# =============================================================================

class RenderWorker(QThread):
    """Renders the full animation to GIF in a background thread."""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(str)       # output path
    error = pyqtSignal(str)

    def __init__(self, frames, swap_pairs, swap_schedule, output_path,
                 pixel_scale, swap_duration, fps, highlight_frames, output_scale):
        super().__init__()
        self.frames = frames
        self.swap_pairs = swap_pairs
        self.swap_schedule = swap_schedule
        self.output_path = output_path
        self.pixel_scale = pixel_scale
        self.swap_duration = swap_duration
        self.fps = fps
        self.highlight_frames = highlight_frames
        self.output_scale = output_scale

    def run(self):
        try:
            create_swap_animation(
                self.frames, self.swap_pairs, self.swap_schedule,
                output_path=self.output_path,
                pixel_scale=self.pixel_scale,
                swap_duration_frames=self.swap_duration,
                fps=self.fps,
                highlight_frames=self.highlight_frames,
                output_scale=self.output_scale
            )
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# LIVE PREVIEW ENGINE
# =============================================================================

class PreviewEngine:
    """
    Renders individual frames on demand for live preview.
    Mirrors the logic of create_swap_animation but frame-by-frame.
    """

    def __init__(self, frames, swap_pairs, swap_schedule,
                 pixel_scale, swap_duration_frames, highlight_frames):
        self.frames = frames
        self.swap_pairs = swap_pairs
        self.swap_schedule = swap_schedule
        self.pixel_scale = pixel_scale
        self.swap_duration_frames = swap_duration_frames
        self.highlight_frames = highlight_frames

        self.num_frames, self.orig_h, self.orig_w, _ = frames.shape
        self.scaled_h = self.orig_h * pixel_scale
        self.scaled_w = self.orig_w * pixel_scale

        # Total output frames
        if swap_pairs and swap_schedule:
            last_swap_start = swap_schedule[-1]
            last_swap_end = last_swap_start + highlight_frames + swap_duration_frames
            self.total_frames = max(self.num_frames, last_swap_end)
        else:
            self.total_frames = self.num_frames

        self.reset()

    def reset(self):
        """Reset to frame 0."""
        self.perm = np.zeros((self.orig_h, self.orig_w, 2), dtype=np.int32)
        for y in range(self.orig_h):
            for x in range(self.orig_w):
                self.perm[y, x] = [y, x]
        self.active_swaps = []
        self.swap_idx = 0
        self.current_frame_idx = -1
        # Cache completed swap state up to a frame
        self._last_rendered = -1

    def render_frame(self, out_frame_idx):
        """Render a single output frame. Must be called sequentially from reset."""
        if out_frame_idx < self._last_rendered:
            # Need to re-render from the start
            self.reset()

        # Fast-forward through any skipped frames (update swap state)
        for fi in range(self._last_rendered + 1, out_frame_idx + 1):
            self._advance_state(fi)

        self._last_rendered = out_frame_idx

        video_frame_idx = min(out_frame_idx, self.num_frames - 1)
        video_frame = self.frames[video_frame_idx]

        output_frame = np.zeros((self.scaled_h, self.scaled_w, 3), dtype=np.uint8)

        # Positions being animated (swapping phase)
        animating_positions = set()
        for swap in self.active_swaps:
            if swap['phase'] == 'swapping':
                animating_positions.add(swap['pos1'])
                animating_positions.add(swap['pos2'])

        ps = self.pixel_scale

        # Draw pixels at current positions (vectorized for speed)
        for y in range(self.orig_h):
            for x in range(self.orig_w):
                if (x, y) in animating_positions:
                    continue
                orig_y, orig_x = self.perm[y, x]
                color = video_frame[orig_y, orig_x]
                output_frame[y * ps:(y + 1) * ps, x * ps:(x + 1) * ps] = color

        # Grid lines
        grid_color = 40
        for y in range(0, self.scaled_h, ps):
            output_frame[y, :] = grid_color
        for x in range(0, self.scaled_w, ps):
            output_frame[:, x] = grid_color

        # Draw swap effects
        for swap in self.active_swaps:
            if swap['phase'] == 'completed':
                continue

            frames_elapsed = out_frame_idx - swap['start_frame']
            x1, y1 = swap['pos1']
            x2, y2 = swap['pos2']

            c1x = x1 * ps + ps // 2
            c1y = y1 * ps + ps // 2
            c2x = x2 * ps + ps // 2
            c2y = y2 * ps + ps // 2

            if swap['phase'] == 'highlight':
                pulse_progress = frames_elapsed / self.highlight_frames
                pulse = 0.5 + 0.5 * np.sin(pulse_progress * np.pi * 3)

                for cx, cy in [(c1x, c1y), (c2x, c2y)]:
                    # Red border around pixel
                    border_thickness = max(2, ps // 3)
                    half = ps // 2
                    border_color = np.array([255, int(50 * pulse), int(50 * pulse)], dtype=np.uint8)

                    for t in range(border_thickness):
                        y_top = max(0, cy - half - t)
                        y_bot = min(self.scaled_h - 1, cy + half + t)
                        x_left = max(0, cx - half - border_thickness)
                        x_right = min(self.scaled_w, cx + half + border_thickness + 1)
                        output_frame[y_top, x_left:x_right] = border_color
                        output_frame[y_bot, x_left:x_right] = border_color

                    for t in range(border_thickness):
                        x_l = max(0, cx - half - t)
                        x_r = min(self.scaled_w - 1, cx + half + t)
                        y_top = max(0, cy - half - border_thickness)
                        y_bot = min(self.scaled_h, cy + half + border_thickness + 1)
                        output_frame[y_top:y_bot, x_l] = border_color
                        output_frame[y_top:y_bot, x_r] = border_color

            elif swap['phase'] == 'swapping':
                swap_elapsed = out_frame_idx - swap['swap_start_frame']
                progress = swap_elapsed / self.swap_duration_frames

                if progress >= 1.0:
                    # Draw final positions
                    for (px, py) in [swap['pos1'], swap['pos2']]:
                        orig_y, orig_x = self.perm[py, px]
                        color = video_frame[orig_y, orig_x]
                        output_frame[py * ps:(py + 1) * ps, px * ps:(px + 1) * ps] = color
                    continue

                t = ease_in_out_cubic(progress)

                orig_y1, orig_x1 = self.perm[y1, x1]
                orig_y2, orig_x2 = self.perm[y2, x2]
                color1 = video_frame[orig_y1, orig_x1]
                color2 = video_frame[orig_y2, orig_x2]

                center1 = np.array([c1x, c1y], dtype=float)
                center2 = np.array([c2x, c2y], dtype=float)

                cur1 = center1 + t * (center2 - center1)
                cur2 = center2 + t * (center1 - center2)

                arc_height = ps * 2 * np.sin(progress * np.pi)
                cur1[1] -= arc_height
                cur2[1] -= arc_height

                half = ps // 2
                for center, color in [(cur1, color1), (cur2, color2)]:
                    icx, icy = int(center[0]), int(center[1])
                    y_s = max(0, icy - half)
                    y_e = min(self.scaled_h, icy + half)
                    x_s = max(0, icx - half)
                    x_e = min(self.scaled_w, icx + half)
                    output_frame[y_s:y_e, x_s:x_e] = color

        return output_frame

    def _advance_state(self, fi):
        """Advance swap state machine for frame fi."""
        # Start new swaps
        while self.swap_idx < len(self.swap_schedule) and self.swap_schedule[self.swap_idx] <= fi:
            pos1, pos2 = self.swap_pairs[self.swap_idx]
            self.active_swaps.append({
                'start_frame': fi,
                'pos1': pos1,
                'pos2': pos2,
                'phase': 'highlight',
            })
            self.swap_idx += 1

        # Update phases
        for swap in self.active_swaps:
            if swap['phase'] == 'completed':
                continue

            frames_elapsed = fi - swap['start_frame']

            if swap['phase'] == 'highlight':
                if frames_elapsed >= self.highlight_frames:
                    swap['phase'] = 'swapping'
                    swap['swap_start_frame'] = fi

            if swap['phase'] == 'swapping':
                swap_elapsed = fi - swap['swap_start_frame']
                progress = swap_elapsed / self.swap_duration_frames
                if progress >= 1.0:
                    x1, y1 = swap['pos1']
                    x2, y2 = swap['pos2']
                    self.perm[y1, x1], self.perm[y2, x2] = (
                        self.perm[y2, x2].copy(), self.perm[y1, x1].copy()
                    )
                    swap['phase'] = 'completed'


# =============================================================================
# MAIN GUI
# =============================================================================

class AnimatedShuffleGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animated Pixel Shuffle")
        self.setMinimumSize(1200, 800)
        self.resize(1200, 800)

        self.frames = None
        self.video_fps = 30
        self.swap_pairs = []
        self.swap_schedule = []
        self.preview_engine = None
        self.render_worker = None
        self.playing = False
        self.current_preview_frame = 0

        self._build_ui()

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._play_tick)

    # ----- UI construction -----

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: controls
        controls = QWidget()
        controls.setMaximumWidth(320)
        controls.setMinimumWidth(280)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setSpacing(8)

        # -- Video group --
        vid_group = QGroupBox("Video")
        vid_lay = QVBoxLayout(vid_group)

        browse_row = QHBoxLayout()
        self.video_path_edit = QLineEdit("cab_ride_trimmed.mkv")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_video)
        browse_row.addWidget(self.video_path_edit, 1)
        browse_row.addWidget(browse_btn)
        vid_lay.addLayout(browse_row)

        self._add_param_row(vid_lay, "Frames:", "frames_edit", "600")
        self._add_param_row(vid_lay, "Start frame:", "start_frame_edit", "30")
        self._add_param_row(vid_lay, "Input scale:", "input_scale_edit", "0.1")

        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self._load_video)
        vid_lay.addWidget(self.load_btn)

        self.video_info_label = QLabel("No video loaded")
        self.video_info_label.setWordWrap(True)
        vid_lay.addWidget(self.video_info_label)

        ctrl_layout.addWidget(vid_group)

        # -- Animation group --
        anim_group = QGroupBox("Animation")
        anim_lay = QVBoxLayout(anim_group)

        self._add_param_row(anim_lay, "Num swaps:", "num_swaps_edit", "10")
        self._add_param_row(anim_lay, "Pixel scale:", "pixel_scale_edit", "12")
        self._add_param_row(anim_lay, "Swap duration:", "swap_duration_edit", "48")
        self._add_param_row(anim_lay, "Highlight frames:", "highlight_frames_edit", "12")
        self._add_param_row(anim_lay, "Seed:", "seed_edit", "420")

        self.overlap_check = QCheckBox("Allow swap overlap")
        anim_lay.addWidget(self.overlap_check)

        self._add_param_row(anim_lay, "Stagger (overlap):", "stagger_edit", "15")

        self.generate_btn = QPushButton("Generate Swaps")
        self.generate_btn.clicked.connect(self._generate_swaps)
        self.generate_btn.setEnabled(False)
        anim_lay.addWidget(self.generate_btn)

        self.swap_info_label = QLabel("No swaps generated")
        self.swap_info_label.setWordWrap(True)
        anim_lay.addWidget(self.swap_info_label)

        ctrl_layout.addWidget(anim_group)

        # -- Export group --
        export_group = QGroupBox("Export")
        export_lay = QVBoxLayout(export_group)

        self._add_param_row(export_lay, "Output FPS:", "fps_edit", "20")
        self._add_param_row(export_lay, "Output scale:", "output_scale_edit", "0.25")

        out_row = QHBoxLayout()
        self.output_path_edit = QLineEdit("shuffle_animation.gif")
        out_browse = QPushButton("...")
        out_browse.setFixedWidth(30)
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_path_edit, 1)
        out_row.addWidget(out_browse)
        export_lay.addLayout(out_row)

        self.export_btn = QPushButton("Export GIF")
        self.export_btn.clicked.connect(self._export_gif)
        self.export_btn.setEnabled(False)
        export_lay.addWidget(self.export_btn)

        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        export_lay.addWidget(self.export_progress)

        self.export_status_label = QLabel("")
        self.export_status_label.setWordWrap(True)
        export_lay.addWidget(self.export_status_label)

        ctrl_layout.addWidget(export_group)
        ctrl_layout.addStretch()

        root.addWidget(controls)

        # Right: preview area
        right = QWidget()
        right_lay = QVBoxLayout(right)

        # Preview label
        preview_title = QLabel("Preview")
        preview_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #d63384;")
        preview_title.setAlignment(Qt.AlignCenter)
        right_lay.addWidget(preview_title)

        # Scroll area for image
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #f8d7e3; border: 2px solid #ffb6c1; border-radius: 8px;")

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Load a video and generate swaps to preview")
        scroll.setWidget(self.preview_label)
        right_lay.addWidget(scroll, 1)

        # Playback controls
        playback_row = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        playback_row.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_play)
        self.stop_btn.setEnabled(False)
        playback_row.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_preview)
        self.reset_btn.setEnabled(False)
        playback_row.addWidget(self.reset_btn)

        right_lay.addLayout(playback_row)

        # Frame slider
        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._slider_changed)
        slider_row.addWidget(self.frame_slider, 1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(80)
        slider_row.addWidget(self.frame_label)

        right_lay.addLayout(slider_row)

        # Playback speed
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(3)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        speed_row.addWidget(self.speed_slider, 1)
        self.speed_label = QLabel("3x")
        self.speed_label.setFixedWidth(30)
        speed_row.addWidget(self.speed_label)
        right_lay.addLayout(speed_row)

        root.addWidget(right, 1)

    def _add_param_row(self, layout, label_text, attr_name, default):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(110)
        edit = QLineEdit(default)
        edit.setFixedWidth(70)
        setattr(self, attr_name, edit)
        row.addWidget(label)
        row.addWidget(edit)
        row.addStretch()
        layout.addLayout(row)

    # ----- Actions -----

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video files (*.mkv *.mp4 *.avi *.mov);;All files (*)"
        )
        if path:
            self.video_path_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GIF", self.output_path_edit.text(),
            "GIF files (*.gif);;All files (*)"
        )
        if path:
            self.output_path_edit.setText(path)

    def _load_video(self):
        video_path = self.video_path_edit.text().strip()
        if not video_path or not os.path.exists(video_path):
            self.video_info_label.setText("File not found!")
            return

        self.load_btn.setEnabled(False)
        self.load_btn.setText("Loading...")
        QApplication.processEvents()

        try:
            num_frames = int(self.frames_edit.text())
            start_frame = int(self.start_frame_edit.text())
            scale = float(self.input_scale_edit.text())

            self.frames, self.video_fps = load_video_frames(
                video_path, num_frames=num_frames,
                start_frame=start_frame, scale=scale
            )

            h, w = self.frames.shape[1:3]
            self.video_info_label.setText(
                f"Loaded {len(self.frames)} frames, {w}x{h} pixels"
            )
            self.generate_btn.setEnabled(True)

            # Show first frame as preview
            self._show_frame_preview(0)

        except Exception as e:
            self.video_info_label.setText(f"Error: {e}")
        finally:
            self.load_btn.setEnabled(True)
            self.load_btn.setText("Load Video")

    def _show_frame_preview(self, frame_idx):
        """Show a raw video frame (before swaps are generated)."""
        if self.frames is None:
            return
        frame_idx = min(frame_idx, len(self.frames) - 1)
        frame = self.frames[frame_idx]
        ps = int(self.pixel_scale_edit.text())
        h, w = frame.shape[:2]
        # Scale up so pixels are visible
        scaled = np.repeat(np.repeat(frame, ps, axis=0), ps, axis=1)
        self._display_numpy(scaled)

    def _generate_swaps(self):
        if self.frames is None:
            return

        num_swaps = int(self.num_swaps_edit.text())
        highlight_frames = int(self.highlight_frames_edit.text())
        swap_duration = int(self.swap_duration_edit.text())
        seed = int(self.seed_edit.text())
        overlap = self.overlap_check.isChecked()
        stagger = int(self.stagger_edit.text())

        h, w = self.frames.shape[1:3]

        self.swap_pairs, self.swap_schedule = generate_swap_schedule(
            len(self.frames), num_swaps, w, h,
            stagger_frames=stagger, seed=seed,
            highlight_frames=highlight_frames,
            swap_duration_frames=swap_duration,
            sequential=not overlap
        )

        pixel_scale = int(self.pixel_scale_edit.text())

        self.preview_engine = PreviewEngine(
            self.frames, self.swap_pairs, self.swap_schedule,
            pixel_scale=pixel_scale,
            swap_duration_frames=swap_duration,
            highlight_frames=highlight_frames
        )

        total = self.preview_engine.total_frames
        self.frame_slider.setEnabled(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(total - 1)
        self.frame_slider.setValue(0)
        self.current_preview_frame = 0

        self.play_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        self.swap_info_label.setText(
            f"{len(self.swap_pairs)} swaps, {total} total frames"
        )

        # Render first frame
        self._render_preview_frame(0)

    def _render_preview_frame(self, idx):
        """Render and display a preview frame."""
        if self.preview_engine is None:
            return
        idx = max(0, min(idx, self.preview_engine.total_frames - 1))
        frame = self.preview_engine.render_frame(idx)
        self._display_numpy(frame)
        self.frame_label.setText(f"{idx} / {self.preview_engine.total_frames - 1}")

    def _display_numpy(self, arr):
        """Display a numpy RGB array on the preview label."""
        h, w, c = arr.shape
        bytes_per_line = w * 3
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pixmap)

    # ----- Playback -----

    def _toggle_play(self):
        if self.playing:
            self._pause_play()
        else:
            self._start_play()

    def _start_play(self):
        if self.preview_engine is None:
            return
        self.playing = True
        self.play_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.frame_slider.setEnabled(False)
        speed = self.speed_slider.value()
        interval = max(5, int(1000 / (20 * speed)))
        self.play_timer.start(interval)

    def _pause_play(self):
        self.playing = False
        self.play_timer.stop()
        self.play_btn.setText("Play")
        self.frame_slider.setEnabled(True)

    def _stop_play(self):
        self._pause_play()
        self.stop_btn.setEnabled(False)

    def _play_tick(self):
        if self.preview_engine is None:
            self._pause_play()
            return

        speed = self.speed_slider.value()
        self.current_preview_frame += speed

        if self.current_preview_frame >= self.preview_engine.total_frames:
            self.current_preview_frame = 0
            self.preview_engine.reset()

        self._render_preview_frame(self.current_preview_frame)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_preview_frame)
        self.frame_slider.blockSignals(False)

    def _slider_changed(self, value):
        if self.playing:
            return
        if self.preview_engine is None:
            return
        # If scrubbing backwards, need to reset engine
        if value < self.current_preview_frame:
            self.preview_engine.reset()
        self.current_preview_frame = value
        self._render_preview_frame(value)

    def _speed_changed(self, value):
        self.speed_label.setText(f"{value}x")
        if self.playing:
            interval = max(5, int(1000 / (20 * value)))
            self.play_timer.setInterval(interval)

    def _reset_preview(self):
        self._pause_play()
        if self.preview_engine:
            self.preview_engine.reset()
            self.current_preview_frame = 0
            self.frame_slider.setValue(0)
            self._render_preview_frame(0)

    # ----- Export -----

    def _export_gif(self):
        if self.frames is None or not self.swap_pairs:
            return

        output_path = self.output_path_edit.text().strip()
        if not output_path:
            return

        pixel_scale = int(self.pixel_scale_edit.text())
        swap_duration = int(self.swap_duration_edit.text())
        fps = int(self.fps_edit.text())
        highlight_frames = int(self.highlight_frames_edit.text())
        output_scale = float(self.output_scale_edit.text())

        self.export_btn.setEnabled(False)
        self.export_btn.setText("Exporting...")
        self.export_status_label.setText("Rendering animation...")

        self.render_worker = RenderWorker(
            self.frames, self.swap_pairs, self.swap_schedule,
            output_path, pixel_scale, swap_duration, fps,
            highlight_frames, output_scale
        )
        self.render_worker.finished.connect(self._export_done)
        self.render_worker.error.connect(self._export_error)
        self.render_worker.start()

    def _export_done(self, path):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("Export GIF")
        self.export_status_label.setText(f"Saved to {path}")

    def _export_error(self, msg):
        self.export_btn.setEnabled(True)
        self.export_btn.setText("Export GIF")
        self.export_status_label.setText(f"Error: {msg}")


# =============================================================================
# PINK THEME (matches greedy_solver_gui_pyqt.py)
# =============================================================================

def get_pink_stylesheet():
    return """
    QMainWindow {
        background-color: #fff0f5;
    }
    QWidget {
        background-color: #fff0f5;
        color: #8b4563;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    QGroupBox {
        background-color: #ffe4ec;
        border: 2px solid #ffb6c1;
        border-radius: 10px;
        margin-top: 12px;
        padding-top: 10px;
        font-weight: bold;
        color: #d63384;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 10px;
        background-color: #ffb6c1;
        border-radius: 5px;
        color: #fff;
    }
    QPushButton {
        background-color: #ff85a2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 6px 12px;
        font-weight: bold;
        min-height: 24px;
    }
    QPushButton:hover {
        background-color: #ff6b8a;
    }
    QPushButton:pressed {
        background-color: #e55a7b;
    }
    QPushButton:disabled {
        background-color: #ddb8c4;
        color: #f5e6ea;
    }
    QLineEdit {
        background-color: #fff;
        border: 2px solid #ffb6c1;
        border-radius: 6px;
        padding: 4px 8px;
        color: #8b4563;
        selection-background-color: #ff85a2;
    }
    QLineEdit:focus {
        border: 2px solid #ff85a2;
    }
    QLabel {
        background-color: transparent;
        color: #8b4563;
    }
    QCheckBox {
        color: #8b4563;
        spacing: 6px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 2px solid #ffb6c1;
        background-color: #fff;
    }
    QCheckBox::indicator:checked {
        background-color: #ff85a2;
        border: 2px solid #ff85a2;
    }
    QScrollArea {
        background-color: #fff0f5;
        border: none;
    }
    QScrollBar:vertical {
        background-color: #ffe4ec;
        width: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background-color: #ffb6c1;
        border-radius: 5px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #ff85a2;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar:horizontal {
        background-color: #ffe4ec;
        height: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:horizontal {
        background-color: #ffb6c1;
        border-radius: 5px;
        min-width: 20px;
    }
    QScrollBar::handle:horizontal:hover {
        background-color: #ff85a2;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    QSlider::groove:horizontal {
        border: 1px solid #ffb6c1;
        height: 6px;
        background: #ffe4ec;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #ff85a2;
        border: none;
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }
    QSlider::handle:horizontal:hover {
        background: #ff6b8a;
    }
    QProgressBar {
        border: 2px solid #ffb6c1;
        border-radius: 6px;
        text-align: center;
        color: #8b4563;
        background-color: #fff;
    }
    QProgressBar::chunk {
        background-color: #ff85a2;
        border-radius: 4px;
    }
    """


def main():
    app = QApplication(sys.argv)

    font_path = os.path.join(os.path.dirname(__file__), "Kablammo-Regular-VariableFont_MORF.ttf")
    if os.path.exists(font_path):
        QFontDatabase.addApplicationFont(font_path)

    app.setStyleSheet(get_pink_stylesheet())

    window = AnimatedShuffleGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
