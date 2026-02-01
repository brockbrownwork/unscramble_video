#!/usr/bin/env python
# coding: utf-8
"""
Greedy Solver GUI (PyQt5 Version)

Interactive GUI to solve scrambled TV walls using a two-phase approach:
1. Identify high-dissonance positions (likely misplaced TVs)
2. Optimize by rearranging only those positions to minimize total dissonance

Features:
- Multiple scrambling methods (pair swaps, short swaps, full scramble)
- Multiple solving strategies (greedy, top_k_best, simulated annealing)
- Real-time visualization with animation
- Metrics display showing progress and accuracy
"""

import sys
import os
import numpy as np
from PIL import Image
import threading
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QGroupBox, QTextEdit,
    QRadioButton, QButtonGroup, QSlider, QProgressBar, QFileDialog,
    QMessageBox, QFrame, QSizePolicy, QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFontDatabase

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tv_wall import TVWall


class WorkerSignals(QObject):
    """Signals for background worker threads."""
    progress = pyqtSignal(float)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    update_display = pyqtSignal()
    update_metrics = pyqtSignal()
    update_scramble_info = pyqtSignal()
    set_status = pyqtSignal(str)


class ImageCanvas(QLabel):
    """Custom QLabel for displaying images with mouse tracking."""

    mouse_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("background-color: #f8d7e3; border: 2px solid #ffb6c1; border-radius: 8px;")
        self.setMinimumSize(400, 300)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event.x(), event.y())
        super().mouseMoveEvent(event)


class MetricsGraphWidget(QWidget):
    """Widget containing live-updating matplotlib graphs for solver metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Title
        title = QLabel("Live Metrics")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #d63384;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Fancy iteration counter
        self.iteration_label = QLabel("Iteration: —")
        self.iteration_label.setAlignment(Qt.AlignCenter)
        self.iteration_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                font-family: 'Kablammo', 'Georgia', 'Times New Roman', serif;
                color: #d63384;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffe4ec, stop:0.5 #fff0f5, stop:1 #ffe4ec);
                border: 2px solid #ffb6c1;
                border-radius: 12px;
                padding: 8px 16px;
                margin: 5px;
            }
        """)
        layout.addWidget(self.iteration_label)

        # Create matplotlib figures with pink theme
        self.fig_dissonance = Figure(figsize=(3, 2.5), dpi=80, facecolor='#fff0f5')
        self.ax_dissonance = self.fig_dissonance.add_subplot(111)
        self.canvas_dissonance = FigureCanvas(self.fig_dissonance)
        self.canvas_dissonance.setMinimumHeight(180)

        self.fig_correct = Figure(figsize=(3, 2.5), dpi=80, facecolor='#fff0f5')
        self.ax_correct = self.fig_correct.add_subplot(111)
        self.canvas_correct = FigureCanvas(self.fig_correct)
        self.canvas_correct.setMinimumHeight(180)

        self.fig_accuracy = Figure(figsize=(3, 2.5), dpi=80, facecolor='#fff0f5')
        self.ax_accuracy = self.fig_accuracy.add_subplot(111)
        self.canvas_accuracy = FigureCanvas(self.fig_accuracy)
        self.canvas_accuracy.setMinimumHeight(180)

        # Style the axes
        for ax in [self.ax_dissonance, self.ax_correct, self.ax_accuracy]:
            ax.set_facecolor('#ffe4ec')
            ax.tick_params(colors='#8b4563', labelsize=8)
            ax.spines['bottom'].set_color('#ffb6c1')
            ax.spines['top'].set_color('#ffb6c1')
            ax.spines['left'].set_color('#ffb6c1')
            ax.spines['right'].set_color('#ffb6c1')

        # Dissonance graph group
        diss_group = QGroupBox("Total Dissonance")
        diss_layout = QVBoxLayout(diss_group)
        diss_layout.setContentsMargins(5, 10, 5, 5)
        diss_layout.addWidget(self.canvas_dissonance)
        layout.addWidget(diss_group)

        # Correct count graph group
        correct_group = QGroupBox("Correct Positions")
        correct_layout = QVBoxLayout(correct_group)
        correct_layout.setContentsMargins(5, 10, 5, 5)
        correct_layout.addWidget(self.canvas_correct)
        layout.addWidget(correct_group)

        # Accuracy percentage graph group
        accuracy_group = QGroupBox("Accuracy %")
        accuracy_layout = QVBoxLayout(accuracy_group)
        accuracy_layout.setContentsMargins(5, 10, 5, 5)
        accuracy_layout.addWidget(self.canvas_accuracy)
        layout.addWidget(accuracy_group)

        layout.addStretch()

        # Initialize empty plots
        self._init_empty_plots()

    def _init_empty_plots(self):
        """Initialize empty plots with placeholder text."""
        for ax, title in [(self.ax_dissonance, 'Total Dissonance'),
                          (self.ax_correct, 'Correct Positions'),
                          (self.ax_accuracy, 'Accuracy %')]:
            ax.clear()
            ax.set_facecolor('#ffe4ec')
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                    transform=ax.transAxes, color='#c97a94', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig_dissonance.tight_layout()
        self.fig_correct.tight_layout()
        self.fig_accuracy.tight_layout()

        self.canvas_dissonance.draw()
        self.canvas_correct.draw()
        self.canvas_accuracy.draw()

    def update_iteration(self, iteration):
        """Update the iteration counter display."""
        if iteration == 0:
            self.iteration_label.setText("Iteration: —")
        else:
            self.iteration_label.setText(f"Iteration: {iteration:,}")

    def update_graphs(self, dissonance_history, correct_history, total_positions):
        """Update all graphs with new data."""
        if not dissonance_history:
            self._init_empty_plots()
            return

        iterations = list(range(len(dissonance_history)))

        # Update dissonance graph
        self.ax_dissonance.clear()
        self.ax_dissonance.set_facecolor('#ffe4ec')
        self.ax_dissonance.plot(iterations, dissonance_history, color='#ff85a2', linewidth=2)
        self.ax_dissonance.fill_between(iterations, dissonance_history, alpha=0.3, color='#ffb6c1')
        self.ax_dissonance.set_xlabel('Iteration', fontsize=8, color='#8b4563')
        self.ax_dissonance.set_ylabel('Dissonance', fontsize=8, color='#8b4563')
        self.ax_dissonance.tick_params(colors='#8b4563', labelsize=7)
        if len(dissonance_history) > 1:
            self.ax_dissonance.set_xlim(0, max(1, len(iterations) - 1))
            # Set y-axis based on min/max with padding
            d_min, d_max = min(dissonance_history), max(dissonance_history)
            d_range = d_max - d_min if d_max > d_min else d_max * 0.1
            padding = d_range * 0.1
            self.ax_dissonance.set_ylim(d_min - padding, d_max + padding)
        self._style_axis(self.ax_dissonance)
        self.fig_dissonance.tight_layout()
        self.canvas_dissonance.draw()

        # Update correct count graph
        if correct_history:
            self.ax_correct.clear()
            self.ax_correct.set_facecolor('#ffe4ec')
            self.ax_correct.plot(iterations[:len(correct_history)], correct_history,
                                color='#4caf50', linewidth=2)
            self.ax_correct.fill_between(iterations[:len(correct_history)], correct_history,
                                         alpha=0.3, color='#81c784')
            self.ax_correct.axhline(y=total_positions, color='#8b4563', linestyle='--',
                                    linewidth=1, alpha=0.5, label='Total')
            self.ax_correct.set_xlabel('Iteration', fontsize=8, color='#8b4563')
            self.ax_correct.set_ylabel('Count', fontsize=8, color='#8b4563')
            self.ax_correct.tick_params(colors='#8b4563', labelsize=7)
            if len(correct_history) > 1:
                self.ax_correct.set_xlim(0, max(1, len(correct_history) - 1))
                # Set y-axis based on min/max with padding
                c_min, c_max = min(correct_history), max(correct_history)
                c_range = c_max - c_min if c_max > c_min else c_max * 0.1
                padding = c_range * 0.1
                self.ax_correct.set_ylim(c_min - padding, c_max + padding)
            self._style_axis(self.ax_correct)
            self.fig_correct.tight_layout()
            self.canvas_correct.draw()

            # Update accuracy percentage graph
            accuracy_history = [c / total_positions * 100 for c in correct_history]
            self.ax_accuracy.clear()
            self.ax_accuracy.set_facecolor('#ffe4ec')
            self.ax_accuracy.plot(iterations[:len(accuracy_history)], accuracy_history,
                                 color='#2196f3', linewidth=2)
            self.ax_accuracy.fill_between(iterations[:len(accuracy_history)], accuracy_history,
                                          alpha=0.3, color='#64b5f6')
            self.ax_accuracy.axhline(y=100, color='#8b4563', linestyle='--',
                                     linewidth=1, alpha=0.5)
            self.ax_accuracy.set_xlabel('Iteration', fontsize=8, color='#8b4563')
            self.ax_accuracy.set_ylabel('%', fontsize=8, color='#8b4563')
            self.ax_accuracy.tick_params(colors='#8b4563', labelsize=7)
            if len(accuracy_history) > 1:
                self.ax_accuracy.set_xlim(0, max(1, len(accuracy_history) - 1))
                # Set y-axis based on min/max with padding
                a_min, a_max = min(accuracy_history), max(accuracy_history)
                a_range = a_max - a_min if a_max > a_min else a_max * 0.1
                padding = a_range * 0.1
                self.ax_accuracy.set_ylim(a_min - padding, min(100, a_max + padding))
            self._style_axis(self.ax_accuracy)
            self.fig_accuracy.tight_layout()
            self.canvas_accuracy.draw()

    def _style_axis(self, ax):
        """Apply pink theme styling to axis."""
        for spine in ax.spines.values():
            spine.set_color('#ffb6c1')
        ax.xaxis.label.set_color('#8b4563')
        ax.yaxis.label.set_color('#8b4563')

    def clear_graphs(self):
        """Clear all graph data."""
        self._init_empty_plots()
        self.iteration_label.setText("Iteration: —")


class GreedySolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Greedy Solver - Unscramble Video")
        self.setGeometry(50, 50, 1800, 950)

        self.wall = None
        self.all_series = None
        self.current_series = None
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
        self.last_swap = None
        self.pending_swap = None
        self.show_pre_swap = False

        # Display settings
        self.display_scale = 1
        self.current_frame = 0

        # Worker signals
        self.signals = WorkerSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_identify_complete)
        self.signals.error.connect(self._on_error)
        self.signals.update_display.connect(self.update_display)
        self.signals.update_metrics.connect(self.update_metrics)
        self.signals.update_scramble_info.connect(self.update_scramble_info)
        self.signals.set_status.connect(self._set_status)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Left panel - controls (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(340)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5)

        # Video loading
        load_group = QGroupBox("Load Video")
        load_layout = QVBoxLayout(load_group)

        self.load_btn = QPushButton("Open Video...")
        self.load_btn.clicked.connect(self.load_video)
        load_layout.addWidget(self.load_btn)

        self.video_label = QLabel("No video loaded")
        self.video_label.setWordWrap(True)
        load_layout.addWidget(self.video_label)

        left_layout.addWidget(load_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)

        params = [
            ("Frames:", "frames_edit", "50"),
            ("Stride:", "stride_edit", "100"),
            ("Crop % (1-100):", "crop_edit", "50"),
            ("DTW Window (0-1):", "window_edit", "0.1"),
            ("Kernel Size (odd):", "kernel_edit", "3"),
        ]

        for label_text, attr_name, default in params:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(120)
            row.addWidget(label)
            edit = QLineEdit(default)
            edit.setFixedWidth(80)
            setattr(self, attr_name, edit)
            row.addWidget(edit)
            row.addStretch()
            param_layout.addLayout(row)

        # Distance metric
        metric_row = QHBoxLayout()
        metric_label = QLabel("Distance Metric:")
        metric_label.setFixedWidth(120)
        metric_row.addWidget(metric_label)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["dtw", "euclidean", "manhattan", "cosine"])
        self.metric_combo.setCurrentText("euclidean")
        self.metric_combo.setFixedWidth(120)
        metric_row.addWidget(self.metric_combo)
        metric_row.addStretch()
        param_layout.addLayout(metric_row)

        left_layout.addWidget(param_group)

        # Scramble controls
        scramble_group = QGroupBox("Scramble")
        scramble_layout = QVBoxLayout(scramble_group)

        # Num swaps
        swap_row = QHBoxLayout()
        swap_label = QLabel("Num swaps:")
        swap_label.setFixedWidth(80)
        swap_row.addWidget(swap_label)
        self.num_swaps_edit = QLineEdit("10000")
        self.num_swaps_edit.setFixedWidth(70)
        swap_row.addWidget(self.num_swaps_edit)
        swap_row.addStretch()
        scramble_layout.addLayout(swap_row)

        # Max distance
        dist_row = QHBoxLayout()
        dist_label = QLabel("Max dist:")
        dist_label.setFixedWidth(80)
        dist_row.addWidget(dist_label)
        self.max_dist_edit = QLineEdit("10")
        self.max_dist_edit.setFixedWidth(70)
        dist_row.addWidget(self.max_dist_edit)
        dist_row.addWidget(QLabel("px"))
        dist_row.addStretch()
        scramble_layout.addLayout(dist_row)

        # Scramble buttons row 1
        btn_row1 = QHBoxLayout()
        self.pair_swap_btn = QPushButton("Pair Swap")
        self.pair_swap_btn.clicked.connect(self.pair_swap)
        btn_row1.addWidget(self.pair_swap_btn)

        self.short_swap_btn = QPushButton("Short Swap")
        self.short_swap_btn.clicked.connect(self.short_swap)
        btn_row1.addWidget(self.short_swap_btn)

        self.shuffle_btn = QPushButton("Shuffle")
        self.shuffle_btn.clicked.connect(self.shuffle)
        btn_row1.addWidget(self.shuffle_btn)
        scramble_layout.addLayout(btn_row1)

        # Scramble buttons row 2
        btn_row2 = QHBoxLayout()
        self.full_scramble_btn = QPushButton("Full Scramble")
        self.full_scramble_btn.clicked.connect(self.full_scramble)
        btn_row2.addWidget(self.full_scramble_btn)
        btn_row2.addStretch()
        scramble_layout.addLayout(btn_row2)

        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.clicked.connect(self.reset_all)
        scramble_layout.addWidget(self.reset_btn)

        self.scramble_info = QLabel("Swapped: 0 positions")
        self.scramble_info.setWordWrap(True)
        scramble_layout.addWidget(self.scramble_info)

        left_layout.addWidget(scramble_group)

        # Solver controls
        solver_group = QGroupBox("Solver")
        solver_layout = QVBoxLayout(solver_group)

        # Strategy
        strat_row = QHBoxLayout()
        strat_label = QLabel("Strategy:")
        strat_label.setFixedWidth(70)
        strat_row.addWidget(strat_label)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["greedy", "top_k_best", "simulated_annealing"])
        self.strategy_combo.setFixedWidth(160)
        strat_row.addWidget(self.strategy_combo)
        strat_row.addStretch()
        solver_layout.addLayout(strat_row)

        # Top-N
        topn_row = QHBoxLayout()
        topn_label = QLabel("Top-N:")
        topn_label.setFixedWidth(70)
        topn_row.addWidget(topn_label)
        self.topn_edit = QLineEdit("2000")
        self.topn_edit.setFixedWidth(70)
        topn_row.addWidget(self.topn_edit)
        topn_hint = QLabel("(# high-diss)")
        topn_hint.setStyleSheet("font-size: 10px; color: #c97a94;")
        topn_row.addWidget(topn_hint)
        topn_row.addStretch()
        solver_layout.addLayout(topn_row)

        # Top-K
        topk_row = QHBoxLayout()
        topk_label = QLabel("Top-K:")
        topk_label.setFixedWidth(70)
        topk_row.addWidget(topk_label)
        self.topk_edit = QLineEdit("100")
        self.topk_edit.setFixedWidth(70)
        topk_row.addWidget(self.topk_edit)
        topk_hint = QLabel("(for strategies)")
        topk_hint.setStyleSheet("font-size: 10px; color: #c97a94;")
        topk_row.addWidget(topk_hint)
        topk_row.addStretch()
        solver_layout.addLayout(topk_row)

        # Max iterations
        iter_row = QHBoxLayout()
        iter_label = QLabel("Max iters:")
        iter_label.setFixedWidth(70)
        iter_row.addWidget(iter_label)
        self.max_iters_edit = QLineEdit("1000")
        self.max_iters_edit.setFixedWidth(70)
        iter_row.addWidget(self.max_iters_edit)
        iter_row.addStretch()
        solver_layout.addLayout(iter_row)

        # Delay
        delay_row = QHBoxLayout()
        delay_label = QLabel("Delay (ms):")
        delay_label.setFixedWidth(70)
        delay_row.addWidget(delay_label)
        self.delay_edit = QLineEdit("0")
        self.delay_edit.setFixedWidth(70)
        delay_row.addWidget(self.delay_edit)
        delay_row.addStretch()
        solver_layout.addLayout(delay_row)

        # SA parameters
        sa_row = QHBoxLayout()
        sa_label = QLabel("SA Temp:")
        sa_label.setFixedWidth(70)
        sa_row.addWidget(sa_label)
        self.temp_edit = QLineEdit("1.0")
        self.temp_edit.setFixedWidth(60)
        sa_row.addWidget(self.temp_edit)
        cool_label = QLabel("Cool:")
        cool_label.setFixedWidth(35)
        sa_row.addWidget(cool_label)
        self.cooling_edit = QLineEdit("0.95")
        self.cooling_edit.setFixedWidth(60)
        sa_row.addWidget(self.cooling_edit)
        sa_row.addStretch()
        solver_layout.addLayout(sa_row)

        # Solver action buttons
        action_row = QHBoxLayout()
        self.identify_btn = QPushButton("1. Identify")
        self.identify_btn.clicked.connect(self.identify_high_dissonance)
        action_row.addWidget(self.identify_btn)

        self.solve_btn = QPushButton("2. Solve")
        self.solve_btn.clicked.connect(self.start_solver)
        action_row.addWidget(self.solve_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        action_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_solver)
        self.stop_btn.setEnabled(False)
        action_row.addWidget(self.stop_btn)
        solver_layout.addLayout(action_row)

        self.step_btn = QPushButton("Step (one iteration)")
        self.step_btn.clicked.connect(self.step_solver)
        solver_layout.addWidget(self.step_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        solver_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        solver_layout.addWidget(self.status_label)

        left_layout.addWidget(solver_group)

        # Metrics
        results_group = QGroupBox("Metrics")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 9pt; background-color: #fff; color: #8b4563;")
        results_layout.addWidget(self.results_text)

        left_layout.addWidget(results_group)

        # View controls
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        self.view_button_group = QButtonGroup(self)
        views = [
            ("Video Frame", "video"),
            ("Heatmap Overlay", "heatmap_overlay"),
            ("Correctness Overlay", "correctness_overlay"),
            ("Heatmap Only", "heatmap"),
        ]

        for i, (text, value) in enumerate(views):
            radio = QRadioButton(text)
            radio.setProperty("view_mode", value)
            radio.toggled.connect(self.on_view_changed)
            self.view_button_group.addButton(radio, i)
            view_layout.addWidget(radio)
            if value == "video":
                radio.setChecked(True)

        self.view_mode = "video"

        # Overlay alpha
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Overlay alpha:"))
        self.overlay_alpha_edit = QLineEdit("0.4")
        self.overlay_alpha_edit.setFixedWidth(70)
        alpha_row.addWidget(self.overlay_alpha_edit)
        alpha_row.addStretch()
        view_layout.addLayout(alpha_row)

        # Frame slider
        view_layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(1)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        view_layout.addWidget(self.frame_slider)

        left_layout.addWidget(view_group)

        left_layout.addStretch()

        scroll_area.setWidget(left_panel)
        main_layout.addWidget(scroll_area)

        # Center panel - canvas
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = ImageCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.mouse_moved.connect(self.on_canvas_motion)
        center_layout.addWidget(self.canvas)

        self.position_label = QLabel("Position: -")
        center_layout.addWidget(self.position_label)

        main_layout.addWidget(center_panel, stretch=1)

        # Right panel - live graphs sidebar
        graphs_scroll = QScrollArea()
        graphs_scroll.setWidgetResizable(True)
        graphs_scroll.setFixedWidth(300)
        graphs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.metrics_graphs = MetricsGraphWidget()
        graphs_scroll.setWidget(self.metrics_graphs)

        main_layout.addWidget(graphs_scroll)

    def on_view_changed(self, checked):
        if checked:
            button = self.sender()
            self.view_mode = button.property("view_mode")
            self.update_display()

    def _on_progress(self, value):
        self.progress_bar.setValue(int(value))

    def _on_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText("Error")

    def _set_status(self, text):
        self.status_label.setText(text)

    def load_video(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video files (*.mkv *.mp4 *.avi *.mov);;All files (*.*)"
        )
        if not filepath:
            return

        try:
            num_frames = int(self.frames_edit.text())
            stride = int(self.stride_edit.text())
            crop_percent = float(self.crop_edit.text())

            self.status_label.setText("Loading video...")
            QApplication.processEvents()

            self.wall = TVWall(filepath, num_frames=num_frames, stride=stride, crop_percent=crop_percent)

            filename = os.path.basename(filepath)
            self.video_label.setText(f"{filename}\n"
                                     f"Size: {self.wall.width}x{self.wall.height}\n"
                                     f"Frames: {self.wall.num_frames}")

            # Precompute color series
            self.precompute_series()

            # Reset state
            self.reset_state()

            # Update frame slider
            self.frame_slider.setMaximum(self.wall.num_frames - 1)

            # Calculate display scale
            canvas_width = self.canvas.width() or 800
            canvas_height = self.canvas.height() or 600
            scale_x = canvas_width / self.wall.width
            scale_y = canvas_height / self.wall.height
            self.display_scale = max(1, min(scale_x, scale_y, 10))

            self.update_display()
            self.status_label.setText("Video loaded")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {e}")

    def precompute_series(self):
        if self.wall is None:
            return

        height, width = self.wall.height, self.wall.width
        n_frames = self.wall.num_frames

        self.all_series = np.zeros((height, width, 3, n_frames), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                self.all_series[y, x] = self.wall.get_tv_color_series(x, y).T

    def reset_state(self):
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
        self.last_swap = None
        self.pending_swap = None
        self.show_pre_swap = False
        self.update_scramble_info()
        self.results_text.clear()
        self.metrics_graphs.clear_graphs()

    def update_scramble_info(self):
        if self.wall is None:
            self.scramble_info.setText("Swapped: 0 positions")
            return

        n_swapped = len(self.swapped_positions)
        n_pairs = len(self.swap_pairs)
        correct = self.count_correct_positions()
        total = self.wall.num_tvs
        self.scramble_info.setText(f"Swapped: {n_swapped} | Pairs: {n_pairs} | Correct: {correct}/{total}")

    def count_correct_positions(self):
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
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return

        try:
            num_swaps = int(self.num_swaps_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid number of swaps")
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
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return

        try:
            num_swaps = int(self.num_swaps_edit.text())
            max_dist = int(self.max_dist_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid number")
            return

        swap_pairs = self.wall.short_swaps(num_swaps, max_dist)
        for pos1, pos2 in swap_pairs:
            self.swapped_positions.add(pos1)
            self.swapped_positions.add(pos2)
            self.swap_pairs.append((pos1, pos2))

        if len(swap_pairs) < num_swaps:
            QMessageBox.information(self, "Info", f"Made {len(swap_pairs)}/{num_swaps} swaps")

        self.dissonance_map = None
        self.high_dissonance_positions = set()
        self.update_scramble_info()
        self.update_display()

    def shuffle(self):
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return

        try:
            num_positions = int(self.num_swaps_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid number of positions")
            return

        self.wall.reset_swaps()
        self.wall.random_swaps(num_positions)

        self.swapped_positions = set()
        for y in range(self.wall.height):
            for x in range(self.wall.width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                if orig_x != x or orig_y != y:
                    self.swapped_positions.add((x, y))

        self.swap_pairs = []
        self.dissonance_map = None
        self.high_dissonance_positions = set()
        self.update_scramble_info()
        self.update_display()

    def full_scramble(self):
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return

        self.wall.scramble()
        self.swapped_positions = set((x, y) for y in range(self.wall.height)
                                      for x in range(self.wall.width))
        self.swap_pairs = []

        self.dissonance_map = None
        self.high_dissonance_positions = set()
        self.update_scramble_info()
        self.update_display()

    def reset_all(self):
        if self.wall is None:
            return

        self.stop_solver()
        self.wall.reset_swaps()
        self.reset_state()
        self.update_display()

    def get_current_series(self):
        if self.wall is None:
            return None
        return self.wall.get_all_series()

    def identify_high_dissonance(self):
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return

        if len(self.swapped_positions) == 0:
            QMessageBox.information(self, "Info", "No swapped positions to identify")
            return

        self.status_label.setText("Computing dissonance map...")
        QApplication.processEvents()

        thread = threading.Thread(target=self._identify_worker)
        thread.start()

    def _identify_worker(self):
        try:
            kernel_size = int(self.kernel_edit.text())
            if kernel_size < 3:
                kernel_size = 3
            if kernel_size % 2 == 0:
                kernel_size += 1

            window = float(self.window_edit.text())
            metric = self.metric_combo.currentText()

            self.current_series = self.get_current_series()

            height, width = self.wall.height, self.wall.width
            self.dissonance_map = np.zeros((height, width))

            total = height * width
            for idx, (y, x) in enumerate([(y, x) for y in range(height) for x in range(width)]):
                self.dissonance_map[y, x] = self.wall.compute_position_dissonance(
                    x, y, self.current_series, kernel_size, metric, window
                )
                if idx % 100 == 0:
                    progress = (idx + 1) / total * 100
                    self.signals.progress.emit(progress)

            self.high_dissonance_positions = self._get_top_n_dissonance()
            self.signals.finished.emit()

        except Exception as e:
            self.signals.error.emit(f"Identification failed: {e}")

    def _get_top_n_dissonance(self):
        if self.dissonance_map is None:
            return set()

        try:
            top_n = int(self.topn_edit.text())
        except ValueError:
            top_n = 20

        height, width = self.wall.height, self.wall.width
        values_with_pos = []
        for y in range(height):
            for x in range(width):
                values_with_pos.append((self.dissonance_map[y, x], (x, y)))

        values_with_pos.sort(key=lambda x: x[0], reverse=True)

        top_n = min(top_n, len(values_with_pos))
        high_positions = set()
        for i in range(top_n):
            _, pos = values_with_pos[i]
            high_positions.add(pos)

        return high_positions

    def _on_identify_complete(self):
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Identified {len(self.high_dissonance_positions)} high-dissonance positions")
        self.update_metrics()
        self.update_display()

    def update_metrics(self):
        self.results_text.clear()

        if self.wall is None:
            return

        lines = []
        lines.append("=" * 38)
        lines.append("         SOLVER METRICS")
        lines.append("=" * 38)

        total = self.wall.num_tvs
        correct = self.count_correct_positions()
        incorrect = total - correct
        lines.append(f"Total positions: {total}")
        lines.append(f"Currently correct: {correct} ({100*correct/total:.1f}%)")
        lines.append(f"Currently wrong: {incorrect}")
        lines.append("")

        n_swapped = len(self.swapped_positions)
        lines.append(f"Originally swapped: {n_swapped}")

        n_high = len(self.high_dissonance_positions)
        lines.append(f"Detected high-diss: {n_high}")

        if n_swapped > 0 and n_high > 0:
            true_positives = len(self.high_dissonance_positions & self.swapped_positions)
            precision = true_positives / n_high if n_high > 0 else 0
            recall = true_positives / n_swapped if n_swapped > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            lines.append(f"True positives: {true_positives}")
            lines.append(f"Precision: {precision:.2f}")
            lines.append(f"Recall: {recall:.2f}")
            lines.append(f"F1 Score: {f1:.2f}")

        lines.append("")

        if self.dissonance_map is not None:
            d_mean = self.dissonance_map.mean()
            d_std = self.dissonance_map.std()
            d_min = self.dissonance_map.min()
            d_max = self.dissonance_map.max()
            lines.append(f"Dissonance mean: {d_mean:.1f}")
            lines.append(f"Dissonance std: {d_std:.1f}")
            lines.append(f"Dissonance range: [{d_min:.1f}, {d_max:.1f}]")

            if self.high_dissonance_positions:
                high_vals = [self.dissonance_map[y, x] for x, y in self.high_dissonance_positions]
                low_vals = [self.dissonance_map[y, x] for y in range(self.wall.height)
                           for x in range(self.wall.width)
                           if (x, y) not in self.high_dissonance_positions]

                if high_vals and low_vals:
                    lines.append(f"High-diss mean: {np.mean(high_vals):.1f}")
                    lines.append(f"Low-diss mean: {np.mean(low_vals):.1f}")

        lines.append("")

        if self.iteration > 0:
            lines.append("-" * 38)
            lines.append(f"Solver iteration: {self.iteration}")
            if self.total_dissonance_history:
                lines.append(f"Initial total diss: {self.total_dissonance_history[0]:.1f}")
                lines.append(f"Current total diss: {self.total_dissonance_history[-1]:.1f}")
                improvement = self.total_dissonance_history[0] - self.total_dissonance_history[-1]
                lines.append(f"Improvement: {improvement:.1f}")

        lines.append("=" * 38)
        self.results_text.setPlainText("\n".join(lines))

        # Update live graphs and iteration counter
        if self.wall is not None:
            self.metrics_graphs.update_iteration(self.iteration)
            self.metrics_graphs.update_graphs(
                self.total_dissonance_history,
                self.correct_count_history,
                self.wall.num_tvs
            )

    def start_solver(self):
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return

        if len(self.high_dissonance_positions) == 0:
            QMessageBox.information(self, "Info", "Run 'Identify' first to detect high-dissonance positions")
            return

        if self.solver_running:
            return

        self.solver_running = True
        self.solver_paused = False
        self.solve_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)

        self.solver_thread = threading.Thread(target=self._solver_loop)
        self.solver_thread.start()

    def _solver_loop(self):
        try:
            max_iters = int(self.max_iters_edit.text())
            delay_ms = int(self.delay_edit.text())
            strategy = self.strategy_combo.currentText()

            if not self.total_dissonance_history:
                total_d = self._compute_high_diss_total()
                self.total_dissonance_history.append(total_d)
                correct = self.count_correct_positions()
                self.correct_count_history.append(correct)

            while self.solver_running and self.iteration < max_iters:
                while self.solver_paused and self.solver_running:
                    time.sleep(0.1)

                if not self.solver_running:
                    break

                best_swap = self._find_best_swap(strategy)

                if best_swap is not None:
                    self.pending_swap = best_swap
                    self.last_swap = best_swap
                    self.show_pre_swap = True
                    self.signals.update_display.emit()

                    time.sleep(0.5)

                    if not self.solver_running:
                        break

                    self._execute_swap(best_swap, strategy)
                    self.show_pre_swap = False
                    improved = True
                else:
                    self.last_swap = None
                    self.pending_swap = None
                    improved = False

                self.iteration += 1

                total_d = self._compute_high_diss_total()
                self.total_dissonance_history.append(total_d)
                correct = self.count_correct_positions()
                self.correct_count_history.append(correct)

                progress = (self.iteration / max_iters) * 100
                self.signals.progress.emit(progress)
                self.signals.update_metrics.emit()
                self.signals.update_display.emit()
                self.signals.update_scramble_info.emit()

                if not improved and strategy == "greedy":
                    self.signals.set_status.emit("Converged (no improvement)")
                    break

                remaining_delay = max(0, delay_ms / 1000.0 - 0.5)
                if remaining_delay > 0:
                    time.sleep(remaining_delay)

            self.signals.set_status.emit(f"Solver stopped at iteration {self.iteration}")

        except Exception as e:
            self.signals.error.emit(f"Solver failed: {e}")
        finally:
            self.solver_running = False
            QTimer.singleShot(0, self._on_solver_done)

    def _find_best_swap(self, strategy):
        if strategy == "greedy":
            return self._find_greedy_swap()
        elif strategy == "top_k_best":
            return self._find_top_k_swap()
        elif strategy == "simulated_annealing":
            return self._find_sa_swap()
        return None

    def _execute_swap(self, swap, strategy):
        if swap is None:
            return
        pos1, pos2 = swap
        self.wall.swap_positions(pos1, pos2)
        self.current_series = self.get_current_series()
        self.last_swap = swap

        if strategy == "simulated_annealing":
            temp = float(self.temp_edit.text())
            cooling = float(self.cooling_edit.text())
            self.temp_edit.setText(f"{temp * cooling:.4f}")

    def _find_greedy_swap(self):
        if not self.high_dissonance_positions:
            return None

        kernel_size = int(self.kernel_edit.text())
        window = float(self.window_edit.text())
        metric = self.metric_combo.currentText()

        self.current_series = self.get_current_series()

        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        best_pos = max(self.high_dissonance_positions, key=lambda p: all_diss[p])
        x1, y1 = best_pos
        diss_best_before = all_diss[best_pos]

        best_swap = None
        best_improvement = 0

        series_copy = self.current_series.copy()

        for other_pos in self.high_dissonance_positions:
            if other_pos == best_pos:
                continue

            x2, y2 = other_pos
            diss_other_before = all_diss[other_pos]

            series_copy[y1, x1], series_copy[y2, x2] = \
                series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

            diss_best_after = self.wall.compute_position_dissonance(
                x1, y1, series_copy, kernel_size, metric, window)
            diss_other_after = self.wall.compute_position_dissonance(
                x2, y2, series_copy, kernel_size, metric, window)

            improvement = (diss_best_before + diss_other_before) - (diss_best_after + diss_other_after)

            if improvement > best_improvement:
                best_improvement = improvement
                best_swap = other_pos

            series_copy[y1, x1], series_copy[y2, x2] = \
                series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

        if best_swap is not None and best_improvement > 0:
            return (best_pos, best_swap)
        return None

    def _find_top_k_swap(self):
        if not self.high_dissonance_positions:
            return None

        topk = int(self.topk_edit.text())
        kernel_size = int(self.kernel_edit.text())
        window = float(self.window_edit.text())
        metric = self.metric_combo.currentText()

        self.current_series = self.get_current_series()

        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        diss_list = [(all_diss[pos], pos) for pos in self.high_dissonance_positions]
        diss_list.sort(reverse=True)
        top_k_with_diss = diss_list[:topk]
        top_positions = [pos for _, pos in top_k_with_diss]
        diss_before = {pos: d for d, pos in top_k_with_diss}

        if len(top_positions) < 2:
            return None

        best_swap = None
        best_improvement = 0

        series_copy = self.current_series.copy()
        n_positions = len(top_positions)

        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                pos1 = top_positions[i]
                pos2 = top_positions[j]
                x1, y1 = pos1
                x2, y2 = pos2

                diss1_before = diss_before[pos1]
                diss2_before = diss_before[pos2]

                series_copy[y1, x1], series_copy[y2, x2] = \
                    series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

                diss1_after = self.wall.compute_position_dissonance(
                    x1, y1, series_copy, kernel_size, metric, window)
                diss2_after = self.wall.compute_position_dissonance(
                    x2, y2, series_copy, kernel_size, metric, window)

                improvement = (diss1_before + diss2_before) - (diss1_after + diss2_after)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (pos1, pos2)

                series_copy[y1, x1], series_copy[y2, x2] = \
                    series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

        if best_swap is not None and best_improvement > 0:
            return best_swap
        return None

    def _find_sa_swap(self):
        if not self.high_dissonance_positions:
            return None

        topk = int(self.topk_edit.text())
        temp = float(self.temp_edit.text())
        kernel_size = int(self.kernel_edit.text())
        window = float(self.window_edit.text())
        metric = self.metric_combo.currentText()

        self.current_series = self.get_current_series()

        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        diss_list = [(all_diss[pos], pos) for pos in self.high_dissonance_positions]
        diss_list.sort(reverse=True)
        top_k_with_diss = diss_list[:topk]
        top_positions = [pos for _, pos in top_k_with_diss]
        diss_before = {pos: d for d, pos in top_k_with_diss}

        if len(top_positions) < 2:
            return None

        idx1, idx2 = np.random.choice(len(top_positions), size=2, replace=False)
        pos1, pos2 = top_positions[idx1], top_positions[idx2]
        x1, y1 = pos1
        x2, y2 = pos2

        diss1_before = diss_before[pos1]
        diss2_before = diss_before[pos2]

        series_copy = self.current_series.copy()
        series_copy[y1, x1], series_copy[y2, x2] = \
            series_copy[y2, x2].copy(), series_copy[y1, x1].copy()

        diss1_after = self.wall.compute_position_dissonance(
            x1, y1, series_copy, kernel_size, metric, window)
        diss2_after = self.wall.compute_position_dissonance(
            x2, y2, series_copy, kernel_size, metric, window)

        delta = (diss1_after + diss2_after) - (diss1_before + diss2_before)

        accept = False
        if delta < 0:
            accept = True
        else:
            prob = np.exp(-delta / temp) if temp > 0 else 0
            if np.random.random() < prob:
                accept = True

        if accept:
            return (pos1, pos2)
        return None

    def _compute_high_diss_total(self):
        if not self.high_dissonance_positions:
            return 0

        kernel_size = int(self.kernel_edit.text())
        window = float(self.window_edit.text())
        metric = self.metric_combo.currentText()

        if self.current_series is None:
            self.current_series = self.get_current_series()

        all_diss = self.wall.compute_batch_dissonance(
            list(self.high_dissonance_positions),
            self.current_series, kernel_size, metric, window
        )

        return sum(all_diss.values())

    def step_solver(self):
        if self.wall is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return

        if len(self.high_dissonance_positions) == 0:
            QMessageBox.information(self, "Info", "Run 'Identify' first")
            return

        strategy = self.strategy_combo.currentText()

        if not self.total_dissonance_history:
            self.current_series = self.get_current_series()
            total_d = self._compute_high_diss_total()
            self.total_dissonance_history.append(total_d)
            correct = self.count_correct_positions()
            self.correct_count_history.append(correct)

        best_swap = self._find_best_swap(strategy)

        if best_swap is not None:
            self.pending_swap = best_swap
            self.last_swap = best_swap
            self.show_pre_swap = True
            self.update_display()
            QApplication.processEvents()

            QTimer.singleShot(100, lambda: self._complete_step(best_swap, strategy))
        else:
            self.last_swap = None
            self.pending_swap = None
            self.iteration += 1
            self.status_label.setText(f"Step {self.iteration}: No improvement")
            self.update_metrics()
            self.update_display()
            self.update_scramble_info()

    def _complete_step(self, swap, strategy):
        self._execute_swap(swap, strategy)
        self.show_pre_swap = False
        self.iteration += 1

        total_d = self._compute_high_diss_total()
        self.total_dissonance_history.append(total_d)
        correct = self.count_correct_positions()
        self.correct_count_history.append(correct)

        self.status_label.setText(f"Step {self.iteration}: Improved")

        self.update_metrics()
        self.update_display()
        self.update_scramble_info()

    def toggle_pause(self):
        if not self.solver_running:
            return

        self.solver_paused = not self.solver_paused
        if self.solver_paused:
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused")
        else:
            self.pause_btn.setText("Pause")
            self.status_label.setText("Running...")

    def stop_solver(self):
        self.solver_running = False
        self.solver_paused = False
        if self.solver_thread and self.solver_thread.is_alive():
            self.solver_thread.join(timeout=1.0)

        self._on_solver_done()

    def _on_solver_done(self):
        self.solver_running = False
        self.solver_paused = False
        self.solve_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)

    def update_display(self):
        if self.wall is None:
            return

        mode = self.view_mode

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

        # Convert PIL to QPixmap
        if img.mode == 'RGB':
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
        else:
            img = img.convert('RGB')
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)

        # Draw swap line if needed
        if self.last_swap is not None:
            pixmap = self._draw_swap_line(pixmap)

        self.canvas.setPixmap(pixmap)

    def _draw_swap_line(self, pixmap):
        if self.last_swap is None or self.wall is None:
            return pixmap

        pos1, pos2 = self.last_swap
        x1, y1 = pos1
        x2, y2 = pos2

        scale = self.display_scale
        cx1 = (x1 + 0.5) * scale
        cy1 = (y1 + 0.5) * scale
        cx2 = (x2 + 0.5) * scale
        cy2 = (y2 + 0.5) * scale

        dx = cx2 - cx1
        dy = cy2 - cy1
        length = (dx**2 + dy**2) ** 0.5

        if length < 0.001:
            return pixmap

        dx /= length
        dy /= length

        # Offset to keep arrows just outside the pixel boundaries
        arrow_size = 8
        pixel_offset = scale * 0.5 + 4  # Half pixel size + small margin

        start_x = cx1 + dx * pixel_offset
        start_y = cy1 + dy * pixel_offset
        end_x = cx2 - dx * pixel_offset
        end_y = cy2 - dy * pixel_offset

        remaining_length = length - 2 * pixel_offset
        if remaining_length > 0:
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))

            # Draw arrowheads
            angle = np.arctan2(dy, dx)

            # Arrow at end (pointing toward pos2)
            arrow_tip_end = (end_x, end_y)
            ax1 = arrow_tip_end[0] - arrow_size * np.cos(angle - np.pi/6)
            ay1 = arrow_tip_end[1] - arrow_size * np.sin(angle - np.pi/6)
            ax2 = arrow_tip_end[0] - arrow_size * np.cos(angle + np.pi/6)
            ay2 = arrow_tip_end[1] - arrow_size * np.sin(angle + np.pi/6)
            painter.drawLine(int(arrow_tip_end[0]), int(arrow_tip_end[1]), int(ax1), int(ay1))
            painter.drawLine(int(arrow_tip_end[0]), int(arrow_tip_end[1]), int(ax2), int(ay2))

            # Arrow at start (pointing toward pos1)
            arrow_tip_start = (start_x, start_y)
            ax1 = arrow_tip_start[0] + arrow_size * np.cos(angle - np.pi/6)
            ay1 = arrow_tip_start[1] + arrow_size * np.sin(angle - np.pi/6)
            ax2 = arrow_tip_start[0] + arrow_size * np.cos(angle + np.pi/6)
            ay2 = arrow_tip_start[1] + arrow_size * np.sin(angle + np.pi/6)
            painter.drawLine(int(arrow_tip_start[0]), int(arrow_tip_start[1]), int(ax1), int(ay1))
            painter.drawLine(int(arrow_tip_start[0]), int(arrow_tip_start[1]), int(ax2), int(ay2))

            painter.end()

        return pixmap

    def get_overlay_image(self, base_img, overlay_img):
        if base_img is None or overlay_img is None:
            return base_img

        try:
            alpha = float(self.overlay_alpha_edit.text())
            alpha = max(0.0, min(1.0, alpha))
        except ValueError:
            alpha = 0.4

        base = base_img.convert('RGB')
        overlay = overlay_img.convert('RGB')

        blended = Image.blend(base, overlay, alpha)
        return blended

    def get_heatmap_image(self):
        if self.dissonance_map is None:
            return Image.new('RGB', (self.wall.width, self.wall.height), (50, 50, 50))

        d_min = self.dissonance_map.min()
        d_max = self.dissonance_map.max()

        if d_max > d_min:
            d_norm = (self.dissonance_map - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(self.dissonance_map)

        rgb = np.zeros((self.wall.height, self.wall.width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(d_norm * 2 * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((d_norm - 0.5) * 2 * 255, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip((d_norm - 0.8) * 5 * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(rgb, 'RGB')

    def get_correctness_image(self):
        if self.wall is None:
            return None

        rgb = np.zeros((self.wall.height, self.wall.width, 3), dtype=np.uint8)

        max_dist = np.sqrt(self.wall.width**2 + self.wall.height**2)

        for y in range(self.wall.height):
            for x in range(self.wall.width):
                orig_x, orig_y = self.wall.get_original_position(x, y)
                if orig_x == x and orig_y == y:
                    rgb[y, x] = [0, 255, 0]
                else:
                    dist = np.sqrt((orig_x - x)**2 + (orig_y - y)**2)
                    intensity = int(100 + 155 * (dist / max_dist))
                    rgb[y, x] = [intensity, 0, 0]

        return Image.fromarray(rgb, 'RGB')

    def on_frame_change(self, value):
        self.current_frame = value
        self.update_display()

    def on_canvas_motion(self, x, y):
        if self.wall is None:
            return

        px = int(x / self.display_scale)
        py = int(y / self.display_scale)

        if px < 0 or px >= self.wall.width or py < 0 or py >= self.wall.height:
            self.position_label.setText("Position: -")
            return

        info = f"Position: ({px}, {py})"

        orig_x, orig_y = self.wall.get_original_position(px, py)
        if orig_x != px or orig_y != py:
            info += f" [orig: ({orig_x}, {orig_y})]"
        else:
            info += " [correct]"

        if self.dissonance_map is not None:
            d = self.dissonance_map[py, px]
            info += f" | Diss: {d:.1f}"

        if (px, py) in self.high_dissonance_positions:
            info += " | HIGH-DISS"

        self.position_label.setText(info)


def get_pink_stylesheet():
    """Return a cute pink theme stylesheet."""
    return """
    /* Main window background */
    QMainWindow {
        background-color: #fff0f5;
    }

    QWidget {
        background-color: #fff0f5;
        color: #8b4563;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    /* Group boxes */
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

    /* Buttons */
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

    /* Text inputs */
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

    /* Combo boxes */
    QComboBox {
        background-color: #fff;
        border: 2px solid #ffb6c1;
        border-radius: 6px;
        padding: 4px 8px;
        color: #8b4563;
        min-height: 20px;
    }

    QComboBox:hover {
        border: 2px solid #ff85a2;
    }

    QComboBox::drop-down {
        border: none;
        width: 20px;
    }

    QComboBox::down-arrow {
        width: 12px;
        height: 12px;
    }

    QComboBox QAbstractItemView {
        background-color: #fff;
        border: 2px solid #ffb6c1;
        selection-background-color: #ffb6c1;
        selection-color: #8b4563;
    }

    /* Labels */
    QLabel {
        background-color: transparent;
        color: #8b4563;
    }

    /* Text edit / results area */
    QTextEdit {
        background-color: #fff;
        border: 2px solid #ffb6c1;
        border-radius: 8px;
        color: #8b4563;
        selection-background-color: #ff85a2;
    }

    /* Scroll area */
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

    /* Radio buttons */
    QRadioButton {
        color: #8b4563;
        spacing: 6px;
    }

    QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border-radius: 9px;
        border: 2px solid #ffb6c1;
        background-color: #fff;
    }

    QRadioButton::indicator:checked {
        background-color: #ff85a2;
        border: 2px solid #ff85a2;
    }

    QRadioButton::indicator:hover {
        border: 2px solid #ff85a2;
    }

    /* Progress bar */
    QProgressBar {
        background-color: #ffe4ec;
        border: 2px solid #ffb6c1;
        border-radius: 8px;
        text-align: center;
        color: #8b4563;
        font-weight: bold;
    }

    QProgressBar::chunk {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #ff85a2, stop:1 #ffb6c1);
        border-radius: 6px;
    }

    /* Sliders */
    QSlider::groove:horizontal {
        height: 8px;
        background-color: #ffe4ec;
        border-radius: 4px;
        border: 1px solid #ffb6c1;
    }

    QSlider::handle:horizontal {
        background-color: #ff85a2;
        width: 18px;
        height: 18px;
        margin: -6px 0;
        border-radius: 9px;
        border: 2px solid #fff;
    }

    QSlider::handle:horizontal:hover {
        background-color: #ff6b8a;
    }

    QSlider::sub-page:horizontal {
        background-color: #ffb6c1;
        border-radius: 4px;
    }

    /* Message boxes */
    QMessageBox {
        background-color: #fff0f5;
    }

    QMessageBox QLabel {
        color: #8b4563;
    }

    QMessageBox QPushButton {
        min-width: 80px;
    }

    /* File dialog */
    QFileDialog {
        background-color: #fff0f5;
    }
    """


def main():
    app = QApplication(sys.argv)

    # Load Kablammo font for fancy iteration counter
    font_path = os.path.join(os.path.dirname(__file__), "Kablammo-Regular-VariableFont_MORF.ttf")
    if os.path.exists(font_path):
        QFontDatabase.addApplicationFont(font_path)

    # Apply the cute pink theme
    app.setStyleSheet(get_pink_stylesheet())

    window = GreedySolverGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
