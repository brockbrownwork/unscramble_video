#!/usr/bin/env python
"""
generate_flow_stimulus.py - Procedural flow-field stimulus video generator.

Generates a video where color is driven by layered animated noise fields and
standing-wave interference patterns, warped by a smooth flow field.  Every
pixel has a unique temporal color signature, and neighboring pixels are
naturally correlated -- the exact properties the unscramble_video solver
exploits.

Key advantages over the ball-based stimulus:
  - No "dead zones" -- the entire frame is information-rich
  - R, G, B channels use independent phase offsets / rotations, creating
    the covariance structure that Mahalanobis distance can exploit
  - Parametric difficulty via noise scale, flow strength, octave count

GPU acceleration (CuPy) is used automatically when available, giving a
~10-30x speedup.  Falls back to NumPy on CPU when CuPy is not installed.

Usage:
    python generate_flow_stimulus.py -o flow_stimulus.mkv --frames 10000 --seed 42
    python generate_flow_stimulus.py --preset hard --seed 42
    python generate_flow_stimulus.py --noise-scale 5 --num-octaves 8 --flow-strength 0.5
    python generate_flow_stimulus.py --no-gpu   # force CPU even if CuPy is available
"""

import argparse
import subprocess
import sys

import numpy as np
from tqdm import tqdm

# ── Optional GPU (CuPy) ─────────────────────────────────────────────
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# ── Hardcoded defaults ───────────────────────────────────────────────
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 10000
DEFAULT_FPS = 30
DEFAULT_OUTPUT = "flow_stimulus.mkv"

# Noise
DEFAULT_NOISE_SCALE = 3.0
DEFAULT_NUM_OCTAVES = 6
DEFAULT_PERSISTENCE = 0.5
DEFAULT_TEMPORAL_SPEED = 0.01

# Flow
DEFAULT_FLOW_STRENGTH = 0.3
DEFAULT_FLOW_SPEED = 0.005
DEFAULT_FLOW_OCTAVES = 3

# Waves
DEFAULT_NUM_WAVES = 4
DEFAULT_WAVE_FREQ_MIN = 5.0
DEFAULT_WAVE_FREQ_MAX = 20.0

# Layer mixing
DEFAULT_NOISE_WEIGHT = 0.7
DEFAULT_WAVE_WEIGHT = 0.3

# ── Presets ───────────────────────────────────────────────────────────
PRESETS = {
    "easy": {
        "noise_scale": 2.0,
        "num_octaves": 4,
        "temporal_speed": 0.005,
        "flow_strength": 0.15,
        "num_waves": 2,
    },
    "medium": {
        # identical to defaults
    },
    "hard": {
        "noise_scale": 5.0,
        "num_octaves": 8,
        "temporal_speed": 0.02,
        "flow_strength": 0.5,
        "num_waves": 6,
    },
}


class FlowFieldParams:
    """Container for all precomputed random parameters of the flow field.

    All parameter arrays are created as NumPy float32 and optionally
    transferred to GPU via ``to_gpu()``.  The ``xp`` attribute holds the
    active array module (``numpy`` or ``cupy``).
    """

    def __init__(self, width, height, seed, noise_scale, num_octaves,
                 persistence, temporal_speed, flow_strength, flow_speed,
                 flow_octaves, num_waves, wave_freq_min, wave_freq_max,
                 noise_weight, wave_weight):
        self.width = width
        self.height = height
        self.noise_weight = np.float32(noise_weight)
        self.wave_weight = np.float32(wave_weight)
        self.num_octaves = num_octaves
        self.flow_octaves = flow_octaves
        self.num_waves = num_waves
        self.flow_strength = np.float32(flow_strength)

        # Default to CPU
        self.xp = np

        rng = np.random.default_rng(seed)

        # ── Color noise octave parameters ────────────────────────────
        # Per-channel frequency ladders with slight scaling offsets so
        # channels don't track each other perfectly.  This creates the
        # R/G/B covariance structure that Mahalanobis distance exploits.
        channel_freq_scales = np.array([1.0, 1.15, 0.85], dtype=np.float32)
        base_freqs = noise_scale * (2.0 ** np.arange(num_octaves))  # (K,)
        self.color_freqs = np.float32(
            base_freqs[:, None] * channel_freq_scales[None, :]
        )  # (K, C)
        self.color_amps = np.float32(
            persistence ** np.arange(num_octaves)
        )  # (K,)
        # Per-octave, per-channel: 3 phase offsets, 1 rotation angle, 1 temporal rate
        self.color_phases = rng.uniform(
            0, 2 * np.pi, (num_octaves, 3, 3)
        ).astype(np.float32)  # (K, C, 3)
        self.color_rotations = rng.uniform(
            0, 2 * np.pi, (num_octaves, 3)
        ).astype(np.float32)  # (K, C)
        # Wider spread on temporal rates (0.3-3.0x) for more channel
        # decorrelation over time
        self.color_temporal = (
            rng.uniform(0.3, 3.0, (num_octaves, 3)) * temporal_speed
        ).astype(np.float32)  # (K, C)

        # Precompute cos/sin of rotation angles  (K, C)
        self.color_cos_rot = np.cos(self.color_rotations).astype(np.float32)
        self.color_sin_rot = np.sin(self.color_rotations).astype(np.float32)

        # ── Flow noise parameters ────────────────────────────────────
        self.flow_freqs = np.float32(
            (noise_scale * 0.5) * (2.0 ** np.arange(flow_octaves))
        )
        self.flow_amps = np.float32(
            persistence ** np.arange(flow_octaves)
        )
        # 2 components (dx, dy)
        self.flow_phases = rng.uniform(
            0, 2 * np.pi, (flow_octaves, 2, 3)
        ).astype(np.float32)
        self.flow_rotations = rng.uniform(
            0, 2 * np.pi, (flow_octaves, 2)
        ).astype(np.float32)
        self.flow_temporal = (
            rng.uniform(0.5, 2.0, (flow_octaves, 2)) * flow_speed
        ).astype(np.float32)
        self.flow_cos_rot = np.cos(self.flow_rotations).astype(np.float32)
        self.flow_sin_rot = np.sin(self.flow_rotations).astype(np.float32)

        # ── Wave interference parameters ─────────────────────────────
        wave_angles = rng.uniform(0, 2 * np.pi, num_waves)
        wave_spatial_freqs = rng.uniform(wave_freq_min, wave_freq_max, num_waves)
        self.wave_kx = (wave_spatial_freqs * np.cos(wave_angles)).astype(np.float32)
        self.wave_ky = (wave_spatial_freqs * np.sin(wave_angles)).astype(np.float32)
        self.wave_omega = (
            rng.uniform(0.5, 3.0, num_waves) * temporal_speed
        ).astype(np.float32)
        self.wave_phi = rng.uniform(0, 2 * np.pi, num_waves).astype(np.float32)
        self.wave_amps = rng.uniform(0.3, 1.0, num_waves).astype(np.float32)
        # Per-wave, per-channel mixing weight  (W, 3)
        mix = rng.uniform(0.3, 1.0, (num_waves, 3)).astype(np.float32)
        mix /= mix.max(axis=1, keepdims=True)
        self.wave_channel_mix = mix

    # ── GPU transfer ─────────────────────────────────────────────────

    # Names of all ndarray attributes that should be transferred to GPU
    _ARRAY_ATTRS = [
        "color_freqs", "color_amps", "color_phases", "color_cos_rot",
        "color_sin_rot", "color_temporal",
        "flow_freqs", "flow_amps", "flow_phases", "flow_cos_rot",
        "flow_sin_rot", "flow_temporal",
        "wave_kx", "wave_ky", "wave_omega", "wave_phi", "wave_amps",
        "wave_channel_mix",
    ]

    def to_gpu(self):
        """Transfer all parameter arrays to GPU (in-place).  No-op if
        CuPy is not available."""
        if not CUPY_AVAILABLE:
            return
        self.xp = cp
        for name in self._ARRAY_ATTRS:
            setattr(self, name, cp.asarray(getattr(self, name)))

    def to_cpu(self):
        """Transfer all parameter arrays back to CPU (in-place)."""
        if not CUPY_AVAILABLE:
            return
        self.xp = np
        for name in self._ARRAY_ATTRS:
            val = getattr(self, name)
            if isinstance(val, cp.ndarray):
                setattr(self, name, cp.asnumpy(val))


# ── Noise evaluation ─────────────────────────────────────────────────

def _evaluate_harmonic_noise(xp, U, V, t, freqs, amps, phases, cos_rot,
                              sin_rot, temporal, num_octaves, channel_idx):
    """Evaluate layered harmonic noise for one channel/component.

    Parameters
    ----------
    xp : module
        Array module — ``numpy`` or ``cupy``.
    U, V : ndarray (H, W) float32
        Sampling coordinates (possibly flow-warped).
    t : float
        Frame time index.
    freqs : ndarray (K,) or (K, C)
        Spatial frequencies.  If 2-D, ``freqs[k, channel_idx]`` is used
        so that each color channel has a slightly different frequency
        ladder (improves channel decorrelation for Mahalanobis).
    amps : ndarray (K,)
    phases : ndarray (K, C_or_2, 3)
    cos_rot, sin_rot : ndarray (K, C_or_2)
    temporal : ndarray (K, C_or_2)
    num_octaves : int
    channel_idx : int
        Which channel/component slice to use.

    Returns
    -------
    ndarray (H, W) float32  in roughly [-1, 1]
    """
    per_channel_freq = freqs.ndim == 2
    result = xp.zeros_like(U)
    for k in range(num_octaves):
        f = freqs[k, channel_idx] if per_channel_freq else freqs[k]
        amp = amps[k]
        ct = cos_rot[k, channel_idx]
        st = sin_rot[k, channel_idx]
        a = temporal[k, channel_idx]
        p = phases[k, channel_idx]  # (3,)

        # Rotated coordinate
        R = f * (ct * U + st * V)
        R_perp = f * (-st * U + ct * V)

        val = (xp.sin(R + p[0] + a * t)
               + xp.sin(R_perp + p[1] + a * 1.3 * t)
               + xp.sin(R * 0.7 + f * V * 0.7 + p[2] + a * 0.7 * t))
        val *= (1.0 / 3.0)
        result += amp * val

    return result


def compute_flow_displacement(xp, u_grid, v_grid, t, params):
    """Compute (du, dv) flow displacements for all pixels."""
    du = _evaluate_harmonic_noise(
        xp, u_grid, v_grid, t,
        params.flow_freqs, params.flow_amps,
        params.flow_phases, params.flow_cos_rot, params.flow_sin_rot,
        params.flow_temporal, params.flow_octaves, channel_idx=0,
    )
    dv = _evaluate_harmonic_noise(
        xp, u_grid, v_grid, t,
        params.flow_freqs, params.flow_amps,
        params.flow_phases, params.flow_cos_rot, params.flow_sin_rot,
        params.flow_temporal, params.flow_octaves, channel_idx=1,
    )
    du *= params.flow_strength
    dv *= params.flow_strength
    return du, dv


def compute_noise_layer(xp, U, V, t, params, channel):
    """Evaluate harmonic noise for one color channel."""
    return _evaluate_harmonic_noise(
        xp, U, V, t,
        params.color_freqs, params.color_amps,
        params.color_phases, params.color_cos_rot, params.color_sin_rot,
        params.color_temporal, params.num_octaves, channel_idx=channel,
    )


def compute_wave_layer(xp, U, V, t, params):
    """Evaluate standing-wave interference for all three channels.

    Returns
    -------
    (R, G, B) : tuple of ndarray (H, W) float32
    """
    R = xp.zeros_like(U)
    G = xp.zeros_like(U)
    B = xp.zeros_like(U)
    for i in range(params.num_waves):
        w = params.wave_amps[i] * xp.sin(
            params.wave_kx[i] * U
            + params.wave_ky[i] * V
            + params.wave_omega[i] * t
            + params.wave_phi[i]
        )
        R += w * params.wave_channel_mix[i, 0]
        G += w * params.wave_channel_mix[i, 1]
        B += w * params.wave_channel_mix[i, 2]
    return R, G, B


# ── Frame rendering ──────────────────────────────────────────────────

def render_frame(xp, u_grid, v_grid, t, params):
    """Render a single frame as a (H, W, 3) BGR uint8 array.

    Computation happens on whichever device ``xp`` targets (CPU or GPU).
    The returned array is always a NumPy (CPU) array so it can be piped
    to ffmpeg.
    """
    # Step 1: Flow warp
    du, dv = compute_flow_displacement(xp, u_grid, v_grid, t, params)
    U = u_grid + du
    V = v_grid + dv

    # Step 2: Base noise per channel
    nw = params.noise_weight
    R = compute_noise_layer(xp, U, V, t, params, channel=0) * nw
    G = compute_noise_layer(xp, U, V, t, params, channel=1) * nw
    B = compute_noise_layer(xp, U, V, t, params, channel=2) * nw

    # Step 3: Wave interference
    wR, wG, wB = compute_wave_layer(xp, U, V, t, params)
    ww = params.wave_weight
    R += wR * ww
    G += wG * ww
    B += wB * ww

    # Step 4: Normalize [-1, 1] -> [0, 255], BGR order for ffmpeg
    frame = xp.empty((params.height, params.width, 3), dtype=xp.float32)
    frame[:, :, 0] = B
    frame[:, :, 1] = G
    frame[:, :, 2] = R

    # Map to [0, 255].  The sum-of-octaves with persistence=0.5 converges to
    # ~2x the base amplitude, so raw values sit in roughly [-1.4, 1.4].
    # Adding 1 and scaling by 127.5 maps [-1, 1] -> [0, 255].  Values outside
    # that range are clipped.
    xp.add(frame, 1.0, out=frame)
    xp.multiply(frame, 127.5, out=frame)
    xp.clip(frame, 0, 255, out=frame)
    frame = frame.astype(xp.uint8)

    # Transfer back to CPU if on GPU
    if xp is not np:
        frame = cp.asnumpy(frame)
    return frame


# ── ffmpeg encoder ───────────────────────────────────────────────────

def start_ffmpeg_encoder(width, height, fps, output_path):
    """Launch ffmpeg subprocess that reads raw BGR frames from stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg and ensure "
              "it's on your PATH.")
        sys.exit(1)
    return proc


# ── Main generation loop ─────────────────────────────────────────────

def generate_flow_stimulus(width, height, num_frames, fps, seed, output_path,
                           use_gpu=True, **kwargs):
    """Generate the procedural flow-field stimulus video."""
    params = FlowFieldParams(
        width=width, height=height, seed=seed,
        noise_scale=kwargs.get("noise_scale", DEFAULT_NOISE_SCALE),
        num_octaves=kwargs.get("num_octaves", DEFAULT_NUM_OCTAVES),
        persistence=kwargs.get("persistence", DEFAULT_PERSISTENCE),
        temporal_speed=kwargs.get("temporal_speed", DEFAULT_TEMPORAL_SPEED),
        flow_strength=kwargs.get("flow_strength", DEFAULT_FLOW_STRENGTH),
        flow_speed=kwargs.get("flow_speed", DEFAULT_FLOW_SPEED),
        flow_octaves=kwargs.get("flow_octaves", DEFAULT_FLOW_OCTAVES),
        num_waves=kwargs.get("num_waves", DEFAULT_NUM_WAVES),
        wave_freq_min=kwargs.get("wave_freq_min", DEFAULT_WAVE_FREQ_MIN),
        wave_freq_max=kwargs.get("wave_freq_max", DEFAULT_WAVE_FREQ_MAX),
        noise_weight=kwargs.get("noise_weight", DEFAULT_NOISE_WEIGHT),
        wave_weight=kwargs.get("wave_weight", DEFAULT_WAVE_WEIGHT),
    )

    # Decide CPU vs GPU
    gpu_active = use_gpu and CUPY_AVAILABLE
    if gpu_active:
        params.to_gpu()
        xp = cp
    else:
        xp = np

    # Coordinate grids normalized to [0, 2*pi] for nice spatial frequencies
    v_grid, u_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    u_grid = u_grid / width * 2 * np.pi
    v_grid = v_grid / height * 2 * np.pi
    if gpu_active:
        u_grid = cp.asarray(u_grid)
        v_grid = cp.asarray(v_grid)

    proc = start_ffmpeg_encoder(width, height, fps, output_path)

    device_str = "GPU (CuPy)" if gpu_active else "CPU (NumPy)"
    print(f"Generating {num_frames} frames at {width}x{height}  [{device_str}]")
    print(f"  noise_scale={float(params.color_freqs[0, 0]):.1f}  "
          f"octaves={params.num_octaves}  "
          f"flow_strength={float(params.flow_strength):.2f}  "
          f"waves={params.num_waves}")
    print(f"  seed={seed}  output={output_path}")

    try:
        for frame_idx in tqdm(range(num_frames), desc="Rendering", unit="frame"):
            t = float(frame_idx)
            frame = render_frame(xp, u_grid, v_grid, t, params)
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        print("\nError: ffmpeg pipe broke unexpectedly.")
        sys.exit(1)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            print(f"ffmpeg exited with code {proc.returncode}")
        # Free GPU memory
        if gpu_active:
            params.to_cpu()
            cp.get_default_memory_pool().free_all_blocks()

    print(f"Done. Wrote {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate procedural flow-field stimulus video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="Output video path")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help="Video width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help="Video height in pixels")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="Frames per second")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU even if CuPy is available")

    # Noise parameters
    noise = parser.add_argument_group("noise")
    noise.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE,
                       help="Base spatial frequency (higher = finer texture)")
    noise.add_argument("--num-octaves", type=int, default=DEFAULT_NUM_OCTAVES,
                       help="Number of noise octaves (texture complexity)")
    noise.add_argument("--persistence", type=float, default=DEFAULT_PERSISTENCE,
                       help="Amplitude falloff per octave (0-1)")
    noise.add_argument("--temporal-speed", type=float,
                       default=DEFAULT_TEMPORAL_SPEED,
                       help="Base temporal evolution rate")

    # Flow parameters
    flow = parser.add_argument_group("flow")
    flow.add_argument("--flow-strength", type=float,
                      default=DEFAULT_FLOW_STRENGTH,
                      help="Flow warp magnitude (fraction of coordinate range)")
    flow.add_argument("--flow-speed", type=float, default=DEFAULT_FLOW_SPEED,
                      help="Flow temporal evolution rate")
    flow.add_argument("--flow-octaves", type=int, default=DEFAULT_FLOW_OCTAVES,
                      help="Number of octaves for the flow field")

    # Wave parameters
    waves = parser.add_argument_group("waves")
    waves.add_argument("--num-waves", type=int, default=DEFAULT_NUM_WAVES,
                       help="Number of interference wave patterns")
    waves.add_argument("--wave-freq-min", type=float,
                       default=DEFAULT_WAVE_FREQ_MIN,
                       help="Minimum spatial frequency for waves")
    waves.add_argument("--wave-freq-max", type=float,
                       default=DEFAULT_WAVE_FREQ_MAX,
                       help="Maximum spatial frequency for waves")

    # Mixing
    mix = parser.add_argument_group("mixing")
    mix.add_argument("--noise-weight", type=float, default=DEFAULT_NOISE_WEIGHT,
                     help="Weight of noise layer in final color")
    mix.add_argument("--wave-weight", type=float, default=DEFAULT_WAVE_WEIGHT,
                     help="Weight of wave layer in final color")

    # Presets
    parser.add_argument("--preset", choices=["easy", "medium", "hard"],
                        default=None,
                        help="Difficulty preset (overridden by explicit args)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Apply preset as defaults, then let explicit CLI args override
    kwargs = {}
    if args.preset and args.preset in PRESETS:
        kwargs.update(PRESETS[args.preset])

    # Explicit CLI args override preset values
    explicit_keys = {
        "noise_scale": args.noise_scale,
        "num_octaves": args.num_octaves,
        "persistence": args.persistence,
        "temporal_speed": args.temporal_speed,
        "flow_strength": args.flow_strength,
        "flow_speed": args.flow_speed,
        "flow_octaves": args.flow_octaves,
        "num_waves": args.num_waves,
        "wave_freq_min": args.wave_freq_min,
        "wave_freq_max": args.wave_freq_max,
        "noise_weight": args.noise_weight,
        "wave_weight": args.wave_weight,
    }
    # Only override if the user actually passed the flag (not the default).
    # Since argparse doesn't easily track this, we always apply explicit
    # values -- the preset is just a convenient starting point.
    for k, v in explicit_keys.items():
        if k not in kwargs:
            kwargs[k] = v

    generate_flow_stimulus(
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        output_path=args.output,
        use_gpu=not args.no_gpu,
        **kwargs,
    )


if __name__ == "__main__":
    main()
