#!/usr/bin/env python3
"""
TSNFA Monte Carlo Simulator v1.2 (Deliverable 2, revision)
==========================================================
VERSION HISTORY
  v1.1  Original Deliverable-2 simulator. ProposedMethod implemented the
        EMA variant reverse-engineered from deployed STM32 hardware
        (max-across-bins, gamma_d MEAN filter, gated exponential-smoothing
        noise floor, trigger on filtered value). Produced the results in
        the originally submitted manuscript. The median variant described
        in manuscript Algorithm 1 was specified but, by mistake, never
        implemented.
  v1.2  (this file, August 2026) ProposedMethod is now variant-dispatching:
        * 'median' (DEFAULT): first implementation of Algorithm 1, with
          two corrections validated in the v1.2 verification study
          (tsnfa_variants.py): trigger on the Stage-1 median (Defence 2
          in the trigger path) and detection-gated Stage-2 floor updates
          (prevents self-poisoning by long events). Per-bin buffers with
          OR-across-bins trigger logic, as the manuscript describes.
        * 'ema': the v1.1 class verbatim, kept for regression comparison
          (--tsnfa-variant ema reproduces v1.1 results exactly).
        Also: CLI overrides added (mirrors simulator_m4f.py pattern);
        results JSON records the variant in _simulation_parameters;
        two path bugs fixed (snapshot path doubling, output-dir override
        not reaching derived filenames).
============================================
Compares TSNFA (Proposed) against four locked classical comparator algorithms:

  Slot 1: Lipski FFT energy detector (Lipski et al. 2021)        [Cortex-M0]
  Slot 2: CA-CFAR self-referencing temporal (Finn & Johnson 1968) [Cortex-M0]
  Slot 3: OS-CFAR self-referencing temporal (Rohling 1983)        [Cortex-M0]
  Slot 4: CUSUM Tartakovsky variant (Torre et al. 2023)           [Cortex-M0]

All four classical comparators are real algorithms imported from
comparators_classical.py. ML-based comparators (Cerutti's compact KD-RNN
and Gong's AST) require a separate study with full neural-network
implementations and are evaluated in a follow-up paper.

Output directory defaults to U:/MONTECARLO/data; subdirs created automatically.

Author: GNACODE INC
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import heapq
from enum import Enum
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import json
import time
import sys
import os
from datetime import datetime, timedelta

# Locked classical comparators (Slots 1-4) - see comparators_classical.py
from comparators_classical import (
    LipskiFFTMethod,
    CACFARMethod,
    OSCFARMethod,
    CUSUMMethod,
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class Logger:
    """Simple logger with levels and timestamps"""
    
    LEVELS = {'DEBUG': 0, 'INFO': 1, 'PROGRESS': 2, 'WARNING': 3, 'ERROR': 4}
    
    def __init__(self, level: str = 'INFO', show_timestamp: bool = True):
        self.level = self.LEVELS.get(level.upper(), 1)
        self.show_timestamp = show_timestamp
        self.start_time = time.time()
        
    def _format(self, level: str, msg: str) -> str:
        elapsed = time.time() - self.start_time
        if self.show_timestamp:
            return f"[{elapsed:8.1f}s] [{level:8s}] {msg}"
        return f"[{level:8s}] {msg}"
    
    def debug(self, msg: str):
        if self.level <= self.LEVELS['DEBUG']:
            print(self._format('DEBUG', msg))
    
    def info(self, msg: str):
        if self.level <= self.LEVELS['INFO']:
            print(self._format('INFO', msg))
    
    def progress(self, msg: str):
        if self.level <= self.LEVELS['PROGRESS']:
            print(self._format('PROGRESS', msg))
    
    def warning(self, msg: str):
        if self.level <= self.LEVELS['WARNING']:
            print(self._format('WARNING', msg))
    
    def error(self, msg: str):
        if self.level <= self.LEVELS['ERROR']:
            print(self._format('ERROR', msg))
    
    def section(self, title: str):
        """Print a section header"""
        if self.level <= self.LEVELS['INFO']:
            print("\n" + "="*70)
            print(f" {title}")
            print("="*70)
    
    def subsection(self, title: str):
        """Print a subsection header"""
        if self.level <= self.LEVELS['INFO']:
            print(f"\n--- {title} ---")


# Global logger instance
log = Logger(level='INFO')


# =============================================================================
# ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗ ██╗   ██╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗██║   ██║██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║██║   ██║██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
# =============================================================================
# EDIT THESE SETTINGS TO CONTROL THE SIMULATION
# =============================================================================

# --- SPEED vs ACCURACY ---
# Options: 'FAST', 'MEDIUM', 'ACCURATE', 'OVERNIGHT'
#   FAST:      ~1 min,  1 hour sim,  2 MC runs  (quick test)
#   MEDIUM:    ~5 min,  4 hour sim,  3 MC runs  (default)
#   ACCURATE:  ~30 min, 24 hour sim, 5 MC runs  (paper quality)
#   OVERNIGHT: ~1 hr,   24 hour sim, 10 MC runs (publication quality)
SIMULATION_PRESET = 'ACCURATE'

# --- NETWORK SIZES TO SIMULATE ---
# Set to True/False to include/exclude each network size
RUN_10_NODES = True
RUN_50_NODES = False
RUN_1000_NODES = False

# --- ALGORITHM PARAMETERS (from real hardware data) ---
# γ_d: Digital noise filter window [3-5]
GAMMA_D = 3

# γ_a: Long-term adaptation window [64-128]  
GAMMA_A = 64

# ζ: Threshold coefficient (threshold = ζ × noise_floor)
# From real hardware: noise/threshold ≈ 0.17, so ζ = 6
ZETA = 6.0

# TSNFA variant: 'median' = Algorithm 1 (corrected: Stage-1-median trigger,
# gated Stage-2 floor). 'ema' = legacy v1.1 deployed-hardware model.
TSNFA_VARIANT = 'median'
TSNFA_ALL_VARIANTS = True   # run all three TSNFA variants as parallel slots
MC_JOBS = 1                 # worker processes for Monte Carlo replicates (1 = sequential)
TSNFA_CONFIRM = 1           # consecutive above-threshold frames required (hybrid trigger)

# --- EVENT PARAMETERS ---
# Event rate (events per hour per node)
EVENT_RATE = 1.0

# Event SNR in dB (must exceed 20*log10(ζ) ≈ 16 dB to be detected)
EVENT_SNR_DB = 12.0

# Event frequency band (Hz) - human movement detection
EVENT_FREQ_LOW = 1.0
EVENT_FREQ_HIGH = 5.0

# --- ZHANG METHOD PARAMETERS (REMOVED in Deliverable 2) ---
# Zhang dropped from comparator pool; TSNFA's defence-1+2 cascade plays
# the role of a frame-based time-domain reference internally.

# --- SLOT 1 PARAMETERS (Lipski FFT, Cortex-M0) ---

LIPSKI_N_BINS_MIN = 3              # minimum bins exceeding threshold per frame
LIPSKI_M_CAL = 100                 # noise frames for calibration (10 sec @ 100 Hz)
LIPSKI_SLOW_UPDATE_ALPHA = 0.01    # EMA rate for adaptive threshold tracking
LIPSKI_SKIP_DC = True              # skip bin 0 (sensor DC offset)

# --- SLOT 2 PARAMETERS (CA-CFAR, Cortex-M0) ---
CACFAR_N_REF = 32                  # reference cells (lagging-only Adaptation C)
CACFAR_N_GUARD = 4                 # guard cells excluding immediate neighbors
CACFAR_P_FA = 1e-3                 # canonical per-sample false-alarm rate
CACFAR_K_PERSISTENCE = 1           # min supra-threshold samples for frame trigger

# --- SLOT 3 PARAMETERS (OS-CFAR, Cortex-M0) ---
OSCFAR_N_REF = 32                  # reference cells
OSCFAR_N_GUARD = 4                 # guard cells
OSCFAR_K_RANK = 24                 # rank index (= ceil(0.75 · N_ref))
OSCFAR_P_FA = 1e-3                 # canonical per-sample false-alarm rate
OSCFAR_K_PERSISTENCE = 1           # frame-level persistence

# --- SLOT 4 PARAMETERS (CUSUM Tartakovsky, Cortex-M0) ---
CUSUM_SNR_FACTOR = 3.0             # anticipated σ_1 / σ_0 (documented hyperparameter)
CUSUM_ALPHA_FA = 1e-5              # per-sample FAR target for η_min calibration
CUSUM_K_END = 100                  # bounded-window length for η_max calibration
CUSUM_M_CAL_FRAMES = 100           # noise frames for empirical threshold calibration



# --- OUTPUT ---
SAVE_RESULTS = True
OUTPUT_DIR = 'U:/MONTECARLO/data'  # Directory for all output files
RESULTS_FILENAME = f'{OUTPUT_DIR}/simulation_results.json'

# --- ROC SWEEP ---
# When enabled, every frame's per-detector strength score is recorded during
# the canonical run, then post-processed to build (FPR, TPR) curves. Memory
# cost: ~10 nodes × ~67k frames × 7 detectors × 8 bytes ≈ 38 MB for a 24 h
# 10-node ACCURATE run. For 1000-node runs, set RECORD_STRENGTHS = False
# (or accept the ~3.8 GB memory cost).
ENABLE_ROC_SWEEP = True
RECORD_STRENGTHS = True            # required for ENABLE_ROC_SWEEP
ROC_NUM_POINTS = 25                # number of operating points per detector

# --- RAW DATA SNAPSHOTS ---
# For long simulations, save periodic raw data samples
ENABLE_SNAPSHOTS = True            # Save raw waveforms (set False to disable)
SNAPSHOT_DURATION_SEC = 60         # Duration of each snapshot (e.g., 1 min = 60 sec)
SNAPSHOT_INTERVAL_SEC = 1800       # Interval between snapshots (e.g., 30 min = 1800 sec)
SNAPSHOT_NODES = 'ALL'             # 'ALL' or list of node IDs e.g. [1, 2, 3]

# --- CONTINUOUS SAVING (for long simulations) ---
CONTINUOUS_SAVE = True             # Save data continuously (don't wait until end)
CHECKPOINT_INTERVAL_SEC = 3600    # Save checkpoint results every N simulated seconds (e.g., 1 hour)
SNAPSHOT_OUTPUT_DIR = f'{OUTPUT_DIR}/snapshots'  # Directory for snapshot files

# --- NOISE MODEL ---
# Fast noise (always present, high frequency)
NOISE_EMI_FREQ = 60.0              # Power line frequency (Hz) - 50 or 60
NOISE_EMI_AMPLITUDE = 0.3          # Relative to base noise (0-1)
NOISE_DIGITAL_PROB = 0.1           # Probability of digital burst per frame
NOISE_DIGITAL_FREQ_MIN = 800       # Digital noise frequency range (Hz)
NOISE_DIGITAL_FREQ_MAX = 2000

# Environmental noise (per-node varying)
NOISE_ENV_ENABLED = True           # Enable environmental noise sources
NOISE_ENV_RAIN_PROB = 0.05         # Probability of rain starting per hour
NOISE_ENV_WIND_PROB = 0.1          # Probability of wind gust per hour
NOISE_ENV_MOTOR_PROB = 0.02        # Probability of motor/machinery nearby

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class ProgressTracker:
    """Track and display simulation progress"""
    
    def __init__(self, total: float, description: str = "Progress", 
                 update_interval: float = 5.0):
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update = 0
        self.last_progress = 0
        
    def update(self, current: float, force: bool = False):
        """Update progress display"""
        now = time.time()
        progress = current / self.total
        
        if force or (now - self.last_update) >= self.update_interval:
            elapsed = now - self.start_time
            
            if progress > 0:
                eta_seconds = (elapsed / progress) * (1 - progress)
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "calculating..."
            
            # Calculate rate
            rate = current / elapsed if elapsed > 0 else 0
            
            # Progress bar
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # Clear line and print progress
            sys.stdout.write(f"\r{self.description}: [{bar}] {progress*100:5.1f}% | "
                           f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                           f"ETA: {eta_str} | "
                           f"Rate: {rate:.1f}/s    ")
            sys.stdout.flush()
            
            self.last_update = now
            self.last_progress = progress
    
    def finish(self):
        """Mark progress as complete"""
        elapsed = time.time() - self.start_time
        print(f"\r{self.description}: [{'█'*30}] 100.0% | "
              f"Completed in {timedelta(seconds=int(elapsed))}              ")


# =============================================================================
# CONFIGURATION
# =============================================================================

# =============================================================================
# TIME PRESETS - Easy configuration for simulation speed vs accuracy
# =============================================================================

class TimePreset:
    """Predefined time configurations for low-frequency sensing (1-5 Hz)
    
    For human movement detection at 1-5 Hz, we need:
    - Sample rate: 100 Hz (adequate for 5 Hz max frequency)
    - FFT size: 128 → frequency resolution = 0.78 Hz/bin
    - Frame duration: 1.28 sec (N/f_s = 128/100)
    """
    
    # FAST: Quick validation runs
    FAST = {
        'name': 'FAST',
        'duration_hours': 1,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 2,
        'description': 'Quick validation (~1 min for 10 nodes)'
    }
    
    # MEDIUM: Reasonable accuracy with moderate runtime
    MEDIUM = {
        'name': 'MEDIUM', 
        'duration_hours': 4,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 3,
        'description': 'Balanced accuracy/speed (~5 min for 10 nodes)'
    }
    
    # ACCURATE: Full simulation with realistic timing
    ACCURATE = {
        'name': 'ACCURATE',
        'duration_hours': 24,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 5,
        'description': 'Publication quality (~30 min for 10 nodes)'
    }
    
    # OVERNIGHT: Maximum accuracy for paper submission
    OVERNIGHT = {
        'name': 'OVERNIGHT',
        'duration_hours': 24,
        'frame_duration': 1.28,     # 128 samples @ 100 Hz
        'fft_size': 128,
        'monte_carlo_runs': 10,
        'description': 'Maximum statistical confidence (~1 hour total)'
    }
    
    @classmethod
    def list_presets(cls):
        """Print available presets"""
        print("\nAvailable Time Presets:")
        print("-" * 60)
        for name in ['FAST', 'MEDIUM', 'ACCURATE', 'OVERNIGHT']:
            preset = getattr(cls, name)
            print(f"  {name:12} - {preset['description']}")
            print(f"               Duration: {preset['duration_hours']}h, "
                  f"Frame: {preset['frame_duration']*1000:.1f}ms, "
                  f"MC runs: {preset['monte_carlo_runs']}")
        print("-" * 60)


@dataclass
class SimulationConfig:
    """Simulation parameters"""
    # Network
    num_nodes: int = 10
    area_size: float = 500.0  # meters (reduced for better connectivity)
    comm_radius: float = 150.0  # meters
    data_rate: float = 250e3  # bits/sec (IEEE 802.15.4)
    
    # Timing - FOR LOW-FREQUENCY SENSING (1-5 Hz human movement)
    # Need adequate frequency resolution: Δf = f_s / N
    # For 1-5 Hz band with bins 1-6: need Δf ≈ 0.78 Hz → f_s = 100 Hz, N = 128
    simulation_duration: float = 4 * 3600  # seconds (default 4 hours)
    frame_duration: float = 1.28  # 128 samples @ 100 Hz = 1.28 sec
    fft_size: int = 128  # N=128 samples
    sample_rate: float = 100.0  # 100 Hz (adequate for 1-5 Hz events)
    
    # Events - HUMAN MOVEMENT DETECTION
    # Events at 1-5 Hz are slow, sustained changes
    # Frame duration = 1.28 sec, so event must last > γ_d × 1.28 = 3.84 sec
    event_rate: float = 1.0      # events/hour/node
    event_duration: float = 5.0  # 5 seconds - human movement (persists > γ_d frames)
    event_decay_tau: float = 2.0 # 2 sec decay time constant (slow, sustained)
    event_snr: float = 18.0      # dB above noise (must exceed ζ=6 threshold, i.e., >16 dB)
    
    # Noise model
    base_noise_power: float = 1.0
    noise_variation_db: float = 6.0  # ±6 dB variation
    noise_cycle_period: float = 3600.0  # 1 hour cycle
    
    # EMI and fast noise
    emi_freq: float = 60.0             # Power line frequency (Hz)
    emi_amplitude: float = 0.3         # Relative to base noise
    digital_noise_prob: float = 0.1    # Probability per frame
    
    # Environmental noise (per-node)
    env_noise_enabled: bool = True     # Enable rain/wind/motor noise
    
    # Raw data snapshots
    enable_snapshots: bool = False
    snapshot_duration: float = 300.0   # seconds (5 min default)
    snapshot_interval: float = 3600.0  # seconds (1 hour default)
    snapshot_nodes: str = 'ALL'        # 'ALL' or list of node IDs
    
    # Continuous saving (for long simulations)
    continuous_save: bool = True       # Save snapshots immediately when completed
    checkpoint_interval: float = 3600.0  # Save checkpoint every N simulated seconds
    snapshot_output_dir: str = ''      # Directory for snapshot files (empty = same as results)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PROPOSED METHOD PARAMETERS - PAPER NOMENCLATURE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # γ_d (gamma_d) ∈ [3, 5]: Digital noise filter window
    # Removes transient spikes that don't persist > γ_d samples
    # True events (human movement) persist > γ_d samples → pass through
    gamma_d: int = 3
    
    # γ_a (gamma_a) ∈ [64, 128]: Long-term adaptation window
    # Higher γ_a = more smoothing = more stable N_k = HIGHER sensitivity
    # Equivalent α = 1 - 1/γ_a (γ_a=64 → α=0.984, γ_a=128 → α=0.992)
    gamma_a: int = 64
    
    # ζ_k (zeta_k): Threshold coefficient per frequency bin k
    # Trigger if X̄_k > ζ_k × N_k
    # From real hardware data: Threshold ≈ 6× average noise level
    # This gives 0 false positives while detecting events with SNR > 16 dB
    zeta_k: float = 6.0
    
    # Noise floor update control: only update N_k if X̄_k < ratio × Threshold
    noise_update_ratio: float = 0.8  # Update when clearly below threshold
    tsnfa_variant: str = 'median'  # 'median' (Alg.1 corrected) | 'hybrid' | 'ema' (legacy v1.1)
    tsnfa_all_variants: bool = False  # run median+hybrid+ema as parallel slots
    tsnfa_confirm: int = 1  # consecutive above-threshold frames required to declare (hybrid)
    
    # Frequency band of interest: 1-5 Hz (human movement, very low frequency)
    # EMI (50/60 Hz) and digital noise (kHz) are outside this band → ignored
    # Events are consistent over > γ_d samples when they occur
    event_freq_low: float = 1.0    # Hz
    event_freq_high: float = 5.0   # Hz
    
    # Legacy parameter aliases (backward compatibility)
    alpha: float = 0.984           # = 1 - 1/γ_a when γ_a=64
    short_term_avg_samples: int = 3  # Alias for γ_d
    delta_margin_ratio: float = 6.0  # Alias for ζ_k
    delta_update_ratio: float = 0.8  # Alias for noise_update_ratio
    num_fft_bins: int = 8
    min_bins_trigger: int = 4

    # ─── Slot 1: Lipski FFT (Cortex-M0) ────────────────────────────────
    lipski_k: float = 3.0
    lipski_n_bins_min: int = 3
    lipski_m_cal: int = 100
    lipski_slow_update_alpha: float = 0.01
    lipski_skip_dc: bool = True

    # ─── Slot 2: CA-CFAR (Cortex-M0) ───────────────────────────────────
    cacfar_n_ref: int = 32
    cacfar_n_guard: int = 4
    cacfar_p_fa: float = 1e-3
    cacfar_k_persistence: int = 1

    # ─── Slot 3: OS-CFAR (Cortex-M0) ───────────────────────────────────
    oscfar_n_ref: int = 32
    oscfar_n_guard: int = 4
    oscfar_k_rank: int = 24
    oscfar_p_fa: float = 1e-3
    oscfar_k_persistence: int = 1

    # ─── Slot 4: CUSUM Tartakovsky (Cortex-M0) ─────────────────────────
    cusum_snr_factor: float = 3.0
    cusum_alpha_fa: float = 1e-5
    cusum_k_end: int = 100
    cusum_m_cal_frames: int = 100



    # Network protocol
    slot_time: float = 0.00032  # 320 µs
    cw_min: int = 8
    cw_max: int = 64
    max_retries: int = 4
    prop_delay_per_m: float = 3.33e-9  # ~speed of light
    processing_delay: float = 0.0025  # 2.5 ms per hop

    # Payload sizes (bits)
    proposed_payload: int = 64  # TSNFA: timestamp + trigger strength
    lipski_payload: int = 96    # Slot 1: timestamp + per-bin scores
    cacfar_payload: int = 64    # Slot 2: timestamp + strength
    oscfar_payload: int = 64    # Slot 3: timestamp + strength
    cusum_payload: int = 80     # Slot 4: timestamp + start/end markers

    # ROC sweep: record per-frame strengths for post-hoc threshold sweep
    record_strengths: bool = False
    roc_num_points: int = 25
    
    # Random seed
    seed: int = 42
    
    @classmethod
    def from_preset(cls, preset: dict, num_nodes: int = 10, **overrides):
        """Create config from a TimePreset"""
        config = cls(
            num_nodes=num_nodes,
            simulation_duration=preset['duration_hours'] * 3600,
            frame_duration=preset['frame_duration'],
            fft_size=preset['fft_size'],
            **overrides
        )
        return config
    
    def estimate_runtime(self) -> str:
        """Estimate simulation runtime"""
        # Rough estimate: ~50,000 frames/second processing speed
        frames_per_node = self.simulation_duration / self.frame_duration
        total_frames = (self.num_nodes - 1) * frames_per_node
        estimated_seconds = total_frames / 50000  # empirical rate
        
        if estimated_seconds < 60:
            return f"~{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return f"~{estimated_seconds/60:.0f} minutes"
        else:
            return f"~{estimated_seconds/3600:.1f} hours"
    
    def __str__(self):
        frames_per_sec = 1.0 / self.frame_duration
        total_frames = (self.num_nodes - 1) * self.simulation_duration / self.frame_duration

        return (f"SimulationConfig(\n"
                f"  Network: {self.num_nodes} nodes, {self.area_size}m² area, "
                f"{self.comm_radius}m radius\n"
                f"  Timing:  {self.simulation_duration/3600:.1f}h duration, "
                f"{self.frame_duration*1000:.1f}ms frames (N={self.fft_size}), "
                f"{frames_per_sec:.1f} frames/sec/node\n"
                f"  Load:    {total_frames/1e6:.1f}M total frames, "
                f"estimated runtime: {self.estimate_runtime()}\n"
                f"  Events:  {self.event_rate}/hr/node, SNR={self.event_snr}dB, "
                f"band={self.event_freq_low}-{self.event_freq_high}Hz\n"
                f"  TSNFA (proposed): γ_d={self.gamma_d}, γ_a={self.gamma_a}, "
                f"ζ_k={self.zeta_k}\n"
                f"  Slot 1 (Lipski):  k={self.lipski_k}, N_bins_min={self.lipski_n_bins_min}\n"
                f"  Slot 2 (CA-CFAR): N_ref={self.cacfar_n_ref}, "
                f"P_fa={self.cacfar_p_fa:.0e}\n"
                f"  Slot 3 (OS-CFAR): N_ref={self.oscfar_n_ref}, "
                f"k={self.oscfar_k_rank}, P_fa={self.oscfar_p_fa:.0e}\n"
                f"  Slot 4 (CUSUM):   SNR_factor={self.cusum_snr_factor}, "
                f"α_fa={self.cusum_alpha_fa:.0e}\n"
                f")")


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    FRAME_READY = 1
    TRUE_EVENT_START = 2
    TRUE_EVENT_END = 3
    PACKET_TX_START = 4
    PACKET_TX_COMPLETE = 5
    PACKET_ARRIVAL = 6
    NOISE_UPDATE = 7


@dataclass(order=True)
class SimEvent:
    """Simulation event for priority queue"""
    time: float
    event_type: EventType = field(compare=False)
    node_id: int = field(compare=False)
    data: dict = field(default_factory=dict, compare=False)


# =============================================================================
# NODE MODELS
# =============================================================================

class ProposedMethod:
    """Temporal Spectral Noise-Floor Adaptation (TSNFA).

    Variant selected by config.tsnfa_variant:

    'median' (DEFAULT, v1.2) - Algorithm 1 of the MDPI Sensors manuscript,
    with two corrections validated in the v1.2 verification study:
      * Per-bin processing: each event-band bin k keeps its own Stage 1
        buffer B_d,k (gamma_d frames), Stage 2 buffer B_a,k (gamma_a
        entries) and noise floor N_hat_k. Trigger is OR across bins.
      * Stage 1: Ntilde_k = median(B_d,k). The TRIGGER compares Ntilde_k
        (not the instantaneous magnitude) against zeta * N_hat_k, so a
        single-frame in-band burst is outvoted by the median (Defence 2
        acts in the trigger path). A persistent event passes from its
        second frame.
      * Stage 2: N_hat_k = median(B_a,k). Inserts are GATED: performed
        only when no bin triggered AND max ratio < noise_update_ratio
        (hysteresis), preventing event energy from poisoning the floor.
        The median additionally tolerates contaminated entries that pass
        the gate (breakdown point gamma_a/2).

    'ema' (legacy v1.1) - deployed-hardware model reverse-engineered from
    STM32 field data: max across band bins -> gamma_d MEAN filter ->
    single EMA noise floor N with alpha = 1 - 1/gamma_a, gated update,
    trigger on the filtered value. Retained for regression comparison.
    """

    def __init__(self, config: SimulationConfig, node_id: int,
                 variant_override: str = None):
        self.config = config
        self.node_id = node_id
        self.fft_size = config.fft_size
        self.variant = variant_override or getattr(config, 'tsnfa_variant', 'median')

        self.freq_resolution = config.sample_rate / self.fft_size
        self.event_freq_low = config.event_freq_low
        self.event_freq_high = config.event_freq_high
        self.bin_low = max(1, int(self.event_freq_low / self.freq_resolution))
        self.bin_high = min(self.fft_size // 2 - 1,
                            int(self.event_freq_high / self.freq_resolution))
        self.n_monitored_bins = max(1, self.bin_high - self.bin_low + 1)

        self.gamma_d = config.gamma_d
        self.gamma_a = config.gamma_a
        self.zeta_k = config.zeta_k
        self.noise_update_ratio = config.noise_update_ratio

        # --- median-variant state: per-bin ring buffers ---
        self.Bd = [[] for _ in range(self.n_monitored_bins)]
        self.Ba = [[] for _ in range(self.n_monitored_bins)]

        # --- legacy EMA state ---
        self.alpha = 1.0 - 1.0 / self.gamma_a
        expected_magnitude = np.sqrt(self.fft_size) * np.sqrt(config.base_noise_power)
        self.N = expected_magnitude
        self.filter_buffer = []
        self.max_buffer = expected_magnitude
        self.max_buffer_window = []

        self.frames_processed = 0
        self.triggers_issued = 0
        self.confirm_n = max(1, getattr(config, 'tsnfa_confirm', 1))
        self._consec = 0  # consecutive above-threshold frames (hybrid confirmation)

    # ------------------------------------------------------------------
    def process_frame(self, samples: np.ndarray, current_noise_power: float) -> Tuple[bool, float]:
        if self.variant == 'ema':
            return self._process_frame_ema(samples)
        if self.variant == 'hybrid':
            return self._process_frame_hybrid(samples)
        return self._process_frame_median(samples)

    # ------------------------------------------------------------------
    def _process_frame_median(self, samples: np.ndarray) -> Tuple[bool, float]:
        """Algorithm 1 (corrected). Returns (trigger, max Ntilde/Threshold ratio)."""
        self.frames_processed += 1

        spectrum = np.fft.fft(samples)
        mags = np.abs(spectrum[self.bin_low:self.bin_high + 1])

        trigger = False
        max_ratio = 0.0
        ntildes = np.empty(self.n_monitored_bins)

        for i in range(self.n_monitored_bins):
            # Stage 1: short median filter (updates every frame; feeds trigger)
            bd = self.Bd[i]
            bd.append(mags[i])
            if len(bd) > self.gamma_d:
                bd.pop(0)
            ntilde = float(np.median(bd))
            ntildes[i] = ntilde

            # Current noise floor (Stage 2 median); bootstrap from Stage 1
            ba = self.Ba[i]
            n_hat = float(np.median(ba)) if ba else ntilde
            threshold = self.zeta_k * n_hat

            ratio = ntilde / threshold if threshold > 0 else 0.0
            if ratio > max_ratio:
                max_ratio = ratio
            if ntilde > threshold:
                trigger = True  # OR logic across bins

        # Stage 2 floor update: gated on no-detection with hysteresis
        if (not trigger) and (max_ratio < self.noise_update_ratio):
            for i in range(self.n_monitored_bins):
                ba = self.Ba[i]
                ba.append(ntildes[i])
                if len(ba) > self.gamma_a:
                    ba.pop(0)

        if trigger:
            self.triggers_issued += 1
            log.debug(f"Node {self.node_id} TRIGGER (median): "
                      f"max_ratio={max_ratio:.2f}")
        return trigger, max_ratio

    # ------------------------------------------------------------------
    def _process_frame_hybrid(self, samples: np.ndarray) -> Tuple[bool, float]:
        """Hybrid v1.2 (design C): identical front end to the deployed v1.1
        detector -- max across event-band bins, gamma_d-frame MEAN filter,
        detection-gated floor update -- with ONLY the noise-floor estimator
        changed from exponential smoothing to the MEDIAN of a gamma_a-entry
        buffer. The median floor supplies the manuscript's robustness claim
        (breakdown point gamma_a/2, bounded re-anchoring) while trigger
        behaviour matches deployed hardware."""
        self.frames_processed += 1

        full_spectrum = np.abs(np.fft.fft(samples))
        band_spectrum = full_spectrum[self.bin_low:self.bin_high + 1]
        if len(band_spectrum) == 0:
            band_spectrum = full_spectrum[1:2]
        X_raw = float(np.max(band_spectrum))

        self.filter_buffer.append(X_raw)
        if len(self.filter_buffer) > self.gamma_d:
            self.filter_buffer.pop(0)
        X_bar = float(np.mean(self.filter_buffer))

        # Median noise floor over gamma_a gated entries (Ba[0] reused as buffer)
        ba = self.Ba[0]
        n_hat = float(np.median(ba)) if ba else self.N  # bootstrap from init estimate
        threshold = self.zeta_k * n_hat
        ratio = X_bar / threshold if threshold > 0 else 0.0
        crossing = ratio > 1.0
        # M-of-N confirmation (binary integration): declare only after
        # confirm_n consecutive above-threshold frames.
        self._consec = self._consec + 1 if crossing else 0
        trigger = self._consec >= self.confirm_n

        if (not crossing) and (ratio < self.noise_update_ratio):
            ba.append(X_bar)
            if len(ba) > self.gamma_a:
                ba.pop(0)

        if trigger:
            self.triggers_issued += 1
        return trigger, ratio

    def _process_frame_ema(self, samples: np.ndarray) -> Tuple[bool, float]:
        """Legacy v1.1 behaviour, byte-equivalent to the original class."""
        self.frames_processed += 1

        full_spectrum = np.abs(np.fft.fft(samples))
        band_spectrum = full_spectrum[self.bin_low:self.bin_high + 1]
        if len(band_spectrum) == 0:
            band_spectrum = full_spectrum[1:2]
        X_raw = np.max(band_spectrum)

        self.filter_buffer.append(X_raw)
        if len(self.filter_buffer) > self.gamma_d:
            self.filter_buffer.pop(0)
        X_bar = np.mean(self.filter_buffer)

        self.max_buffer_window.append(X_bar)
        if len(self.max_buffer_window) > self.gamma_a:
            self.max_buffer_window.pop(0)
        self.max_buffer = max(self.max_buffer_window) if self.max_buffer_window else X_bar

        Threshold = self.zeta_k * self.N
        ratio = X_bar / Threshold if Threshold > 0 else 0
        trigger = ratio > 1.0

        if not trigger:
            if ratio < self.config.noise_update_ratio:
                self.N = self.alpha * self.N + (1 - self.alpha) * X_bar

        if trigger:
            self.triggers_issued += 1
            log.debug(f"Node {self.node_id} TRIGGER: X_bar={X_bar:.1f}, N={self.N:.1f}, "
                      f"Threshold={Threshold:.1f}, ratio={ratio:.2f}")
        return trigger, ratio

    # ------------------------------------------------------------------
    def reset(self):
        """Reset state for new simulation"""
        self.Bd = [[] for _ in range(self.n_monitored_bins)]
        self.Ba = [[] for _ in range(self.n_monitored_bins)]
        expected_magnitude = np.sqrt(self.fft_size) * np.sqrt(self.config.base_noise_power)
        self.N = expected_magnitude
        self.max_buffer = expected_magnitude
        self.max_buffer_window = []
        self.filter_buffer = []
        self.frames_processed = 0
        self.triggers_issued = 0
        self._consec = 0

    def get_stats(self) -> Dict:
        stats = {
            'variant': self.variant,
            'frames_processed': self.frames_processed,
            'triggers_issued': self.triggers_issued,
            'gamma_d': self.gamma_d,
            'gamma_a': self.gamma_a,
            'zeta_k': self.zeta_k,
            'monitored_band_Hz': f"{self.event_freq_low}-{self.event_freq_high}",
            'monitored_bins': f"{self.bin_low}-{self.bin_high}"
        }
        if self.variant == 'ema':
            stats.update({'N': float(self.N),
                          'Threshold': float(self.zeta_k * self.N),
                          'max_buffer': float(self.max_buffer)})
        else:
            floors = [float(np.median(ba)) if ba else 0.0 for ba in self.Ba]
            stats.update({'noise_floors': floors,
                          'thresholds': [self.zeta_k * f for f in floors]})
        return stats


class NoiseGenerator:
    """
    Realistic noise generator with per-node dynamics.
    
    Two categories of noise:
    1. FAST NOISE (always present, high frequency - OUTSIDE event band):
       - EMI: 50/60 Hz power line + harmonics
       - Sampling artifacts: Quantization noise, ADC glitches
       - Digital switching: kHz range bursts from MCU, regulators
       
    2. ENVIRONMENTAL NOISE (varying per node - may overlap event band):
       - Rain: Broadband noise, intensity varies over hours
       - Wind: Low frequency rumble (1-10 Hz), gusts
       - Motors/Propellers: Specific frequencies based on RPM
       
    The proposed method's FFT filtering removes fast noise (outside 1-5 Hz band)
    but environmental noise may partially overlap the event band.
    """
    
    def __init__(self, node_id: int, config: 'SimulationConfig', seed: int = None):
        self.node_id = node_id
        self.config = config
        
        # Per-node random state for reproducibility
        self.rng = np.random.RandomState(seed if seed else node_id * 12345)
        
        # Fast noise parameters (constant for this node)
        self.emi_phase = self.rng.uniform(0, 2 * np.pi)  # Random phase offset
        self.emi_freq = config.emi_freq if hasattr(config, 'emi_freq') else 60.0
        
        # Environmental noise state (varies over time)
        self.rain_intensity = 0.0      # 0 = no rain, 1 = heavy rain
        self.rain_duration = 0.0       # Remaining rain time
        self.wind_intensity = 0.0      # 0 = calm, 1 = strong gusts
        self.wind_gust_time = 0.0      # Time of current gust
        self.motor_active = False      # Nearby motor running
        self.motor_freq = 0.0          # Motor frequency (Hz)
        self.motor_duration = 0.0      # Remaining motor time
        
        # Propeller/fan parameters (if motor active)
        self.propeller_blades = self.rng.choice([2, 3, 4, 6])  # Number of blades
        
    def update_environmental_state(self, dt: float, current_time: float):
        """Update environmental noise sources based on time progression"""
        
        # Rain dynamics - slow changes over hours
        if self.rain_duration > 0:
            self.rain_duration -= dt
            # Rain intensity varies slowly
            self.rain_intensity += self.rng.uniform(-0.01, 0.01) * dt
            self.rain_intensity = np.clip(self.rain_intensity, 0.1, 1.0)
            if self.rain_duration <= 0:
                self.rain_intensity = 0.0
        else:
            # Chance of rain starting (per hour → per second probability)
            rain_prob_per_sec = 0.05 / 3600  # ~5% per hour
            if self.rng.random() < rain_prob_per_sec * dt:
                self.rain_intensity = self.rng.uniform(0.2, 0.8)
                self.rain_duration = self.rng.uniform(600, 3600)  # 10 min to 1 hour
        
        # Wind dynamics - gusts are shorter events
        if self.wind_intensity > 0:
            # Wind dies down
            self.wind_intensity *= np.exp(-dt / 30)  # 30 sec decay
            if self.wind_intensity < 0.05:
                self.wind_intensity = 0.0
        else:
            # Chance of wind gust
            wind_prob_per_sec = 0.1 / 3600  # ~10% per hour
            if self.rng.random() < wind_prob_per_sec * dt:
                self.wind_intensity = self.rng.uniform(0.3, 1.0)
        
        # Motor/machinery dynamics
        if self.motor_duration > 0:
            self.motor_duration -= dt
            if self.motor_duration <= 0:
                self.motor_active = False
                self.motor_freq = 0.0
        else:
            # Chance of motor starting nearby
            motor_prob_per_sec = 0.02 / 3600  # ~2% per hour
            if self.rng.random() < motor_prob_per_sec * dt:
                self.motor_active = True
                # Motor frequency: RPM / 60 → Hz, typical 1800-3600 RPM
                rpm = self.rng.uniform(1200, 3600)
                self.motor_freq = rpm / 60  # 20-60 Hz base frequency
                self.motor_duration = self.rng.uniform(60, 600)  # 1-10 minutes
    
    def generate_noise(self, n_samples: int, t: np.ndarray, base_noise_power: float) -> np.ndarray:
        """
        Generate realistic noise signal for one frame.
        
        Args:
            n_samples: Number of samples in frame
            t: Time array for this frame
            base_noise_power: Base noise power level
            
        Returns:
            Noise signal array
        """
        noise = np.zeros(n_samples)
        
        # =====================================================================
        # 1. THERMAL/SENSOR NOISE (broadband white noise)
        # =====================================================================
        noise += self.rng.randn(n_samples) * np.sqrt(base_noise_power)
        
        # =====================================================================
        # 2. EMI - Power line interference (50/60 Hz + harmonics)
        #    OUTSIDE 1-5 Hz event band → filtered by FFT
        # =====================================================================
        emi_amp = 0.3 * np.sqrt(base_noise_power)
        noise += emi_amp * np.sin(2 * np.pi * self.emi_freq * t + self.emi_phase)
        noise += emi_amp * 0.5 * np.sin(2 * np.pi * 2 * self.emi_freq * t + self.emi_phase)  # 2nd harmonic
        noise += emi_amp * 0.25 * np.sin(2 * np.pi * 3 * self.emi_freq * t + self.emi_phase)  # 3rd harmonic
        
        # =====================================================================
        # 3. DIGITAL SWITCHING NOISE (kHz range bursts)
        #    OUTSIDE 1-5 Hz event band → filtered by FFT
        # =====================================================================
        if self.rng.random() < 0.1:  # 10% chance per frame
            digital_freq = self.rng.uniform(800, 2000)
            digital_amp = self.rng.uniform(0.5, 2.0) * np.sqrt(base_noise_power)
            burst_start = self.rng.randint(0, n_samples // 2)
            burst_len = self.rng.randint(10, 30)
            burst_mask = np.zeros(n_samples)
            burst_mask[burst_start:min(burst_start + burst_len, n_samples)] = 1.0
            noise += digital_amp * np.sin(2 * np.pi * digital_freq * t) * burst_mask
        
        # =====================================================================
        # 4. RAIN NOISE (broadband, partially in event band)
        #    Rain creates broadband noise including 1-5 Hz components
        # =====================================================================
        if self.rain_intensity > 0:
            # Rain is broadband but has low-frequency rumble
            rain_noise = self.rng.randn(n_samples) * self.rain_intensity * 0.5 * np.sqrt(base_noise_power)
            # Add low-frequency component (partially in event band!)
            rain_low_freq = self.rng.uniform(0.5, 3.0)  # Hz - overlaps event band
            rain_noise += self.rain_intensity * 0.3 * np.sqrt(base_noise_power) * \
                         np.sin(2 * np.pi * rain_low_freq * t + self.rng.uniform(0, 2*np.pi))
            noise += rain_noise
        
        # =====================================================================
        # 5. WIND NOISE (low frequency rumble, IN event band)
        #    Wind creates 0.5-5 Hz pressure fluctuations
        # =====================================================================
        if self.wind_intensity > 0:
            # Wind is primarily low frequency - IN the event band
            wind_freq = self.rng.uniform(0.5, 4.0)  # Hz - IN event band!
            wind_amp = self.wind_intensity * 0.6 * np.sqrt(base_noise_power)
            # Wind has irregular pattern
            wind_mod = 1 + 0.5 * np.sin(2 * np.pi * 0.2 * t)  # Slow modulation
            noise += wind_amp * np.sin(2 * np.pi * wind_freq * t) * wind_mod
        
        # =====================================================================
        # 6. MOTOR/PROPELLER NOISE (specific frequency + harmonics)
        #    Motor base freq 20-60 Hz (outside band), but propeller blades
        #    create subharmonics that may be in 1-5 Hz band
        # =====================================================================
        if self.motor_active and self.motor_freq > 0:
            motor_amp = 0.4 * np.sqrt(base_noise_power)
            # Base motor frequency (outside event band)
            noise += motor_amp * np.sin(2 * np.pi * self.motor_freq * t)
            
            # Propeller blade-pass frequency = motor_freq * blades
            # But also creates LOW frequency vibration from imbalance
            imbalance_freq = self.motor_freq / self.propeller_blades  # Could be 5-20 Hz
            if imbalance_freq < 10:  # If low enough to matter
                noise += motor_amp * 0.3 * np.sin(2 * np.pi * imbalance_freq * t)
        
        return noise
    
    def get_state(self) -> Dict:
        """Return current environmental state for debugging/visualization"""
        return {
            'rain_intensity': self.rain_intensity,
            'rain_duration': self.rain_duration,
            'wind_intensity': self.wind_intensity,
            'motor_active': self.motor_active,
            'motor_freq': self.motor_freq,
        }


@dataclass
class RawDataSnapshot:
    """Container for raw waveform data snapshot"""
    timestamp: float              # Simulation time when snapshot started
    duration: float               # Duration of snapshot in seconds
    sample_rate: float            # Samples per second
    node_data: Dict[int, Dict]    # {node_id: {'samples': array, 'events': list, 'triggers': list}}
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        trigger_keys = ['triggers_proposed', 'triggers_tsnfa_hybrid', 'triggers_tsnfa_ema',
                        'triggers_lipski', 'triggers_cacfar',
                        'triggers_oscfar', 'triggers_cusum']
        return {
            'timestamp': self.timestamp,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'nodes': {
                str(node_id): {
                    'samples': data['samples'].tolist() if isinstance(data['samples'], np.ndarray) else data['samples'],
                    'events': data.get('events', []),
                    'noise_state': data.get('noise_state', {}),
                    **{k: data.get(k, []) for k in trigger_keys},
                }
                for node_id, data in self.node_data.items()
            }
        }


# =============================================================================
# NETWORK MODEL
# =============================================================================

@dataclass
class Node:
    """Sensor node in the mesh network"""
    node_id: int
    x: float
    y: float
    neighbors: List[int] = field(default_factory=list)
    route_to_sink: List[int] = field(default_factory=list)
    hop_count: int = 0
    
    # Detection methods (six locked comparator slots + TSNFA proposed)
    proposed: Optional[ProposedMethod] = None       # TSNFA reference algorithm
    tsnfa_hybrid: Optional[ProposedMethod] = None   # TSNFA hybrid (mean trig / median floor)
    tsnfa_ema:    Optional[ProposedMethod] = None   # TSNFA legacy v1.1 (EMA floor)
    lipski:   Optional[LipskiFFTMethod] = None      # Slot 1
    cacfar:   Optional[CACFARMethod] = None         # Slot 2
    oscfar:   Optional[OSCFARMethod] = None         # Slot 3
    cusum:    Optional[CUSUMMethod] = None          # Slot 4
    
    # Noise generator (per-node dynamics)
    noise_gen: Optional[NoiseGenerator] = None
    
    # State
    tx_queue: List = field(default_factory=list)
    is_transmitting: bool = False
    backoff_stage: int = 0
    current_cw: int = 8


class MeshNetwork:
    """Mesh network topology and routing"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.nodes: Dict[int, Node] = {}
        self.sink_id = 0
        
        log.subsection("Building Network Topology")
        self._build_topology()
        
    def _build_topology(self):
        """Create random node placement and connectivity"""
        np.random.seed(self.config.seed)
        
        log.info(f"Placing {self.config.num_nodes} nodes in {self.config.area_size}m × "
                f"{self.config.area_size}m area")
        
        # Place sink at center
        center = self.config.area_size / 2
        self.nodes[0] = Node(0, center, center)
        log.debug(f"Sink node placed at ({center:.0f}, {center:.0f})")
        
        # Place other nodes randomly
        for i in range(1, self.config.num_nodes):
            x = np.random.uniform(0, self.config.area_size)
            y = np.random.uniform(0, self.config.area_size)
            self.nodes[i] = Node(i, x, y)
            self.nodes[i].proposed = ProposedMethod(self.config, i)
            if getattr(self.config, 'tsnfa_all_variants', False):
                self.nodes[i].tsnfa_hybrid = ProposedMethod(self.config, i, variant_override='hybrid')
                self.nodes[i].tsnfa_ema = ProposedMethod(self.config, i, variant_override='ema')
            self.nodes[i].lipski = LipskiFFTMethod(
                node_id=i,
                fft_size=self.config.fft_size,
                k=self.config.lipski_k,
                n_bins_min=self.config.lipski_n_bins_min,
                m_cal=self.config.lipski_m_cal,
                slow_update_alpha=self.config.lipski_slow_update_alpha,
                skip_dc=self.config.lipski_skip_dc,
            )
            self.nodes[i].cacfar = CACFARMethod(
                node_id=i,
                n_ref=self.config.cacfar_n_ref,
                n_guard=self.config.cacfar_n_guard,
                p_fa=self.config.cacfar_p_fa,
                k_persistence=self.config.cacfar_k_persistence,
            )
            self.nodes[i].oscfar = OSCFARMethod(
                node_id=i,
                n_ref=self.config.oscfar_n_ref,
                n_guard=self.config.oscfar_n_guard,
                k_rank=self.config.oscfar_k_rank,
                p_fa=self.config.oscfar_p_fa,
                k_persistence=self.config.oscfar_k_persistence,
            )
            self.nodes[i].cusum = CUSUMMethod(
                node_id=i,
                snr_factor=self.config.cusum_snr_factor,
                alpha_fa=self.config.cusum_alpha_fa,
                K_end=self.config.cusum_k_end,
                m_cal_frames=self.config.cusum_m_cal_frames,
            )
        
        log.info(f"Placed {self.config.num_nodes - 1} sensor nodes")
        
        # Build connectivity (unit-disk model)
        log.info(f"Building connectivity with radius={self.config.comm_radius}m")
        positions = np.array([[n.x, n.y] for n in self.nodes.values()])
        distances = cdist(positions, positions)
        
        total_links = 0
        for i in range(self.config.num_nodes):
            for j in range(self.config.num_nodes):
                if i != j and distances[i, j] <= self.config.comm_radius:
                    self.nodes[i].neighbors.append(j)
                    total_links += 1
        
        avg_neighbors = total_links / self.config.num_nodes
        log.info(f"Created {total_links//2} bidirectional links "
                f"(avg {avg_neighbors:.1f} neighbors/node)")
        
        # Compute routing (BFS from sink)
        self._compute_routes()
        
    def _compute_routes(self):
        """Compute minimum-hop routes to sink using BFS"""
        log.info("Computing minimum-hop routes to sink...")
        
        visited = {0}
        queue = [(0, 0)]  # (node_id, hop_count)
        self.nodes[0].hop_count = 0
        self.nodes[0].route_to_sink = []
        
        while queue:
            current, hops = queue.pop(0)
            
            for neighbor in self.nodes[current].neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.nodes[neighbor].hop_count = hops + 1
                    self.nodes[neighbor].route_to_sink = (
                        self.nodes[current].route_to_sink + [current]
                    )
                    queue.append((neighbor, hops + 1))
        
        # Handle disconnected nodes
        disconnected = 0
        for node_id, node in self.nodes.items():
            if node_id not in visited:
                node.hop_count = 999
                node.route_to_sink = []
                disconnected += 1
        
        # Compute hop statistics
        hop_counts = [n.hop_count for n in self.nodes.values() if n.hop_count < 999]
        if hop_counts:
            log.info(f"Routing complete: max_hops={max(hop_counts)}, "
                    f"avg_hops={np.mean(hop_counts):.1f}")
        
        if disconnected > 0:
            log.warning(f"{disconnected} nodes are disconnected from sink!")
    
    def get_propagation_delay(self, from_id: int, to_id: int) -> float:
        """Calculate propagation delay between two nodes"""
        n1, n2 = self.nodes[from_id], self.nodes[to_id]
        distance = np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        return distance * self.config.prop_delay_per_m
    
    def get_transmission_time(self, payload_bits: int) -> float:
        """Calculate transmission time for given payload"""
        return payload_bits / self.config.data_rate
    
    def print_topology_stats(self):
        """Print detailed topology statistics"""
        log.subsection("Network Topology Statistics")
        
        hop_counts = [n.hop_count for n in self.nodes.values() if n.hop_count < 999]
        neighbor_counts = [len(n.neighbors) for n in self.nodes.values()]
        
        log.info(f"Total nodes: {self.config.num_nodes}")
        log.info(f"Connected nodes: {len(hop_counts)}")
        log.info(f"Hop count distribution: min={min(hop_counts)}, max={max(hop_counts)}, "
                f"mean={np.mean(hop_counts):.1f}, median={np.median(hop_counts):.0f}")
        log.info(f"Neighbor count distribution: min={min(neighbor_counts)}, "
                f"max={max(neighbor_counts)}, mean={np.mean(neighbor_counts):.1f}")


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

@dataclass
class TriggerRecord:
    """Record of a trigger event"""
    time: float
    node_id: int
    method: str  # 'proposed' | 'lipski' | 'cacfar' | 'oscfar' | 'cusum'
    is_true_positive: bool
    strength: float
    arrival_time: Optional[float] = None
    latency: Optional[float] = None


class NetworkSimulator:
    """Discrete-event simulation engine"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        log.section(f"Initializing {config.num_nodes}-Node Network Simulator")
        log.info(str(config))
        
        self.network = MeshNetwork(config)
        self.event_queue: List[SimEvent] = []
        self.current_time = 0.0
        
        # Noise state (global baseline - varies slowly)
        self.current_noise_power = config.base_noise_power
        
        # Per-node noise generators
        self._init_noise_generators()
        
        # True event tracking
        self.active_events: Dict[int, float] = {}  # node_id -> event_end_time
        self.true_event_times: Dict[int, List[float]] = defaultdict(list)
        
        # Results tracking - one trigger list per algorithm
        self.proposed_triggers: List[TriggerRecord] = []
        self.tsnfa_hybrid_triggers: List[TriggerRecord] = []
        self.tsnfa_ema_triggers: List[TriggerRecord] = []
        self.lipski_triggers: List[TriggerRecord] = []
        self.cacfar_triggers: List[TriggerRecord] = []
        self.oscfar_triggers: List[TriggerRecord] = []
        self.cusum_triggers: List[TriggerRecord] = []

        # Network stats - per-algorithm byte counters
        self.total_bytes_proposed = 0
        self.total_bytes_tsnfa_hybrid = 0
        self.total_bytes_tsnfa_ema = 0
        self.total_bytes_lipski = 0
        self.total_bytes_cacfar = 0
        self.total_bytes_oscfar = 0
        self.total_bytes_cusum = 0
        self.congestion_events = 0
        self.packet_collisions = 0

        # Channel state
        self.channel_busy_until = 0.0

        # Per-frame strength recording for ROC sweep (only if enabled).
        # Layout: dict of method -> list of (time, node_id, strength) tuples.
        # We use plain Python lists during the run for speed; convert to numpy
        # at result-computation time for memory efficiency.
        self._record_strengths = bool(getattr(config, 'record_strengths', False))
        self.frame_strengths: Dict[str, List[Tuple[float, int, float]]] = {
            'proposed': [], 'tsnfa_hybrid': [], 'tsnfa_ema': [],
            'lipski': [], 'cacfar': [], 'oscfar': [],
            'cusum': [],
        } if self._record_strengths else {}
        
        # Snapshot tracking
        self.snapshots: List[RawDataSnapshot] = []
        self.current_snapshot: Optional[Dict] = None
        self.snapshot_start_time: float = 0.0
        self.next_snapshot_time: float = 0.0 if config.enable_snapshots else float('inf')
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'true_events_started': 0,
            'true_events_ended': 0,
            'noise_updates': 0,
            'tx_attempts': 0,
            'tx_completions': 0
        }
    
    def _init_noise_generators(self):
        """Initialize per-node noise generators"""
        for node_id, node in self.network.nodes.items():
            if node_id != 0:  # Skip sink node
                node.noise_gen = NoiseGenerator(
                    node_id=node_id,
                    config=self.config,
                    seed=self.config.seed + node_id * 1000
                )
        
    def initialize(self):
        """Set up initial events"""
        log.subsection("Initializing Simulation State")
        
        self.event_queue = []
        self.current_time = 0.0
        self.proposed_triggers = []
        self.tsnfa_hybrid_triggers = []
        self.tsnfa_ema_triggers = []
        self.lipski_triggers = []
        self.cacfar_triggers = []
        self.oscfar_triggers = []
        self.cusum_triggers = []
        self.total_bytes_proposed = 0
        self.total_bytes_tsnfa_hybrid = 0
        self.total_bytes_tsnfa_ema = 0
        self.total_bytes_lipski = 0
        self.total_bytes_cacfar = 0
        self.total_bytes_oscfar = 0
        self.total_bytes_cusum = 0
        self.congestion_events = 0
        self.channel_busy_until = 0.0
        self.stats = {k: 0 for k in self.stats}

        # Reset frame strength recording (each Monte Carlo run starts fresh)
        if self._record_strengths:
            for k in self.frame_strengths:
                self.frame_strengths[k] = []

        # Reset node states
        log.info("Resetting node states...")
        for node in self.network.nodes.values():
            if node.proposed:
                node.proposed.reset()
            if node.tsnfa_hybrid:
                node.tsnfa_hybrid.reset()
            if node.tsnfa_ema:
                node.tsnfa_ema.reset()
            if node.lipski:
                node.lipski.reset()
            if node.cacfar:
                node.cacfar.reset()
            if node.oscfar:
                node.oscfar.reset()
            if node.cusum:
                node.cusum.reset()
            node.tx_queue = []
            node.is_transmitting = False
            node.backoff_stage = 0
            node.current_cw = self.config.cw_min
        
        # Schedule initial frame processing for all nodes
        log.info(f"Scheduling initial frames for {self.config.num_nodes - 1} sensor nodes...")
        for node_id in range(1, self.config.num_nodes):
            offset = np.random.uniform(0, self.config.frame_duration)
            heapq.heappush(self.event_queue, SimEvent(
                time=offset,
                event_type=EventType.FRAME_READY,
                node_id=node_id
            ))
        
        # Schedule true events (Poisson process)
        self._schedule_true_events()
        
        # Schedule noise updates
        heapq.heappush(self.event_queue, SimEvent(
            time=60.0,
            event_type=EventType.NOISE_UPDATE,
            node_id=-1
        ))
        
        log.info(f"Event queue initialized with {len(self.event_queue)} events")
    
    def _schedule_true_events(self):
        """Generate Poisson-distributed true events for all nodes"""
        log.info(f"Scheduling true events (rate={self.config.event_rate}/hr/node)...")
        
        rate_per_second = self.config.event_rate / 3600.0
        total_events = 0
        
        for node_id in range(1, self.config.num_nodes):
            t = 0.0
            node_events = 0
            while t < self.config.simulation_duration:
                t += np.random.exponential(1.0 / rate_per_second)
                if t < self.config.simulation_duration:
                    self.true_event_times[node_id].append(t)
                    heapq.heappush(self.event_queue, SimEvent(
                        time=t,
                        event_type=EventType.TRUE_EVENT_START,
                        node_id=node_id
                    ))
                    node_events += 1
                    total_events += 1
        
        expected = (self.config.num_nodes - 1) * self.config.event_rate * \
                   (self.config.simulation_duration / 3600)
        log.info(f"Scheduled {total_events} true events (expected: {expected:.0f})")
    
    def _generate_frame_samples(self, node_id: int) -> np.ndarray:
        """Generate signal samples for one frame using per-node noise generator
        
        Noise model with frequency separation for human movement detection:
        ─────────────────────────────────────────────────────────────────────
        FAST NOISE (always present, OUTSIDE 1-5 Hz band):
        - EMI: 50/60 Hz power line + harmonics
        - Digital: kHz range switching noise
        → Filtered out by FFT band selection
        
        ENVIRONMENTAL NOISE (per-node varying, may overlap event band):
        - Rain: Broadband + low-frequency rumble
        - Wind: 0.5-5 Hz pressure fluctuations (IN event band!)
        - Motors: Base freq outside band, but imbalance harmonics may be inside
        
        TRUE EVENTS (human movement, 1-5 Hz):
        - Occur in 1-5 Hz band
        - Consistent over > γ_d samples
        - Low frequency, slow changes
        """
        n_samples = self.config.fft_size
        t = np.linspace(0, self.config.frame_duration, n_samples)
        
        # Get node's noise generator
        node = self.network.nodes.get(node_id)
        
        if node and node.noise_gen:
            # Update environmental state based on time elapsed
            node.noise_gen.update_environmental_state(
                dt=self.config.frame_duration, 
                current_time=self.current_time
            )
            
            # Generate noise using per-node generator
            signal = node.noise_gen.generate_noise(n_samples, t, self.current_noise_power)
        else:
            # Fallback: simple white noise
            signal = np.random.randn(n_samples) * np.sqrt(self.current_noise_power)
        
        # Add TRUE EVENT signal (human movement) - in 1-5 Hz band
        if node_id in self.active_events:
            # Event frequency in monitored band (1-5 Hz for human movement)
            freq_margin = 0.2  # Hz margin from band edges
            event_freq = np.random.uniform(
                self.config.event_freq_low + freq_margin,
                self.config.event_freq_high - freq_margin
            )
            
            event_amplitude = np.sqrt(self.current_noise_power * 
                                     10**(self.config.event_snr / 10))
            
            # Human movement is a slow, sustained change (not a sharp transient)
            event_signal = event_amplitude * np.sin(2 * np.pi * event_freq * t)
            envelope = np.exp(-t / self.config.event_decay_tau)
            signal += event_signal * envelope
        
        return signal
    
    def _is_true_positive(self, node_id: int, trigger_time: float) -> bool:
        """Check if trigger corresponds to a true event
        
        A trigger is a true positive if it occurs within a window around a true event.
        The window accounts for:
        1. Event duration (event_duration)
        2. γ_d averaging tail (gamma_d * frame_duration)
        3. Some margin for timing jitter
        """
        # Window extends from event start to (event_end + γ_d tail)
        margin = self.config.frame_duration  # Small margin for timing jitter
        window_before = margin  # Can trigger slightly before event starts
        window_after = self.config.event_duration + self.config.gamma_d * self.config.frame_duration + margin
        
        for event_time in self.true_event_times[node_id]:
            if (event_time - window_before) <= trigger_time <= (event_time + window_after):
                return True
        return False
    
    # =========================================================================
    # SNAPSHOT COLLECTION
    # =========================================================================
    
    def _should_collect_snapshot(self, node_id: int) -> bool:
        """Check if we should collect data for this node"""
        if self.config.snapshot_nodes == 'ALL':
            return True
        elif isinstance(self.config.snapshot_nodes, list):
            return node_id in self.config.snapshot_nodes
        return False
    
    def _start_snapshot(self):
        """Start a new snapshot collection period"""
        self.snapshot_start_time = self.current_time
        self.current_snapshot = {}
        
        # Initialize data structures for each node
        for node_id in range(1, self.config.num_nodes):
            if self._should_collect_snapshot(node_id):
                node = self.network.nodes.get(node_id)
                self.current_snapshot[node_id] = {
                    'samples': [],
                    'timestamps': [],
                    'events': [],
                    'triggers_proposed': [],
                    'triggers_tsnfa_hybrid': [],
                    'triggers_tsnfa_ema': [],
                    'triggers_lipski': [],
                    'triggers_cacfar': [],
                    'triggers_oscfar': [],
                    'triggers_cusum': [],
                    'noise_state': node.noise_gen.get_state() if node and node.noise_gen else {}
                }
        
        log.info(f"Snapshot started at t={self.current_time:.1f}s, "
                f"collecting {len(self.current_snapshot)} nodes for {self.config.snapshot_duration}s")
    
    def _collect_snapshot_data(self, node_id: int, samples: np.ndarray, node: 'Node'):
        """Collect frame data for snapshot"""
        if node_id not in self.current_snapshot:
            return
        
        # Store samples and timestamp
        self.current_snapshot[node_id]['samples'].extend(samples.tolist())
        self.current_snapshot[node_id]['timestamps'].append(self.current_time)
        
        # Track active events
        if node_id in self.active_events:
            self.current_snapshot[node_id]['events'].append({
                'time': self.current_time,
                'type': 'active'
            })
    
    def _end_snapshot(self):
        """End current snapshot and save"""
        if self.current_snapshot is None:
            return

        trigger_keys = ['triggers_proposed', 'triggers_tsnfa_hybrid', 'triggers_tsnfa_ema',
                        'triggers_lipski', 'triggers_cacfar',
                        'triggers_oscfar', 'triggers_cusum']

        # Create snapshot object
        snapshot = RawDataSnapshot(
            timestamp=self.snapshot_start_time,
            duration=self.config.snapshot_duration,
            sample_rate=self.config.sample_rate,
            node_data={
                node_id: {
                    'samples': np.array(data['samples']),
                    'events': data['events'],
                    'noise_state': data['noise_state'],
                    **{k: data.get(k, []) for k in trigger_keys},
                }
                for node_id, data in self.current_snapshot.items()
            }
        )

        snapshot_idx = len(self.snapshots)
        self.snapshots.append(snapshot)

        # Calculate size for logging
        total_samples = sum(len(data['samples']) for data in self.current_snapshot.values())
        log.info(f"Snapshot completed at t={self.current_time:.1f}s: "
                f"{len(self.current_snapshot)} nodes, {total_samples:,} samples")

        # CONTINUOUS SAVE: Save snapshot immediately to disk
        if self.config.continuous_save and self.config.snapshot_output_dir:
            self._save_single_snapshot(snapshot, snapshot_idx)

        self.current_snapshot = None

    def _save_single_snapshot(self, snapshot: RawDataSnapshot, idx: int):
        """Save a single snapshot immediately to disk"""
        try:
            output_dir = self.config.snapshot_output_dir
            os.makedirs(output_dir, exist_ok=True)

            # Save as individual .npz file
            filename = os.path.join(output_dir, f"snapshot_{idx:04d}_t{snapshot.timestamp:.0f}s.npz")

            trigger_keys = ['triggers_proposed', 'triggers_tsnfa_hybrid', 'triggers_tsnfa_ema',
                        'triggers_lipski', 'triggers_cacfar',
                            'triggers_oscfar', 'triggers_cusum']

            arrays = {}
            for node_id, data in snapshot.node_data.items():
                arrays[f"node{node_id}_samples"] = data['samples']
                for tk in trigger_keys:
                    arrays[f"node{node_id}_{tk}"] = np.array(data.get(tk, []))
            
            # Add metadata
            arrays['_metadata'] = np.array([snapshot.timestamp, snapshot.duration, snapshot.sample_rate])
            
            np.savez_compressed(filename, **arrays)
            log.info(f"  → Saved snapshot {idx} to {filename}")
            
        except Exception as e:
            log.info(f"  ! Warning: Could not save snapshot {idx}: {e}")
    
    def _check_snapshot_timing(self):
        """Check if we need to start/end snapshot collection"""
        if not self.config.enable_snapshots:
            return
        
        # Check if we should start a new snapshot
        if self.current_snapshot is None and self.current_time >= self.next_snapshot_time:
            self._start_snapshot()
            self.next_snapshot_time = self.current_time + self.config.snapshot_interval
        
        # Check if current snapshot should end
        if self.current_snapshot is not None:
            elapsed = self.current_time - self.snapshot_start_time
            if elapsed >= self.config.snapshot_duration:
                self._end_snapshot()
    
    def save_snapshots(self, filename: str):
        """Save collected snapshots to file
        
        Snapshots are saved as compressed numpy archives (.npz) for efficiency,
        with a companion JSON file for metadata.
        """
        if not self.snapshots:
            log.info("No snapshots to save")
            return
        
        import gzip
        
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        # Save metadata as JSON
        metadata = {
            'num_snapshots': len(self.snapshots),
            'sample_rate': self.config.sample_rate,
            'snapshot_duration': self.config.snapshot_duration,
            'snapshot_interval': self.config.snapshot_interval,
            'num_nodes': self.config.num_nodes,
            'snapshots': []
        }
        
        for i, snap in enumerate(self.snapshots):
            snap_meta = {
                'index': i,
                'timestamp': snap.timestamp,
                'duration': snap.duration,
                'nodes': list(snap.node_data.keys()),
                'samples_per_node': {
                    str(nid): len(data['samples']) 
                    for nid, data in snap.node_data.items()
                }
            }
            metadata['snapshots'].append(snap_meta)
        
        meta_filename = f"{base_name}_snapshots_meta.json"
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save raw data as compressed numpy archive
        data_filename = f"{base_name}_snapshots_data.npz"
        
        arrays_to_save = {}
        trigger_keys = ['triggers_proposed', 'triggers_tsnfa_hybrid', 'triggers_tsnfa_ema',
                        'triggers_lipski', 'triggers_cacfar',
                        'triggers_oscfar', 'triggers_cusum']
        for i, snap in enumerate(self.snapshots):
            for node_id, data in snap.node_data.items():
                key = f"snap{i}_node{node_id}_samples"
                arrays_to_save[key] = np.array(data['samples'])

                # Save events and triggers as arrays too
                key_events = f"snap{i}_node{node_id}_events"
                arrays_to_save[key_events] = np.array([
                    e['time'] for e in data.get('events', [])
                ]) if data.get('events') else np.array([])

                for tk in trigger_keys:
                    arrays_to_save[f"snap{i}_node{node_id}_{tk}"] = np.array(
                        data.get(tk, [])
                    )
        
        np.savez_compressed(data_filename, **arrays_to_save)
        
        # Calculate file sizes
        meta_size = os.path.getsize(meta_filename)
        data_size = os.path.getsize(data_filename)
        total_samples = sum(
            len(data['samples']) 
            for snap in self.snapshots 
            for data in snap.node_data.values()
        )
        
        log.info(f"Snapshots saved:")
        log.info(f"  Metadata: {meta_filename} ({meta_size/1024:.1f} KB)")
        log.info(f"  Data: {data_filename} ({data_size/1024/1024:.2f} MB)")
        log.info(f"  Total: {len(self.snapshots)} snapshots, {total_samples:,} samples")
        
        return meta_filename, data_filename

    def _update_noise_power(self):
        """Update time-varying noise power"""
        old_power = self.current_noise_power
        phase = 2 * np.pi * self.current_time / self.config.noise_cycle_period
        variation_db = self.config.noise_variation_db * np.sin(phase)
        variation_db += np.random.uniform(-1, 1)
        self.current_noise_power = self.config.base_noise_power * 10**(variation_db / 10)
        
        self.stats['noise_updates'] += 1
        log.debug(f"Noise power updated: {old_power:.3f} -> {self.current_noise_power:.3f} "
                 f"({variation_db:+.1f} dB)")
    
    def _attempt_transmission(self, node_id: int, method: str, trigger_record: TriggerRecord):
        """Attempt to transmit a trigger packet with CSMA/CA"""
        node = self.network.nodes[node_id]
        self.stats['tx_attempts'] += 1
        
        if node.hop_count == 999:
            log.debug(f"Node {node_id} disconnected, dropping {method} packet")
            return
        
        payload_map = {
            'proposed': self.config.proposed_payload,
            'lipski':   self.config.lipski_payload,
            'cacfar':   self.config.cacfar_payload,
            'oscfar':   self.config.oscfar_payload,
            'cusum':    self.config.cusum_payload,
        }
        payload = payload_map.get(method, self.config.proposed_payload)
        
        if self.current_time < self.channel_busy_until:
            # Channel busy - backoff
            cw = min(self.config.cw_max, self.config.cw_min * (2 ** node.backoff_stage))
            backoff_slots = np.random.randint(0, cw)
            backoff_time = backoff_slots * self.config.slot_time
            
            node.backoff_stage = min(node.backoff_stage + 1, self.config.max_retries)
            self.congestion_events += 1
            
            log.debug(f"Node {node_id} {method}: channel busy, backoff={backoff_time*1000:.2f}ms "
                     f"(stage {node.backoff_stage})")
            
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + backoff_time,
                event_type=EventType.PACKET_TX_START,
                node_id=node_id,
                data={'method': method, 'record': trigger_record, 'retry': True}
            ))
        else:
            # Channel free - transmit
            tx_time = self.network.get_transmission_time(payload)
            self.channel_busy_until = self.current_time + tx_time
            
            log.debug(f"Node {node_id} {method}: transmitting {payload} bits, "
                     f"tx_time={tx_time*1000:.2f}ms")
            
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + tx_time,
                event_type=EventType.PACKET_TX_COMPLETE,
                node_id=node_id,
                data={'method': method, 'record': trigger_record}
            ))
            
            bytes_attr = f'total_bytes_{method}'
            if hasattr(self, bytes_attr):
                setattr(self, bytes_attr, getattr(self, bytes_attr) + payload // 8)
            
            node.backoff_stage = 0
    
    def _propagate_to_sink(self, node_id: int, method: str, trigger_record: TriggerRecord):
        """Propagate packet through mesh to sink"""
        node = self.network.nodes[node_id]
        self.stats['tx_completions'] += 1
        
        if not node.route_to_sink:
            prop_delay = self.network.get_propagation_delay(node_id, 0)
            arrival_time = self.current_time + prop_delay + self.config.processing_delay
            hops = 1
        else:
            total_delay = 0
            current = node_id
            hops = 0
            for next_hop in node.route_to_sink[::-1] + [0]:
                prop_delay = self.network.get_propagation_delay(current, next_hop)
                total_delay += prop_delay + self.config.processing_delay
                
                payload = {'proposed': self.config.proposed_payload,
                           'lipski':   self.config.lipski_payload,
                           'cacfar':   self.config.cacfar_payload,
                           'oscfar':   self.config.oscfar_payload,
                           'cusum':    self.config.cusum_payload,
                          }.get(method, self.config.proposed_payload)
                total_delay += self.network.get_transmission_time(payload)
                
                current = next_hop
                hops += 1
            
            arrival_time = self.current_time + total_delay
        
        trigger_record.arrival_time = arrival_time
        trigger_record.latency = arrival_time - trigger_record.time
        
        log.debug(f"Node {node_id} {method}: packet delivered via {hops} hops, "
                 f"latency={trigger_record.latency*1000:.2f}ms")
    
    def process_event(self, event: SimEvent):
        """Process a single simulation event"""
        self.current_time = event.time
        
        if event.event_type == EventType.FRAME_READY:
            self.stats['frames_processed'] += 1
            node = self.network.nodes[event.node_id]
            samples = self._generate_frame_samples(event.node_id)

            # Collect snapshot data if active
            if self.current_snapshot is not None:
                self._collect_snapshot_data(event.node_id, samples, node)

            # ─────────────────────────────────────────────────────────────
            # Run TSNFA reference (proposed) and all six comparators.
            # All eight detectors share the same input frame, so we keep
            # the implementation uniform via a list of (name, detector,
            # trigger_list_attr, snapshot_key) tuples.
            # ─────────────────────────────────────────────────────────────
            detector_specs = [
                ('proposed', node.proposed, 'proposed_triggers', 'triggers_proposed'),
                ('tsnfa_hybrid', node.tsnfa_hybrid, 'tsnfa_hybrid_triggers', 'triggers_tsnfa_hybrid'),
                ('tsnfa_ema', node.tsnfa_ema, 'tsnfa_ema_triggers', 'triggers_tsnfa_ema'),
                ('lipski',   node.lipski,   'lipski_triggers',   'triggers_lipski'),
                ('cacfar',   node.cacfar,   'cacfar_triggers',   'triggers_cacfar'),
                ('oscfar',   node.oscfar,   'oscfar_triggers',   'triggers_oscfar'),
                ('cusum',    node.cusum,    'cusum_triggers',    'triggers_cusum'),
            ]

            for method_name, detector, trigger_attr, snap_key in detector_specs:
                if detector is None:
                    continue
                trigger, strength = detector.process_frame(
                    samples, self.current_noise_power
                )
                # Record strength for ROC sweep if enabled
                if self._record_strengths:
                    self.frame_strengths[method_name].append(
                        (self.current_time, event.node_id, float(strength))
                    )
                # Snapshot recording
                if (self.current_snapshot is not None
                        and event.node_id in self.current_snapshot):
                    if trigger:
                        self.current_snapshot[event.node_id] \
                            .setdefault(snap_key, []) \
                            .append(self.current_time)
                # Trigger record + transmission attempt
                if trigger:
                    is_tp = self._is_true_positive(event.node_id, self.current_time)
                    record = TriggerRecord(
                        time=self.current_time,
                        node_id=event.node_id,
                        method=method_name,
                        is_true_positive=is_tp,
                        strength=strength,
                    )
                    getattr(self, trigger_attr).append(record)
                    self._attempt_transmission(event.node_id, method_name, record)

            # Schedule next frame
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + self.config.frame_duration,
                event_type=EventType.FRAME_READY,
                node_id=event.node_id
            ))
        
        elif event.event_type == EventType.TRUE_EVENT_START:
            self.stats['true_events_started'] += 1
            self.active_events[event.node_id] = (
                self.current_time + self.config.event_duration)
            
            log.debug(f"TRUE EVENT started at node {event.node_id}, t={self.current_time:.3f}s")
            
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + self.config.event_duration,
                event_type=EventType.TRUE_EVENT_END,
                node_id=event.node_id
            ))
        
        elif event.event_type == EventType.TRUE_EVENT_END:
            self.stats['true_events_ended'] += 1
            if event.node_id in self.active_events:
                del self.active_events[event.node_id]
        
        elif event.event_type == EventType.PACKET_TX_START:
            self._attempt_transmission(
                event.node_id, 
                event.data['method'],
                event.data['record']
            )
        
        elif event.event_type == EventType.PACKET_TX_COMPLETE:
            self._propagate_to_sink(
                event.node_id,
                event.data['method'],
                event.data['record']
            )
        
        elif event.event_type == EventType.NOISE_UPDATE:
            self._update_noise_power()
            heapq.heappush(self.event_queue, SimEvent(
                time=self.current_time + 60.0,
                event_type=EventType.NOISE_UPDATE,
                node_id=-1
            ))
    
    def run(self) -> Dict:
        """Run the simulation"""
        log.section("Running Simulation")
        
        self.initialize()
        self.network.print_topology_stats()
        
        log.subsection("Simulation Progress")
        progress = ProgressTracker(
            self.config.simulation_duration, 
            description="Simulating",
            update_interval=2.0
        )
        
        events_processed = 0
        last_stats_time = 0
        stats_interval = self.config.simulation_duration / 10  # Report every 10%
        
        # Checkpoint tracking
        last_checkpoint_time = 0
        checkpoint_count = 0
        
        while self.event_queue and self.current_time < self.config.simulation_duration:
            event = heapq.heappop(self.event_queue)
            self.process_event(event)
            events_processed += 1
            
            # Check snapshot timing (start/end snapshots)
            self._check_snapshot_timing()
            
            # Update progress bar
            progress.update(self.current_time)
            
            # Periodic detailed stats
            if self.current_time - last_stats_time >= stats_interval:
                last_stats_time = self.current_time
                self._print_interim_stats()
            
            # CHECKPOINT SAVING: Save partial results periodically
            if self.config.continuous_save and self.config.checkpoint_interval > 0:
                if self.current_time - last_checkpoint_time >= self.config.checkpoint_interval:
                    last_checkpoint_time = self.current_time
                    checkpoint_count += 1
                    self._save_checkpoint(checkpoint_count)
        
        # End any active snapshot
        if self.current_snapshot is not None:
            self._end_snapshot()
        
        progress.finish()
        
        log.subsection("Simulation Complete")
        log.info(f"Total events processed: {events_processed:,}")
        log.info(f"Simulation time: {self.current_time/3600:.2f} hours")
        if self.snapshots:
            log.info(f"Collected {len(self.snapshots)} raw data snapshots")
        
        return self._compute_results()
    
    def _save_checkpoint(self, checkpoint_num: int):
        """Save a checkpoint of current results"""
        try:
            output_dir = self.config.snapshot_output_dir or '.'
            os.makedirs(output_dir, exist_ok=True)

            method_pairs = [
                ('proposed', self.proposed_triggers),
                ('tsnfa_h',  self.tsnfa_hybrid_triggers),
                ('tsnfa_e',  self.tsnfa_ema_triggers),
                ('lipski',   self.lipski_triggers),
                ('cacfar',   self.cacfar_triggers),
                ('oscfar',   self.oscfar_triggers),
                ('cusum',    self.cusum_triggers),
            ]
            stat_block = {}
            for name, trig_list in method_pairs:
                tp = sum(1 for t in trig_list if t.is_true_positive)
                fp = len(trig_list) - tp
                stat_block[name] = {'tp': tp, 'fp': fp}

            checkpoint_data = {
                'checkpoint_num': checkpoint_num,
                'sim_time_sec': self.current_time,
                'sim_time_hours': self.current_time / 3600,
                'events_so_far': self.stats['true_events_started'],
                'methods': stat_block,
                'snapshots_collected': len(self.snapshots),
                'congestion_events': self.congestion_events
            }

            filename = os.path.join(output_dir, f"checkpoint_{checkpoint_num:03d}.json")
            with open(filename, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            summary = ", ".join(f"{n}={s['tp']}TP/{s['fp']}FP"
                                for n, s in stat_block.items())
            log.info(f"  Checkpoint {checkpoint_num} saved: "
                    f"t={self.current_time/3600:.1f}h, {summary}")

        except Exception as e:
            log.info(f"  ! Checkpoint save failed: {e}")

        return self._compute_results()

    def _print_interim_stats(self):
        """Print interim statistics during simulation"""
        hours = self.current_time / 3600
        parts = [f't={hours:.1f}h: frames={self.stats["frames_processed"]:,}']
        parts.append(f'events={self.stats["true_events_started"]}')
        method_pairs = [
            ('prop',    self.proposed_triggers),
            ('ts_h',    self.tsnfa_hybrid_triggers),
            ('ts_e',    self.tsnfa_ema_triggers),
            ('lipski',  self.lipski_triggers),
            ('cacfar',  self.cacfar_triggers),
            ('oscfar',  self.oscfar_triggers),
            ('cusum',   self.cusum_triggers),
        ]
        for name, trig_list in method_pairs:
            tp = sum(1 for t in trig_list if t.is_true_positive)
            fp = len(trig_list) - tp
            parts.append(f'{name}={tp}TP/{fp}FP')
        log.info('  ' + ', '.join(parts))

    def _compute_results(self) -> Dict:
        """Compute performance metrics for all methods.

        Computes BOTH event-level and frame-level metrics:

          Event-level (PRIMARY for paper):
            events_detected — count of true events with at least one trigger in
                              their detection window
            event_detection_rate — events_detected / total events
            fp_clusters_outside — distinct false-positive clusters outside
                                  any event window (consecutive triggers
                                  within FP_CLUSTER_WINDOW_SEC are one cluster)
            event_precision — events_detected / (events_detected + fp_clusters)

          Frame-level (per-trigger - reflects bandwidth/redundancy cost):
            true_positives — triggers inside any event window (formerly TP)
            false_positives — triggers outside any event window
            redundancy_factor — TP / events_detected (avg redundant triggers
                                per detected event)
            frame_precision — TP / (TP + FP)

        The 'detection_rate' field at the top of each method's results dict
        is the EVENT-LEVEL detection rate (the headline number).
        """
        log.subsection("Computing Results")

        total_true_events = sum(len(events) for events in self.true_event_times.values())
        log.info(f"Total true events: {total_true_events}")

        hours = self.config.simulation_duration / 3600.0
        num_sensor_nodes = self.config.num_nodes - 1

        # Detection window parameters
        margin = self.config.frame_duration
        window_before = margin
        window_after = self.config.event_duration + self.config.gamma_d * self.config.frame_duration + margin

        # FP clustering parameter: consecutive false positives from the same
        # detector on the same node within this many seconds count as one
        # false-alarm cluster (avoids penalizing chatty detectors per-trigger)
        fp_cluster_window = 5.0  # seconds

        all_trigger_lists = {
            'proposed': self.proposed_triggers,
            'tsnfa_hybrid': self.tsnfa_hybrid_triggers,
            'tsnfa_ema': self.tsnfa_ema_triggers,
            'lipski':   self.lipski_triggers,
            'cacfar':   self.cacfar_triggers,
            'oscfar':   self.oscfar_triggers,
            'cusum':    self.cusum_triggers,
        }

        results = {
            'config': {
                'num_nodes': self.config.num_nodes,
                'duration_hours': hours,
                'event_rate': self.config.event_rate
            },
            'true_events': {
                'total': total_true_events,
                'per_node_per_hour': total_true_events / num_sensor_nodes / hours if num_sensor_nodes > 0 else 0
            },
        }

        bytes_map = {
            'proposed': self.total_bytes_proposed,
            'tsnfa_hybrid': self.total_bytes_tsnfa_hybrid,
            'tsnfa_ema': self.total_bytes_tsnfa_ema,
            'lipski':   self.total_bytes_lipski,
            'cacfar':   self.total_bytes_cacfar,
            'oscfar':   self.total_bytes_oscfar,
            'cusum':    self.total_bytes_cusum,
        }

        # Build a per-node lookup of event windows for fast trigger classification
        # node_id -> list of (window_start, window_end) sorted by window_start
        event_windows_by_node = {}
        for node_id, event_times in self.true_event_times.items():
            windows = sorted([
                (et - window_before, et + window_after) for et in event_times
            ])
            event_windows_by_node[node_id] = windows

        def _trigger_in_any_window(node_id, t):
            """Returns True if trigger time t falls inside any event window."""
            windows = event_windows_by_node.get(node_id, [])
            for ws, we in windows:
                if t > we:
                    continue
                if t < ws:
                    return False  # sorted, so no later window will match
                return True
            return False

        for method_name, trigger_list in all_trigger_lists.items():
            # Frame-level counts (existing TP/FP based on is_true_positive flag)
            tp_frames = sum(1 for t in trigger_list if t.is_true_positive)
            fp_frames = sum(1 for t in trigger_list if not t.is_true_positive)
            latencies = [t.latency for t in trigger_list
                        if t.latency is not None and t.is_true_positive]

            # ─── Event-level scoring ────────────────────────────────────────
            # An event is "detected" if at least one trigger from this method
            # falls inside its window.
            # Pre-bucket triggers by node_id for efficient lookup.
            triggers_by_node = defaultdict(list)
            for tr in trigger_list:
                triggers_by_node[tr.node_id].append(tr.time)
            for nid in triggers_by_node:
                triggers_by_node[nid].sort()

            events_detected = 0
            for node_id, event_times in self.true_event_times.items():
                trig_times = triggers_by_node.get(node_id, [])
                if not trig_times:
                    continue
                trig_arr = np.asarray(trig_times)
                for et in event_times:
                    ws, we = et - window_before, et + window_after
                    # any trigger time in [ws, we]?
                    idx_lo = np.searchsorted(trig_arr, ws, side='left')
                    if idx_lo < len(trig_arr) and trig_arr[idx_lo] <= we:
                        events_detected += 1

            fn_events = total_true_events - events_detected

            # ─── FP clustering ──────────────────────────────────────────────
            # Triggers outside ANY event window, grouped by node, with
            # consecutive triggers within fp_cluster_window seconds
            # collapsed into a single FP cluster.
            fp_clusters = 0
            for node_id, trig_times in triggers_by_node.items():
                last_fp_time = -1e18
                for t in trig_times:
                    if _trigger_in_any_window(node_id, t):
                        continue
                    if (t - last_fp_time) >= fp_cluster_window:
                        fp_clusters += 1
                    last_fp_time = t

            # ─── Derived metrics ────────────────────────────────────────────
            event_detection_rate = (events_detected / total_true_events * 100
                                    if total_true_events > 0 else 0)
            event_precision = (events_detected / (events_detected + fp_clusters) * 100
                               if (events_detected + fp_clusters) > 0 else 0)
            redundancy_factor = (tp_frames / events_detected
                                 if events_detected > 0 else 0)
            frame_precision = (tp_frames / (tp_frames + fp_frames) * 100
                               if (tp_frames + fp_frames) > 0 else 0)
            far_frames_per_hour_per_node = (fp_frames / hours / num_sensor_nodes
                                            if num_sensor_nodes > 0 else 0)
            far_clusters_per_hour_per_node = (fp_clusters / hours / num_sensor_nodes
                                              if num_sensor_nodes > 0 else 0)

            log.info(f"{method_name}: {events_detected}/{total_true_events} events, "
                    f"TP_frames={tp_frames}, FP_frames={fp_frames}, "
                    f"FP_clusters={fp_clusters}, redundancy={redundancy_factor:.1f}")

            results[method_name] = {
                # Trigger counts
                'triggers': len(trigger_list),
                'true_positives': tp_frames,
                'false_positives': fp_frames,
                'false_negatives': fn_events,
                'fp_clusters_outside': fp_clusters,
                # Event-level (HEADLINE)
                'events_detected': events_detected,
                'detection_rate': event_detection_rate,    # = event_detection_rate; kept for backwards compat
                'event_detection_rate': event_detection_rate,
                'event_precision': event_precision,
                # Frame-level (redundancy/cost dimension)
                'redundancy_factor': redundancy_factor,
                'frame_precision': frame_precision,
                'precision': frame_precision,              # backwards compat
                'miss_rate': fn_events / total_true_events * 100 if total_true_events > 0 else 0,
                # FAR variants
                'false_alarm_rate': far_frames_per_hour_per_node,    # backwards compat (per-frame)
                'false_alarm_rate_frames': far_frames_per_hour_per_node,
                'false_alarm_rate_clusters': far_clusters_per_hour_per_node,
                # Latency
                'latency_mean_ms': np.mean(latencies) * 1000 if latencies else 0,
                'latency_median_ms': np.median(latencies) * 1000 if latencies else 0,
                'latency_90th_ms': np.percentile(latencies, 90) * 1000 if len(latencies) >= 2 else (latencies[0] * 1000 if latencies else 0),
                'latency_99th_ms': np.percentile(latencies, 99) * 1000 if len(latencies) >= 2 else (latencies[0] * 1000 if latencies else 0),
                # Network load
                'network_load_bytes_per_hour': bytes_map.get(method_name, 0) / hours,
            }

        results['network'] = {
            'congestion_events': self.congestion_events,
            'congestion_per_day': self.congestion_events * 24 / hours
        }
        results['simulation_stats'] = self.stats
        results['snapshots'] = {
            'count': len(self.snapshots),
            'duration_sec': self.config.snapshot_duration if self.snapshots else 0,
            'interval_sec': self.config.snapshot_interval if self.snapshots else 0
        }

        # ─── ROC sweep (post-hoc, using recorded strengths) ─────────────
        if self._record_strengths:
            log.info("Computing ROC sweep from recorded strengths...")
            results['roc_sweep'] = self._compute_roc_sweep(
                event_windows_by_node=event_windows_by_node,
                num_points=self.config.roc_num_points,
                fp_cluster_window=fp_cluster_window,
                num_sensor_nodes=num_sensor_nodes,
                hours=hours,
            )

        return results

    def _compute_roc_sweep(self, event_windows_by_node: Dict,
                           num_points: int = 25,
                           fp_cluster_window: float = 5.0,
                           num_sensor_nodes: int = 1,
                           hours: float = 1.0) -> Dict:
        """Build ROC curves by post-hoc thresholding of recorded strengths.

        For each detector, we have a list of (time, node_id, strength) tuples
        for every frame. The canonical operating point uses threshold = 1.0
        (strength > 1.0 → trigger). For ROC, we sweep the threshold across a
        log-spaced range and recompute (event_detection_rate, FP_rate) at
        each setting.

        Algorithm (post-hoc, single pass per threshold):
          1. Pre-build a flat list of all events with their windows
          2. For each threshold:
               - select frames where strength > threshold (these are triggers)
               - bucket triggers by node, sort by time
               - for each event: searchsorted to check if any trigger falls
                 in its window
               - for each non-event-window trigger: cluster (5 sec window)
          3. Sort the resulting (FP, DR) pairs by FP and apply upper-envelope
             (Pareto frontier) — guarantees monotonic ROC curve.

        Returns dict: method_name -> {
            'thresholds':           list of threshold values (post-pareto)
            'event_dr':             event detection rates 0-100 at each threshold
            'fp_per_hour_per_node': FP cluster rates at each threshold
            'fp_clusters_total':    total FP cluster counts
        }
        """
        # Build flat event list: (node_id, window_start, window_end)
        all_events = []
        for node_id, event_times in self.true_event_times.items():
            windows = event_windows_by_node.get(node_id, [])
            for et, (ws, we) in zip(event_times, windows):
                all_events.append((node_id, ws, we))
        total_true_events = len(all_events)
        roc_data = {}

        # Build per-node sorted window list once for FP classification:
        # node_id -> sorted list of (ws, we)
        node_windows_sorted = {
            nid: sorted(ws_list)
            for nid, ws_list in event_windows_by_node.items()
        }

        def _trigger_in_any_event_window(node_id, t):
            """True if t lies inside any event window on this node."""
            windows = node_windows_sorted.get(node_id, [])
            # Binary search would be cleaner, but linear scan with early exit
            # is fine since events are sparse (~1 per hour per node).
            for ws, we in windows:
                if t < ws:
                    return False
                if t <= we:
                    return True
            return False

        for method_name, strength_list in self.frame_strengths.items():
            if not strength_list:
                continue

            # Convert to numpy structured array for fast processing
            arr = np.array(strength_list, dtype=[
                ('time', 'f8'), ('node_id', 'i4'), ('strength', 'f8')
            ])

            # Determine threshold range from observed strength distribution
            valid = arr['strength'][arr['strength'] > 0]
            if len(valid) < 10:
                continue

            # Pick log-spaced thresholds spanning the empirical strength range
            # but anchored on quantiles, plus the canonical 1.0
            lo = float(np.quantile(valid, 0.001))
            hi = float(np.quantile(valid, 0.9999))
            if lo <= 0:
                lo = max(1e-6, valid.min())
            if hi <= lo:
                hi = lo * 10

            thresholds = list(np.logspace(np.log10(lo), np.log10(hi), num_points))
            # Always include canonical 1.0 if it falls in range
            if lo < 1.0 < hi:
                thresholds.append(1.0)
            # Add a "very low" threshold (~0 trigger everything) and a "very high"
            # threshold for boundary points
            thresholds.append(lo * 0.1)
            thresholds.append(hi * 10)
            thresholds = sorted(set(float(t) for t in thresholds))

            # ─── Sweep ──────────────────────────────────────────────────────
            event_dr_list = []
            fp_per_hr_list = []
            fp_clusters_list = []

            # Sort the data array by node_id, then time, so we can slice once
            sort_idx = np.lexsort((arr['time'], arr['node_id']))
            arr_sorted = arr[sort_idx]
            node_ids_sorted = arr_sorted['node_id']
            times_sorted = arr_sorted['time']
            strengths_sorted = arr_sorted['strength']

            # Pre-compute slice indices: where each node_id's data starts/ends
            unique_nodes, node_starts = np.unique(node_ids_sorted, return_index=True)
            node_ends = np.concatenate([node_starts[1:], [len(arr_sorted)]])
            node_slices = {
                int(nid): (int(s), int(e))
                for nid, s, e in zip(unique_nodes, node_starts, node_ends)
            }

            for th in thresholds:
                # ─── Per-event detection ──────────────────────────────
                events_det = 0
                for node_id, ws, we in all_events:
                    sl = node_slices.get(node_id)
                    if sl is None:
                        continue
                    s_idx, e_idx = sl
                    # Find trigger times in this node, in [ws, we], with strength > th
                    sub_times = times_sorted[s_idx:e_idx]
                    sub_strengths = strengths_sorted[s_idx:e_idx]
                    # Time range
                    lo_idx = np.searchsorted(sub_times, ws, side='left')
                    hi_idx = np.searchsorted(sub_times, we, side='right')
                    if lo_idx == hi_idx:
                        continue
                    # Within time range, is any strength > th?
                    if (sub_strengths[lo_idx:hi_idx] > th).any():
                        events_det += 1

                # ─── FP cluster counting ──────────────────────────────
                fp_clusters_count = 0
                for node_id, (s_idx, e_idx) in node_slices.items():
                    sub_times = times_sorted[s_idx:e_idx]
                    sub_strengths = strengths_sorted[s_idx:e_idx]
                    trig_mask = sub_strengths > th
                    if not trig_mask.any():
                        continue
                    trig_times_node = sub_times[trig_mask]

                    last_fp_time = -1e18
                    for t in trig_times_node:
                        if _trigger_in_any_event_window(node_id, t):
                            continue
                        if (t - last_fp_time) >= fp_cluster_window:
                            fp_clusters_count += 1
                        last_fp_time = t

                event_dr = (events_det / total_true_events * 100
                            if total_true_events > 0 else 0.0)
                fp_per_hr = (fp_clusters_count / hours / num_sensor_nodes
                             if num_sensor_nodes > 0 else 0.0)
                event_dr_list.append(event_dr)
                fp_per_hr_list.append(fp_per_hr)
                fp_clusters_list.append(fp_clusters_count)

            # ─── Apply upper-envelope (Pareto frontier) ────────────────
            # In a correctly-implemented ROC sweep, lowering the threshold
            # should monotonically increase BOTH DR and FP. Sort by FP, then
            # take the running max of DR — this is the achievable ROC curve.
            # Any "internal" non-monotonic point is dominated by points to
            # its lower-left and is kept as-is (it's a real operating point,
            # but the curve we plot is the upper-left envelope).
            order = np.argsort(fp_per_hr_list)
            fp_sorted = [fp_per_hr_list[i] for i in order]
            dr_sorted = [event_dr_list[i] for i in order]
            th_sorted = [thresholds[i] for i in order]
            fp_clusters_sorted = [fp_clusters_list[i] for i in order]

            # Running max of DR -> upper envelope
            dr_envelope = []
            running_max = -1.0
            for d in dr_sorted:
                if d > running_max:
                    running_max = d
                dr_envelope.append(running_max)

            roc_data[method_name] = {
                'thresholds': th_sorted,
                'event_dr_raw': dr_sorted,           # raw points (debug)
                'event_dr': dr_envelope,             # monotonic envelope (plot this)
                'fp_per_hour_per_node': fp_sorted,
                'fp_clusters_total': fp_clusters_sorted,
            }

            log.info(f"  {method_name}: {len(thresholds)} ROC points, "
                    f"DR range [{min(dr_envelope):.1f}, {max(dr_envelope):.1f}]%, "
                    f"FP range [{min(fp_sorted):.3f}, {max(fp_sorted):.2f}]/hr/node")

        return roc_data

def _mc_worker(args):
    """Run one Monte Carlo replicate in a worker process.

    Returns (run_idx, results, snapshots_or_None, strengths_payload_or_None).
    Seeding is identical to the sequential path (42 + run_idx), so parallel
    and sequential execution produce bit-identical aggregated results.
    Snapshot/strength payloads are captured from run 0 only, matching the
    sequential behaviour.
    """
    import copy
    run_idx, config = args
    cfg = copy.deepcopy(config)
    cfg.seed = 42 + run_idx
    sim = NetworkSimulator(cfg)
    results = sim.run()
    snapshots = sim.snapshots if (run_idx == 0 and sim.snapshots) else None
    strengths = None
    if run_idx == 0 and getattr(sim, '_record_strengths', False):
        strengths = {
            'frame_strengths': dict(sim.frame_strengths),
            'true_event_times': dict(sim.true_event_times),
            'config': {
                'frame_duration': cfg.frame_duration,
                'event_duration': cfg.event_duration,
                'gamma_d': cfg.gamma_d,
                'simulation_duration_sec': cfg.simulation_duration,
                'num_nodes': cfg.num_nodes,
                'event_snr_db': cfg.event_snr,
                'event_rate_per_hour': cfg.event_rate,
                'seed': cfg.seed,
            },
        }
    return run_idx, results, snapshots, strengths


def run_monte_carlo(config: SimulationConfig, num_runs: int = 10,
                    jobs: int = 1) -> Tuple[Dict, Optional[List]]:
    """Run multiple simulations and aggregate results for all methods"""
    log.section(f"Monte Carlo Study: {num_runs} runs")

    all_results = []
    first_run_snapshots = None
    first_run_strengths_payload = None  # for ROC offline-reprocessing

    ALL_METHODS = ['proposed', 'tsnfa_hybrid', 'tsnfa_ema',
                   'lipski', 'cacfar', 'oscfar', 'cusum']

    if jobs > 1:
        # ── Parallel replicates (process pool). Same seeds as sequential. ──
        from concurrent.futures import ProcessPoolExecutor, as_completed
        n_workers = min(jobs, num_runs)
        log.info(f"Running {num_runs} Monte Carlo replicates on "
                 f"{n_workers} worker processes...")
        results_by_idx = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_mc_worker, (r, config)): r
                       for r in range(num_runs)}
            for fut in as_completed(futures):
                run_idx, results, snapshots, strengths = fut.result()
                results_by_idx[run_idx] = results
                if snapshots is not None:
                    first_run_snapshots = snapshots
                    log.info(f"Collected {len(snapshots)} snapshots from first run")
                if strengths is not None:
                    first_run_strengths_payload = strengths
                    log.info("Captured first-run strength data for ROC offline-reprocessing")
                dr_parts = [f"{m}={results[m]['detection_rate']:.1f}%"
                            for m in ALL_METHODS if m in results]
                log.info(f"Run {run_idx + 1}/{num_runs} complete: " + ", ".join(dr_parts))
        all_results = [results_by_idx[r] for r in range(num_runs)]
        num_runs_done = num_runs
    else:
      for run in range(num_runs):
        log.subsection(f"Monte Carlo Run {run + 1}/{num_runs}")
        config.seed = 42 + run
        sim = NetworkSimulator(config)
        results = sim.run()
        all_results.append(results)

        if run == 0 and sim.snapshots:
            first_run_snapshots = sim.snapshots
            log.info(f"Collected {len(first_run_snapshots)} snapshots from first run")

        # Capture strength data + event metadata from the first run only
        # (so we can re-plot the ROC offline without re-simulating)
        if run == 0 and getattr(sim, '_record_strengths', False):
            first_run_strengths_payload = {
                'frame_strengths': dict(sim.frame_strengths),
                'true_event_times': dict(sim.true_event_times),
                'config': {
                    'frame_duration': config.frame_duration,
                    'event_duration': config.event_duration,
                    'gamma_d': config.gamma_d,
                    'simulation_duration_sec': config.simulation_duration,
                    'num_nodes': config.num_nodes,
                    'event_snr_db': config.event_snr,
                    'event_rate_per_hour': config.event_rate,
                    'seed': config.seed,
                },
            }
            log.info(f"Captured first-run strength data for ROC offline-reprocessing")

        dr_parts = []
        for m in ALL_METHODS:
            if m in results:
                dr_parts.append(f"{m}={results[m]['detection_rate']:.1f}%")
        log.info(f"Run {run + 1} complete: " + ", ".join(dr_parts))
    
    log.subsection("Aggregating Results")

    aggregated = {
        'config': all_results[0]['config'],
        'num_runs': num_runs,
        'network': {}
    }

    # Carry ROC sweep data from first run (post-hoc thresholding works
    # on the same physical phenomena across runs - aggregating ROC curves
    # across runs is a separate analysis we can add later if needed)
    if 'roc_sweep' in all_results[0]:
        aggregated['roc_sweep'] = all_results[0]['roc_sweep']

    for method in ALL_METHODS:
        if method not in all_results[0]:
            continue
        aggregated[method] = {}
        for metric in all_results[0][method].keys():
            values = [r[method][metric] for r in all_results]
            # Only aggregate scalar-numeric metrics
            try:
                vals_float = [float(v) for v in values]
            except (TypeError, ValueError):
                # Non-numeric (e.g., nested dicts) - keep first run's value
                aggregated[method][metric] = values[0]
                continue
            aggregated[method][metric] = {
                'mean': float(np.mean(vals_float)),
                'std': float(np.std(vals_float)),
                'min': float(np.min(vals_float)),
                'max': float(np.max(vals_float))
            }

    for metric in all_results[0]['network'].keys():
        values = [r['network'][metric] for r in all_results]
        aggregated['network'][metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }

    return aggregated, first_run_snapshots, first_run_strengths_payload

def format_results_table(results: Dict, title: str) -> str:
    """Format results as ASCII table for all methods"""
    ALL_METHODS = ['proposed', 'tsnfa_hybrid', 'tsnfa_ema',
                   'lipski', 'cacfar', 'oscfar', 'cusum']
    METHOD_LABELS = {
        'proposed': 'TSNFA',
        'tsnfa_hybrid': 'TSNFA-H',
        'tsnfa_ema': 'TSNFA-E',
        'lipski':   'Lipski',
        'cacfar':   'CA-CFAR',
        'oscfar':   'OS-CFAR',
        'cusum':    'CUSUM',
    }
    
    available = [m for m in ALL_METHODS if m in results]
    col_width = 14
    
    lines_out = [
        f"\n{'='*120}",
        f"{title}",
        f"{'='*120}",
        f"Configuration: {results['config']['num_nodes']} nodes, "
        f"{results['config']['duration_hours']:.0f} hours",
        f"{'='*120}",
        "",
    ]
    
    # Header row
    header = f"{'Metric':<32}"
    for m in available:
        header += f" {METHOD_LABELS[m]:>{col_width}}"
    lines_out.append(header)
    lines_out.append(f"{'-'*120}")
    
    def get_val(r, method, metric):
        val = r[method][metric]
        return val['mean'] if isinstance(val, dict) else val
    
    # ─── EVENT-LEVEL METRICS (headline) ────────────────────────────────
    # Events Detected (count)
    row = f"{'Events Detected':<32}"
    for m in available:
        v = get_val(results, m, 'events_detected')
        row += f" {v:>{col_width}.0f}"
    lines_out.append(row)

    # Detection Rate (%) - event level
    row = f"{'Detection Rate (events %)':<32}"
    for m in available:
        v = get_val(results, m, 'detection_rate')
        row += f" {v:>{col_width-1}.1f}%"
    lines_out.append(row)

    # Miss Rate
    row = f"{'Miss Rate (%)':<32}"
    for m in available:
        v = get_val(results, m, 'miss_rate')
        row += f" {v:>{col_width-1}.1f}%"
    lines_out.append(row)

    # FP clusters (event-level FP)
    row = f"{'FP Clusters (outside events)':<32}"
    for m in available:
        v = get_val(results, m, 'fp_clusters_outside')
        row += f" {v:>{col_width}.0f}"
    lines_out.append(row)

    # Event-level precision
    row = f"{'Event Precision (%)':<32}"
    for m in available:
        v = get_val(results, m, 'event_precision')
        row += f" {v:>{col_width-1}.1f}%"
    lines_out.append(row)

    # FAR clusters per hour per node
    row = f"{'FAR clusters (/hr/node)':<32}"
    for m in available:
        v = get_val(results, m, 'false_alarm_rate_clusters')
        row += f" {v:>{col_width}.2f}"
    lines_out.append(row)

    lines_out.append(f"{'-'*120}")

    # ─── FRAME-LEVEL METRICS (redundancy/cost) ──────────────────────────
    # True Positives (frame triggers)
    row = f"{'TP frames':<32}"
    for m in available:
        v = get_val(results, m, 'true_positives')
        row += f" {v:>{col_width}.0f}"
    lines_out.append(row)

    # False Positives (frame triggers)
    row = f"{'FP frames':<32}"
    for m in available:
        v = get_val(results, m, 'false_positives')
        row += f" {v:>{col_width}.0f}"
    lines_out.append(row)

    # Redundancy factor (TP_frames / events_detected)
    row = f"{'Redundancy (TP/event)':<32}"
    for m in available:
        v = get_val(results, m, 'redundancy_factor')
        row += f" {v:>{col_width}.1f}"
    lines_out.append(row)

    # Frame-level FAR
    row = f"{'FAR frames (/hr/node)':<32}"
    for m in available:
        v = get_val(results, m, 'false_alarm_rate_frames')
        row += f" {v:>{col_width}.2f}"
    lines_out.append(row)

    # Frame-level Precision
    row = f"{'Frame Precision (%)':<32}"
    for m in available:
        v = get_val(results, m, 'frame_precision')
        row += f" {v:>{col_width-1}.1f}%"
    lines_out.append(row)

    lines_out.append(f"{'-'*120}")

    # ─── LATENCY ────────────────────────────────────────────────────────
    row = f"{'Mean Latency (ms)':<32}"
    for m in available:
        v = get_val(results, m, 'latency_mean_ms')
        row += f" {v:>{col_width}.1f}"
    lines_out.append(row)

    row = f"{'99th %ile Latency (ms)':<32}"
    for m in available:
        v = get_val(results, m, 'latency_99th_ms')
        row += f" {v:>{col_width}.1f}"
    lines_out.append(row)

    # ─── NETWORK LOAD ───────────────────────────────────────────────────
    def fmt_bytes(b):
        if b > 1e6: return f"{b/1e6:.2f} MB"
        elif b > 1e3: return f"{b/1e3:.1f} kB"
        else: return f"{b:.0f} B"

    row = f"{'Network Load (/hr)':<32}"
    for m in available:
        v = get_val(results, m, 'network_load_bytes_per_hour')
        row += f" {fmt_bytes(v):>{col_width}}"
    lines_out.append(row)

    lines_out.append(f"{'='*120}")

    return '\n'.join(lines_out)


def plot_roc_curves(results: Dict, save_path: str = None):
    """Plot ROC curves (event detection rate vs false-alarm cluster rate)
    for all detectors that have ROC sweep data.

    Returns the matplotlib figure.

    Design choices:
      - X axis: log scale FP clusters per hour per node. The lowest FP rate
        sometimes equals zero (TSNFA) — we floor it at 0.001/hr/node so the
        log axis works, and add a left-side annotation explaining.
      - Y axis: linear, 0-105%, so we can see the top edge clearly.
      - Each detector's canonical operating point (threshold=1.0) is
        highlighted with an enlarged marker.
      - Curves are the upper-envelope (Pareto frontier) so they're
        guaranteed monotonic.
    """
    if 'roc_sweep' not in results:
        log.info("No ROC sweep data in results - skipping ROC plot")
        return None

    log.info("Generating ROC curves plot...")

    METHOD_LABELS = {
        'proposed': 'TSNFA',
        'tsnfa_hybrid': 'TSNFA-H',
        'tsnfa_ema': 'TSNFA-E',
        'lipski':   'Lipski FFT',
        'cacfar':   'CA-CFAR',
        'oscfar':   'OS-CFAR',
        'cusum':    'CUSUM',
    }
    COLORS = {
        'proposed': '#2ecc71', 'tsnfa_hybrid': '#16a085', 'tsnfa_ema': '#7f8c8d', 'lipski': '#3498db', 'cacfar': '#e74c3c',
        'oscfar': '#f39c12', 'cusum': '#9b59b6',
    }
    MARKERS = {
        'proposed': 'o', 'lipski': 's', 'cacfar': '^', 'oscfar': 'D',
        'cusum': 'v',
    }
    DRAW_ORDER = ['lipski', 'cacfar', 'oscfar', 'cusum', 'proposed']

    fig, ax = plt.subplots(figsize=(11, 7))

    # Floor for FP rate of zero (so log axis works)
    FP_FLOOR = 0.001  # one false alarm per 1000 hours per node

    for method in DRAW_ORDER:
        if method not in results['roc_sweep']:
            continue
        data = results['roc_sweep'][method]
        if not data['fp_per_hour_per_node']:
            continue

        fp_arr = np.array(data['fp_per_hour_per_node'], dtype=float)
        dr_arr = np.array(data['event_dr'], dtype=float)
        th_arr = np.array(data['thresholds'], dtype=float)

        # Floor zero FP rates
        fp_plot = np.maximum(fp_arr, FP_FLOOR)

        label = METHOD_LABELS.get(method, method)
        color = COLORS.get(method, '#777777')
        marker = MARKERS.get(method, 'o')

        # Plot the curve (already monotonic via upper-envelope)
        is_proposed = (method == 'proposed')
        ax.plot(fp_plot, dr_arr,
                marker=marker, color=color,
                label=label,
                linewidth=2.5 if is_proposed else 1.8,
                markersize=8 if is_proposed else 6,
                alpha=0.95 if is_proposed else 0.85,
                zorder=10 if is_proposed else 5)

        # Highlight the canonical operating point (threshold closest to 1.0)
        canonical_idx = int(np.argmin(np.abs(th_arr - 1.0)))
        ax.plot(fp_plot[canonical_idx], dr_arr[canonical_idx],
                marker='*', color=color, markersize=18,
                markeredgecolor='black', markeredgewidth=1.5,
                zorder=15)

    ax.set_xscale('log')
    ax.set_xlim([FP_FLOOR * 0.5, 1e3])
    ax.set_ylim([-2, 105])

    ax.set_xlabel('False-alarm cluster rate (per hour per node)', fontsize=12)
    ax.set_ylabel('Event detection rate (%)', fontsize=12)
    ax.set_title('ROC Curves — TSNFA + 4 Classical Comparators\n'
                 '★ marks canonical operating point',
                 fontsize=13, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

    # Annotation for the FP_FLOOR
    ax.annotate('FP=0\n(floored\nto 10⁻³)',
                xy=(FP_FLOOR, 100), xytext=(FP_FLOOR * 0.7, 80),
                fontsize=8, color='gray', ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Reference line at 100% DR for visual comparison
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"ROC figure saved to {save_path}")
    return fig


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_result(results: Dict, num_nodes: int, save_path: str = None, params: Dict = None):
    """Create visualization for a single network size result - all methods"""
    log.info(f"Generating single-result plot for {num_nodes} nodes...")
    
    ALL_METHODS = ['proposed', 'tsnfa_hybrid', 'tsnfa_ema',
                   'lipski', 'cacfar', 'oscfar', 'cusum']
    METHOD_LABELS = {
        'proposed': 'TSNFA',
        'tsnfa_hybrid': 'TSNFA-H',
        'tsnfa_ema': 'TSNFA-E',
        'lipski':   'Lipski',
        'cacfar':   'CA-CFAR',
        'oscfar':   'OS-CFAR',
        'cusum':    'CUSUM',
    }
    COLORS = {
        'proposed': '#2ecc71',  # green - reference TSNFA
        'tsnfa_hybrid': '#16a085',
        'tsnfa_ema': '#7f8c8d',
        'lipski':   '#3498db',  # blue
        'cacfar':   '#e74c3c',  # red
        'oscfar':   '#f39c12',  # orange
        'cusum':    '#9b59b6',  # purple
    }

    available = [m for m in ALL_METHODS if m in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Simulation Results: {num_nodes}-Node Network (TSNFA + 4 Classical Comparators)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    def get_mean(r, method, metric):
        val = r[method][metric]
        return val['mean'] if isinstance(val, dict) else val
    
    def get_std(r, method, metric):
        val = r[method][metric]
        return val.get('std', 0) if isinstance(val, dict) else 0
    
    x = np.arange(len(available))
    width = 0.7
    labels = [METHOD_LABELS[m] for m in available]
    colors = [COLORS[m] for m in available]
    
    metrics = [
        ('detection_rate', 'Detection Rate (%)', '(a) Detection Rate', True),
        ('miss_rate', 'Miss Rate (%)', '(b) Miss Rate', True),
        ('false_alarm_rate', 'FAR (/hr/node)', '(c) False Alarm Rate', False),
        ('precision', 'Precision (%)', '(d) Precision', True),
        ('latency_mean_ms', 'Mean Latency (ms)', '(e) Mean Latency', False),
        ('network_load_bytes_per_hour', 'Network Load (kB/hr)', '(f) Network Load', False),
    ]
    
    for idx, (metric, ylabel, title, is_pct) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        vals = []
        errs = []
        for m in available:
            v = get_mean(results, m, metric)
            if metric == 'network_load_bytes_per_hour':
                v /= 1000  # Convert to kB
            vals.append(v)
            e = get_std(results, m, metric)
            if metric == 'network_load_bytes_per_hour':
                e /= 1000
            errs.append(e)
        
        bars = ax.bar(x, vals, width, color=colors, yerr=errs if any(e > 0 for e in errs) else None, capsize=3)
        if metric == 'network_load_bytes_per_hour':
            ax.set_ylabel('Network Load (kB/hr)')
        else:
            ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        max_val = max(vals) if vals and max(vals) > 0 else 1
        ax.set_ylim([0, max_val * 1.2])
        
        for bar, val in zip(bars, vals):
            fmt = f'{val:.1f}%' if is_pct else f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02, 
                    fmt, ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Figure saved to {save_path}")
    plt.close(fig)
    return fig


def plot_comparison(results_by_size: Dict[int, Dict],
                   save_path: str = None, params: Dict = None):
    """Create comparison visualization across network scales - all methods.
    
    Args:
        results_by_size: dict mapping num_nodes -> aggregated results dict.
                         Bars for each metric are drawn in ascending node-count order.
    """
    log.info("Generating comparison plots...")
    
    ALL_METHODS = ['proposed', 'tsnfa_hybrid', 'tsnfa_ema',
                   'lipski', 'cacfar', 'oscfar', 'cusum']
    METHOD_LABELS = {
        'proposed': 'TSNFA', 'tsnfa_hybrid': 'TSNFA-H', 'tsnfa_ema': 'TSNFA-E', 'lipski':   'Lipski',   'cacfar':   'CA-CFAR',
        'oscfar':   'OS-CFAR', 'cusum':    'CUSUM',
    }
    COLORS = {
        'proposed': '#2ecc71', 'tsnfa_hybrid': '#16a085', 'tsnfa_ema': '#7f8c8d', 'lipski':   '#3498db', 'cacfar':   '#e74c3c',
        'oscfar':   '#f39c12', 'cusum':    '#9b59b6',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Simulation Comparison: Network Scalability (TSNFA + 4 Classical Comparators)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    nodes = sorted(results_by_size.keys())
    all_results = [results_by_size[n] for n in nodes]
    available = [m for m in ALL_METHODS if m in all_results[0]]
    
    def get_mean(r, method, metric):
        val = r[method][metric]
        return val['mean'] if isinstance(val, dict) else val
    
    x = np.arange(len(nodes))
    n_methods = len(available)
    width = 0.8 / n_methods
    
    metrics = [
        ('detection_rate', 'Detection Rate (%)', '(a) Detection Rate'),
        ('miss_rate', 'Miss Rate (%)', '(b) Miss Rate'),
        ('false_alarm_rate', 'FAR (/hr/node)', '(c) False Alarm Rate'),
        ('precision', 'Precision (%)', '(d) Precision'),
        ('latency_99th_ms', '99th %ile Latency (ms)', '(e) 99th Latency'),
        ('network_load_bytes_per_hour', 'Network Load (kB/hr)', '(f) Network Load'),
    ]
    
    for idx, (metric, ylabel, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        for j, m in enumerate(available):
            vals = []
            for r in all_results:
                v = get_mean(r, m, metric)
                if metric == 'network_load_bytes_per_hour':
                    v /= 1000
                vals.append(v)
            
            offset = (j - n_methods/2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, label=METHOD_LABELS[m], 
                   color=COLORS[m], alpha=0.85)
        
        if metric == 'network_load_bytes_per_hour':
            ax.set_ylabel('Network Load (kB/hr)')
        else:
            ax.set_ylabel(ylabel)
        ax.set_xlabel('Number of Nodes')
        ax.set_xticks(x)
        ax.set_xticklabels(nodes)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Figure saved to {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete simulation study using configuration from top of file.

    Optional CLI flags override the module-level constants. Any flag not
    given keeps the file's default. This is what rerun_matrix.py uses.
    """
    global log
    global SIMULATION_PRESET, RUN_10_NODES, RUN_50_NODES, RUN_1000_NODES
    global EVENT_SNR_DB, OUTPUT_DIR, TSNFA_VARIANT, TSNFA_ALL_VARIANTS, MC_JOBS, ZETA, TSNFA_CONFIRM
    global RESULTS_FILENAME, SNAPSHOT_OUTPUT_DIR

    import argparse
    parser = argparse.ArgumentParser(
        description="TSNFA Monte Carlo simulator v1.2 (median-variant TSNFA).",
        add_help=True,
    )
    parser.add_argument('--preset', choices=['FAST', 'MEDIUM', 'ACCURATE', 'OVERNIGHT'],
                        default=None, help="Time preset override")
    parser.add_argument('--run-10', dest='run_10', action='store_true', default=None,
                        help="Run the 10-node network")
    parser.add_argument('--no-run-10', dest='run_10', action='store_false',
                        help="Skip the 10-node network")
    parser.add_argument('--run-50', dest='run_50', action='store_true', default=None,
                        help="Run the 50-node network")
    parser.add_argument('--no-run-50', dest='run_50', action='store_false',
                        help="Skip the 50-node network")
    parser.add_argument('--event-snr-db', type=float, default=None,
                        help="Event SNR in dB override")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory override")
    parser.add_argument('--tsnfa-variant', choices=['median', 'hybrid', 'ema'], default=None,
                        help="TSNFA implementation: 'median' (Alg.1 corrected, v1.2 default) "
                             "or 'ema' (legacy v1.1 regression baseline)")
    parser.add_argument('--tsnfa-all-variants', dest='tsnfa_all_variants',
                        action='store_true', default=None,
                        help='Run median+hybrid+ema TSNFA as parallel slots (paired comparison)')
    parser.add_argument('--no-tsnfa-all-variants', dest='tsnfa_all_variants',
                        action='store_false',
                        help='Run only the --tsnfa-variant slot')
    parser.add_argument('--confirm', type=int, default=None,
                        help='Consecutive above-threshold frames required to declare (hybrid; default 1)')
    parser.add_argument('--zeta', type=float, default=None,
                        help='Threshold coefficient zeta override (default: ZETA constant, 6.0)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow writing into an output dir that already has simulation_results.json')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Worker processes for Monte Carlo replicates '
                             '(default 1 = sequential; max useful = replicate count)')
    args = parser.parse_args()

    if args.preset is not None:        SIMULATION_PRESET = args.preset
    if args.run_10 is not None:        RUN_10_NODES = args.run_10
    if args.run_50 is not None:        RUN_50_NODES = args.run_50
    if args.event_snr_db is not None:  EVENT_SNR_DB = args.event_snr_db
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
        RESULTS_FILENAME = f'{OUTPUT_DIR}/simulation_results.json'
        SNAPSHOT_OUTPUT_DIR = f'{OUTPUT_DIR}/snapshots'
    if args.tsnfa_variant is not None: TSNFA_VARIANT = args.tsnfa_variant
    if args.tsnfa_all_variants is not None: TSNFA_ALL_VARIANTS = args.tsnfa_all_variants
    if args.jobs is not None:               MC_JOBS = max(1, args.jobs)
    if args.zeta is not None:               ZETA = args.zeta
    if args.confirm is not None:            TSNFA_CONFIRM = max(1, args.confirm)
    # Overwrite guard: refuse to clobber an existing completed cell
    _existing = os.path.join(OUTPUT_DIR, 'simulation_results.json')
    if os.path.exists(_existing) and not args.overwrite:
        print(f'ERROR: {_existing} already exists.')
        print('Choose a fresh --output-dir, or pass --overwrite to replace it.')
        return 2

    # Set log level (DEBUG for verbose, INFO for normal, PROGRESS for minimal)
    log = Logger(level='INFO')
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log.section("TSNFA Monte Carlo Simulation Study (Deliverable 2)")
    log.info("Comparing: TSNFA vs Lipski FFT, CA-CFAR, OS-CFAR, CUSUM")
    log.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==========================================================================
    # READ CONFIGURATION FROM TOP OF FILE
    # ==========================================================================
    
    # Map preset name to TimePreset
    preset_map = {
        'FAST': TimePreset.FAST,
        'MEDIUM': TimePreset.MEDIUM,
        'ACCURATE': TimePreset.ACCURATE,
        'OVERNIGHT': TimePreset.OVERNIGHT
    }
    SELECTED_PRESET = preset_map.get(SIMULATION_PRESET.upper(), TimePreset.FAST)
    
    # Show configuration
    log.subsection("Configuration (from top of file)")
    log.info(f"  Preset: {SIMULATION_PRESET}")
    log.info(f"  Network sizes: " + ", ".join([
        "10" if RUN_10_NODES else "",
        "50" if RUN_50_NODES else "",
        "1000" if RUN_1000_NODES else ""
    ]).replace(", ,", ",").strip(", "))
    log.info(f"  TSNFA (proposed): variant={TSNFA_VARIANT}, γ_d={GAMMA_D}, γ_a={GAMMA_A}, ζ={ZETA}")
    log.info(f"  Slot 1 Lipski:    k={LIPSKI_K}, N_bins_min={LIPSKI_N_BINS_MIN}")
    log.info(f"  Slot 2 CA-CFAR:   N_ref={CACFAR_N_REF}, P_fa={CACFAR_P_FA:.0e}")
    log.info(f"  Slot 3 OS-CFAR:   N_ref={OSCFAR_N_REF}, k_rank={OSCFAR_K_RANK}, "
             f"P_fa={OSCFAR_P_FA:.0e}")
    log.info(f"  Slot 4 CUSUM:     SNR_factor={CUSUM_SNR_FACTOR}, "
             f"α_fa={CUSUM_ALPHA_FA:.0e}")
    log.info(f"  Events: {EVENT_RATE}/hr/node, SNR={EVENT_SNR_DB}dB, "
             f"band={EVENT_FREQ_LOW}-{EVENT_FREQ_HIGH}Hz")
    
    # Show available presets
    TimePreset.list_presets()
    log.info(f"\nUsing preset: {SELECTED_PRESET['name']} - {SELECTED_PRESET['description']}")
    
    # ==========================================================================
    # BUILD CONFIGS FOR SELECTED NETWORK SIZES
    # ==========================================================================
    
    configs = {}
    
    # Custom parameters from top of file
    custom_params = {
        'tsnfa_variant': TSNFA_VARIANT,
        'tsnfa_all_variants': TSNFA_ALL_VARIANTS,
        'tsnfa_confirm': TSNFA_CONFIRM,
        'gamma_d': GAMMA_D,
        'gamma_a': GAMMA_A,
        'zeta_k': ZETA,
        'event_rate': EVENT_RATE,
        'event_snr': EVENT_SNR_DB,
        'event_freq_low': EVENT_FREQ_LOW,
        'event_freq_high': EVENT_FREQ_HIGH,
        # Slot 1 - Lipski FFT
        'lipski_k': LIPSKI_K,
        'lipski_n_bins_min': LIPSKI_N_BINS_MIN,
        'lipski_m_cal': LIPSKI_M_CAL,
        'lipski_slow_update_alpha': LIPSKI_SLOW_UPDATE_ALPHA,
        'lipski_skip_dc': LIPSKI_SKIP_DC,
        # Slot 2 - CA-CFAR
        'cacfar_n_ref': CACFAR_N_REF,
        'cacfar_n_guard': CACFAR_N_GUARD,
        'cacfar_p_fa': CACFAR_P_FA,
        'cacfar_k_persistence': CACFAR_K_PERSISTENCE,
        # Slot 3 - OS-CFAR
        'oscfar_n_ref': OSCFAR_N_REF,
        'oscfar_n_guard': OSCFAR_N_GUARD,
        'oscfar_k_rank': OSCFAR_K_RANK,
        'oscfar_p_fa': OSCFAR_P_FA,
        'oscfar_k_persistence': OSCFAR_K_PERSISTENCE,
        # Slot 4 - CUSUM
        'cusum_snr_factor': CUSUM_SNR_FACTOR,
        'cusum_alpha_fa': CUSUM_ALPHA_FA,
        'cusum_k_end': CUSUM_K_END,
        'cusum_m_cal_frames': CUSUM_M_CAL_FRAMES,
        # Snapshot parameters
        'enable_snapshots': ENABLE_SNAPSHOTS,
        'snapshot_duration': SNAPSHOT_DURATION_SEC,
        'snapshot_interval': SNAPSHOT_INTERVAL_SEC,
        'snapshot_nodes': SNAPSHOT_NODES,
        # Continuous saving parameters
        'continuous_save': CONTINUOUS_SAVE,
        'checkpoint_interval': CHECKPOINT_INTERVAL_SEC,
        'snapshot_output_dir': SNAPSHOT_OUTPUT_DIR,
        # Noise model parameters
        'emi_freq': NOISE_EMI_FREQ,
        'env_noise_enabled': NOISE_ENV_ENABLED,
        # ROC sweep
        'record_strengths': RECORD_STRENGTHS and ENABLE_ROC_SWEEP,
        'roc_num_points': ROC_NUM_POINTS,
    }
    
    if RUN_10_NODES:
        configs[10] = SimulationConfig.from_preset(SELECTED_PRESET, num_nodes=10, area_size=300.0, **custom_params)
    if RUN_50_NODES:
        configs[50] = SimulationConfig.from_preset(SELECTED_PRESET, num_nodes=50, area_size=750.0, **custom_params)
    if RUN_1000_NODES:
        configs[1000] = SimulationConfig.from_preset(SELECTED_PRESET, num_nodes=1000, area_size=2500.0, **custom_params)
    
    if not configs:
        log.error("No network sizes selected! Enable at least one: RUN_10_NODES, RUN_50_NODES, or RUN_1000_NODES")
        return
    
    # Adjust Monte Carlo runs based on preset
    num_runs = SELECTED_PRESET['monte_carlo_runs']
    
    results = {}
    
    # Print estimated total runtime
    log.subsection("Estimated Runtimes")
    for num_nodes, config in configs.items():
        estimate = config.estimate_runtime()
        log.info(f"  {num_nodes:4d} nodes × {num_runs} runs: {estimate} per run")
    
    all_snapshots = {}  # Collect snapshots by network size
    all_strengths_payloads = {}   # Collect strength payloads by network size

    for num_nodes, config in configs.items():
        log.section(f"Simulating {num_nodes}-Node Network")
        log.info(str(config))

        mc_results, snapshots, strengths_payload = run_monte_carlo(
            config, num_runs=num_runs, jobs=MC_JOBS)
        results[num_nodes] = mc_results

        if snapshots:
            all_snapshots[num_nodes] = snapshots
        if strengths_payload is not None:
            all_strengths_payloads[num_nodes] = strengths_payload

        # Print results table
        print(format_results_table(results[num_nodes],
                                  f"{num_nodes}-Node Network Results"))

    # Save strength payloads to .npz for offline ROC reprocessing
    if all_strengths_payloads:
        results_base = RESULTS_FILENAME.rsplit('.', 1)[0]
        for num_nodes, payload in all_strengths_payloads.items():
            strengths_path = f"{results_base}_{num_nodes}nodes_strengths.npz"
            try:
                # Convert frame_strengths dict (method -> list of (t, nid, s) tuples)
                # into per-method numpy arrays for compact storage
                arrays_to_save = {}
                for method, tuple_list in payload['frame_strengths'].items():
                    if not tuple_list:
                        continue
                    arr = np.array(tuple_list, dtype=np.float64)
                    arrays_to_save[f"strengths_{method}"] = arr
                # Pack event metadata: per node, the event start times
                for nid, ets in payload['true_event_times'].items():
                    arrays_to_save[f"events_node{nid}"] = np.asarray(ets,
                                                                     dtype=np.float64)
                # Save config as a small JSON-serializable string
                config_json = json.dumps(payload['config'])
                arrays_to_save['_config_json'] = np.array([config_json])
                np.savez_compressed(strengths_path, **arrays_to_save)
                log.info(f"Saved strength data to {strengths_path} "
                        f"({os.path.getsize(strengths_path) / (1<<20):.1f} MB)")
            except Exception as e:
                log.info(f"  ! Failed to save strength data: {e}")
    
    # Save snapshots if any were collected
    if all_snapshots and ENABLE_SNAPSHOTS:
        results_base = RESULTS_FILENAME.rsplit('.', 1)[0]
        
        # Determine output directory (same as results file)
        output_dir = os.path.dirname(RESULTS_FILENAME) or '.'
        
        for num_nodes, snapshots in all_snapshots.items():
            snap_filename = f"{results_base}_{num_nodes}nodes"
            
            # Save metadata
            metadata = {
                'num_snapshots': len(snapshots),
                'sample_rate': configs[num_nodes].sample_rate,
                'snapshot_duration': configs[num_nodes].snapshot_duration,
                'snapshot_interval': configs[num_nodes].snapshot_interval,
                'num_nodes': num_nodes,
                'snapshots': []
            }
            
            arrays_to_save = {}
            for i, snap in enumerate(snapshots):
                snap_meta = {
                    'index': i,
                    'timestamp': snap.timestamp,
                    'duration': snap.duration,
                    'nodes': list(snap.node_data.keys()),
                }
                metadata['snapshots'].append(snap_meta)
                
                for node_id, data in snap.node_data.items():
                    arrays_to_save[f"snap{i}_node{node_id}_samples"] = np.array(data['samples'])
                    for tk in ['triggers_proposed', 'triggers_lipski',
                               'triggers_cacfar', 'triggers_oscfar',
                               'triggers_cusum']:
                        arrays_to_save[f"snap{i}_node{node_id}_{tk}"] = np.array(
                            data.get(tk, []))
            
            meta_file = f"{snap_filename}_snapshots_meta.json"
            data_file = f"{snap_filename}_snapshots_data.npz"
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            np.savez_compressed(data_file, **arrays_to_save)
            
            meta_size = os.path.getsize(meta_file) / 1024
            data_size = os.path.getsize(data_file) / 1024 / 1024
            total_samples = sum(len(d['samples']) for s in snapshots for d in s.node_data.values())
            
            log.info(f"Saved {num_nodes}-node snapshots: {len(snapshots)} snapshots, "
                    f"{total_samples:,} samples ({data_size:.2f} MB)")
            log.info(f"  Metadata: {meta_file}")
            log.info(f"  Data: {data_file}")
    
    # Save results to JSON
    if SAVE_RESULTS:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        # Build output with configuration at top
        output_data = {
            '_simulation_parameters': {
                'preset': SIMULATION_PRESET,
                'proposed_method': {
                    'variant': TSNFA_VARIANT,
                    'confirm': TSNFA_CONFIRM,
                    'gamma_d': GAMMA_D,
                    'gamma_a': GAMMA_A,
                    'zeta': ZETA,
                },
                'comparator_pool': {
                    'slot1_lipski': {
                        'k': LIPSKI_K, 'n_bins_min': LIPSKI_N_BINS_MIN,
                        'm_cal': LIPSKI_M_CAL,
                    },
                    'slot2_cacfar': {
                        'n_ref': CACFAR_N_REF, 'n_guard': CACFAR_N_GUARD,
                        'p_fa': CACFAR_P_FA, 'k_persistence': CACFAR_K_PERSISTENCE,
                    },
                    'slot3_oscfar': {
                        'n_ref': OSCFAR_N_REF, 'n_guard': OSCFAR_N_GUARD,
                        'k_rank': OSCFAR_K_RANK, 'p_fa': OSCFAR_P_FA,
                    },
                    'slot4_cusum': {
                        'snr_factor': CUSUM_SNR_FACTOR, 'alpha_fa': CUSUM_ALPHA_FA,
                        'K_end': CUSUM_K_END,
                    },
                },
                'events': {
                    'rate_per_hour_per_node': EVENT_RATE,
                    'snr_db': EVENT_SNR_DB,
                    'freq_band_hz': [EVENT_FREQ_LOW, EVENT_FREQ_HIGH],
                },
                'network_sizes': {
                    '10_nodes': RUN_10_NODES,
                    '50_nodes': RUN_50_NODES,
                    '1000_nodes': RUN_1000_NODES,
                }
            },
            **results  # Add all results after config
        }
        
        with open(RESULTS_FILENAME, 'w') as f:
            json.dump(convert_numpy(output_data), f, indent=2)
        log.info(f"Results saved to {RESULTS_FILENAME}")
    
    # Generate plot filenames from RESULTS_FILENAME
    results_base = RESULTS_FILENAME.rsplit('.', 1)[0]  # Remove extension
    comparison_plot_path = f"{results_base}_comparison.png"
    single_plot_path = f"{results_base}_results.png"
    roc_plot_path = f"{results_base}_roc.png"

    # Build params dict for plot titles
    plot_params = {
        'proposed': {
            'gamma_d': GAMMA_D,
            'gamma_a': GAMMA_A,
            'zeta': ZETA,
        },
        'events': {
            'rate': EVENT_RATE,
            'snr_db': EVENT_SNR_DB,
        }
    }

    # Generate comparison plot if we have multiple network sizes
    if len(results) >= 2:
        plot_comparison(results,
                       save_path=comparison_plot_path, params=plot_params)
    elif len(results) >= 1:
        # Generate a simple bar chart for single network size
        plot_single_result(list(results.values())[0], list(results.keys())[0],
                          save_path=single_plot_path, params=plot_params)

    # Generate ROC plot if ROC sweep was enabled
    if ENABLE_ROC_SWEEP and len(results) >= 1:
        first_result = list(results.values())[0]
        if 'roc_sweep' in first_result and first_result['roc_sweep']:
            plot_roc_curves(first_result, save_path=roc_plot_path)
    
    # Print final summary
    log.section("FINAL SUMMARY")
    log.info(f"Preset: {SIMULATION_PRESET}")
    log.info(f"Parameters: γ_d={GAMMA_D}, γ_a={GAMMA_A}, ζ={ZETA}")
    
    for num_nodes in sorted(results.keys()):
        r = results[num_nodes]
        log.info(f"\n{num_nodes}-Node Network:")
        ALL_METHODS = ['proposed', 'tsnfa_hybrid', 'tsnfa_ema',
                       'lipski', 'cacfar', 'oscfar', 'cusum']
        METHOD_LABELS = {
            'proposed': 'TSNFA',
            'tsnfa_hybrid': 'TSNFA-H',
            'tsnfa_ema': 'TSNFA-E',
        'tsnfa_hybrid': 'TSNFA-H',
        'tsnfa_ema': 'TSNFA-E',
            'lipski':   'Lipski',
            'cacfar':   'CA-CFAR',
            'oscfar':   'OS-CFAR',
            'cusum':    'CUSUM',
        }
        for m in ALL_METHODS:
            if m in r:
                dr = r[m]['detection_rate']['mean']
                dr_std = r[m]['detection_rate']['std']
                mr = r[m]['miss_rate']['mean']
                far = r[m]['false_alarm_rate']['mean']
                log.info(f"  {METHOD_LABELS[m]:>10}: DR={dr:.1f}% (±{dr_std:.1f}%), "
                        f"MissRate={mr:.1f}%, FAR={far:.2f}/hr/node")
    log.info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()