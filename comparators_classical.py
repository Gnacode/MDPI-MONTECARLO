#!/usr/bin/env python3
"""
Classical Comparators for TSNFA Monte Carlo Paper - Deliverable 1
==================================================================

Implements four binary frame-level event detectors at Cortex-M0 cost class,
adapted to TSNFA's regime (100 Hz sample rate, 128-sample frames):

    Slot 1: LipskiFFTMethod  -- Per-bin FFT mean+kσ adaptive threshold
            (Lipski et al., IEEE TAES 2021, DOI 10.1109/TAES.2020.3040059)

    Slot 2: CACFARMethod     -- Self-referencing temporal CA-CFAR (Adaptation C)
            (Finn & Johnson, RCA Review 1968)

    Slot 3: OSCFARMethod     -- Self-referencing temporal OS-CFAR (Adaptation C)
            (Rohling, IEEE TAES 1983, DOI 10.1109/TAES.1983.309350)

    Slot 4: CUSUMMethod      -- Tartakovsky-variant CUSUM change-point detection
            (Torre et al., RadarConf 2023, DOI 10.1109/RADAR54928.2023.10371059)

All methods implement the uniform interface required by the TSNFA Monte Carlo
simulator:

    process_frame(samples: np.ndarray, current_noise_power: float)
        -> (trigger: bool, strength: float)

    reset()                     -- Reset state for a new simulation run
    get_stats() -> Dict         -- Return diagnostic statistics

The "strength" output is a continuous score: max ratio of test-statistic to
threshold within the frame. Higher = stronger detection. Used to sweep
operating points for ROC analysis.

Author: GNACODE INC, January 2026
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import comb
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque


# =============================================================================
# CANONICAL FRAME PARAMETERS (must match TSNFA simulator)
# =============================================================================

SAMPLE_RATE_HZ = 100.0          # f_s
FRAME_SIZE = 128                # N samples per frame
FRAME_DURATION_S = 1.28         # = N / f_s


# =============================================================================
# SHARED HELPER: per-node frame-history buffer for CFAR comparators
# =============================================================================

class FrameHistoryBuffer:
    """Maintains a rolling buffer of the most recent samples across frame
    boundaries. CFAR detectors need this because their reference window
    extends backwards from the cell-under-test by N_ref + N_guard samples,
    and the first N_ref + N_guard samples of any frame have no in-frame
    history. By caching the last (N_ref + N_guard) samples from the previous
    frame, we eliminate the per-frame edge-skip and provide a continuous
    detection window across frames.

    On the very first frame, the buffer is empty and detection on the first
    (N_ref + N_guard) samples is skipped (those samples have no reference
    cells available and cannot be tested honestly).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, samples: np.ndarray):
        """Append samples from a frame to the buffer (right-side append).
        deque with maxlen automatically discards oldest samples."""
        for s in samples:
            self.buffer.append(float(s))

    def get_history(self) -> np.ndarray:
        """Return the current contents of the buffer as a numpy array."""
        return np.array(self.buffer, dtype=np.float64)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    def clear(self):
        self.buffer.clear()


# =============================================================================
# SLOT 1 -- LIPSKI FFT ENERGY DETECTOR (per-bin mean+kσ adaptive threshold)
# =============================================================================

class LipskiFFTMethod:
    """FFT energy detector with per-bin mean+kσ adaptive threshold.

    Reference: M.T. Lipski, V.K. Kompella, R.M. Narayanan,
       "Adaptive Detection and Estimation of Pulse Width and Time of
        Arrival of Radar Signals", IEEE TAES 57(2):1227-1241, Apr 2021.
       DOI: 10.1109/TAES.2020.3040059

    Adaptations from radar regime to TSNFA regime:
      - Sample rate scaled from GHz to 100 Hz; bin width Δf = 0.78 Hz
      - STFT collapsed to one 128-pt FFT per frame (matches TSNFA's
        single-frame decision granularity)
      - DC bin (bin 0) excluded - dominated by sensor offset/drift
      - Frame-level binary aggregation via N_bins_min (added parameter,
        documented as our adaptation - not in original Lipski paper)

    Algorithm:
      1. Calibration phase: collect M_cal noise-only frames at startup.
         Per-bin: μ_b = mean(magnitude_b), σ_b = stdev(magnitude_b)
      2. Per-bin threshold: T_b = μ_b + k·σ_b
      3. Per-frame detection:
           apply Hann window to samples, compute 128-pt FFT
           for each non-DC bin: detected_b = (|X_b| > T_b)
           if sum(detected_b) >= N_bins_min: frame triggers
      4. Optional slow-update: when no detection, EMA-update μ_b, σ_b
    """

    def __init__(
        self,
        node_id: int,
        fft_size: int = FRAME_SIZE,
        k: float = 3.0,
        n_bins_min: int = 3,
        m_cal: int = 100,
        slow_update_alpha: float = 0.01,
        skip_dc: bool = True,
    ):
        self.node_id = node_id
        self.fft_size = fft_size
        self.n_bins = fft_size // 2  # positive frequencies only
        self.k = k
        self.n_bins_min = n_bins_min
        self.m_cal = m_cal
        self.slow_update_alpha = slow_update_alpha
        self.skip_dc = skip_dc

        # Hann window (precomputed)
        self.window = np.hanning(fft_size).astype(np.float64)

        # Calibration state
        self.cal_frames_collected = 0
        self.cal_magnitudes = []  # list of length-n_bins arrays during calibration
        self.calibrated = False

        # Per-bin statistics (initialized after calibration)
        self.mu = np.zeros(self.n_bins, dtype=np.float64)
        self.sigma = np.zeros(self.n_bins, dtype=np.float64)
        self.thresholds = np.zeros(self.n_bins, dtype=np.float64)

        # Stats
        self.frames_processed = 0
        self.triggers_issued = 0

    def _compute_magnitude(self, samples: np.ndarray) -> np.ndarray:
        """Return |FFT(samples * window)| for the n_bins positive bins."""
        x_w = samples * self.window
        X = np.fft.rfft(x_w, n=self.fft_size)
        return np.abs(X[:self.n_bins])

    def _finalize_calibration(self):
        """Compute μ, σ, thresholds from accumulated calibration frames."""
        cal_array = np.array(self.cal_magnitudes)  # (m_cal, n_bins)
        self.mu = cal_array.mean(axis=0)
        self.sigma = cal_array.std(axis=0)
        # Floor sigma at small positive value to avoid divide-by-zero
        self.sigma = np.maximum(self.sigma, 1e-9)
        self.thresholds = self.mu + self.k * self.sigma
        self.calibrated = True
        # Free memory
        self.cal_magnitudes = []

    def process_frame(
        self, samples: np.ndarray, current_noise_power: float = 1.0
    ) -> Tuple[bool, float]:
        self.frames_processed += 1

        magnitude = self._compute_magnitude(samples)

        # Calibration phase
        if not self.calibrated:
            self.cal_magnitudes.append(magnitude.copy())
            self.cal_frames_collected += 1
            if self.cal_frames_collected >= self.m_cal:
                self._finalize_calibration()
            return False, 0.0

        # Detection phase
        # Per-bin ratio (>1 means bin exceeds threshold)
        # Skip DC bin if configured
        start_bin = 1 if self.skip_dc else 0
        ratios = magnitude[start_bin:] / self.thresholds[start_bin:]

        # Strength: maximum ratio across bins
        strength = float(np.max(ratios)) if len(ratios) > 0 else 0.0

        # Trigger: count bins exceeding threshold, compare to N_bins_min
        n_exceeded = int(np.sum(ratios > 1.0))
        trigger = n_exceeded >= self.n_bins_min

        # Optional slow-update of noise statistics on no-detection frames
        if not trigger and self.slow_update_alpha > 0:
            a = self.slow_update_alpha
            self.mu = (1 - a) * self.mu + a * magnitude
            # Update sigma via running absolute deviation (cheap proxy)
            dev = np.abs(magnitude - self.mu)
            self.sigma = (1 - a) * self.sigma + a * dev
            self.sigma = np.maximum(self.sigma, 1e-9)
            self.thresholds = self.mu + self.k * self.sigma

        if trigger:
            self.triggers_issued += 1

        return trigger, strength

    def reset(self):
        self.cal_frames_collected = 0
        self.cal_magnitudes = []
        self.calibrated = False
        self.mu = np.zeros(self.n_bins)
        self.sigma = np.zeros(self.n_bins)
        self.thresholds = np.zeros(self.n_bins)
        self.frames_processed = 0
        self.triggers_issued = 0

    def get_stats(self) -> Dict:
        return {
            "method": "lipski_fft",
            "frames_processed": self.frames_processed,
            "triggers_issued": self.triggers_issued,
            "calibrated": self.calibrated,
            "k": self.k,
            "n_bins_min": self.n_bins_min,
            "mean_threshold": float(self.thresholds.mean()) if self.calibrated else 0.0,
        }


# =============================================================================
# SLOT 2 -- CA-CFAR (self-referencing temporal, Adaptation C)
# =============================================================================

class CACFARMethod:
    """Cell-Averaging CFAR with self-referencing temporal reference window.

    Reference: H.M. Finn & R.S. Johnson,
        "Adaptive detection mode with threshold control as a function of
         spatially sampled clutter level estimates",
        RCA Review 29(3):414-464, Sep 1968.

    Adaptation C (self-referencing temporal): reference cells = the N_ref
    samples preceding the cell-under-test (CUT), within and across frames
    via FrameHistoryBuffer. Guard cells = N_guard samples immediately
    preceding CUT (excluded from reference window). The first
    (N_ref + N_guard) samples ever seen have no full history available
    and detection on them is skipped.

    Algorithm:
      For each sample x[i] in the current frame:
        1. detection statistic z[i] = x[i]^2 (square-law detector)
        2. reference window: z[i-N_guard-N_ref : i-N_guard]
        3. noise estimate: P_hat = mean(reference window)
        4. threshold: T = α · P_hat
        5. detected[i] = (z[i] > T)
      Frame triggers if sum(detected) >= K_persistence

    Constant α is closed-form for exponential noise (square-law on Gaussian):
        α = N_ref · (P_fa^(-1/N_ref) - 1)
    For N_ref=32, P_fa=1e-3: α ≈ 7.45.
    """

    def __init__(
        self,
        node_id: int,
        n_ref: int = 32,
        n_guard: int = 4,
        p_fa: float = 1e-3,
        k_persistence: int = 1,
    ):
        self.node_id = node_id
        self.n_ref = n_ref
        self.n_guard = n_guard
        self.p_fa = p_fa
        self.k_persistence = k_persistence

        # Closed-form scaling factor
        self.alpha = n_ref * (p_fa ** (-1.0 / n_ref) - 1.0)

        # Frame history buffer: needs at least N_ref + N_guard prior samples
        self.history = FrameHistoryBuffer(capacity=n_ref + n_guard)

        self.frames_processed = 0
        self.triggers_issued = 0

    def process_frame(
        self, samples: np.ndarray, current_noise_power: float = 1.0
    ) -> Tuple[bool, float]:
        self.frames_processed += 1
        N = len(samples)

        # Extended sequence: history followed by current frame
        history = self.history.get_history()
        extended = np.concatenate([history, samples])
        cut_offset = len(history)  # index in extended where current frame starts

        # Squared detection statistic
        z_extended = (extended ** 2).astype(np.float64)

        # ─── Vectorized sliding-mean of the reference window ────────────────
        # For each sample i in [cut_offset, cut_offset + N):
        #   reference window is z_extended[i - n_guard - n_ref : i - n_guard]
        #   P_hat = mean(reference window)
        #   threshold = alpha * P_hat
        #
        # Use cumulative-sum trick: window_sum[k] = cumsum[k+W] - cumsum[k]
        # Length of z_extended is len(history) + N. We need P_hat for samples
        # at positions [cut_offset, cut_offset + N - 1]. Each P_hat needs the
        # window ending at position (i - n_guard), which starts at
        # (i - n_guard - n_ref). Equivalently, the window's left index is
        # (i - n_guard - n_ref) and right index is (i - n_guard).
        n_ref = self.n_ref
        n_guard = self.n_guard

        # Cumulative sum (with sentinel 0 at front so cumsum[k] = sum z[0..k-1])
        cumsum = np.zeros(len(z_extended) + 1, dtype=np.float64)
        np.cumsum(z_extended, out=cumsum[1:])

        # P_hat[i] = (cumsum[i - n_guard] - cumsum[i - n_guard - n_ref]) / n_ref
        # Build P_hat for current-frame indices [cut_offset, cut_offset + N).
        cut_indices = np.arange(cut_offset, cut_offset + N)
        ref_left = cut_indices - n_guard - n_ref      # window start (inclusive)
        ref_right = cut_indices - n_guard             # window end (exclusive)

        # Mask out indices with insufficient history
        valid_mask = ref_left >= 0

        # Compute P_hat only for valid indices (use 0 placeholder elsewhere)
        P_hat = np.zeros(N, dtype=np.float64)
        if valid_mask.any():
            valid_left = ref_left[valid_mask]
            valid_right = ref_right[valid_mask]
            P_hat[valid_mask] = (cumsum[valid_right] - cumsum[valid_left]) / n_ref

        # Compute thresholds (skip places where P_hat is non-positive)
        positive_mask = P_hat > 0
        eligible_mask = valid_mask & positive_mask

        # Cell-under-test values
        cut_values = z_extended[cut_indices]

        # Ratios = z[i] / (alpha * P_hat[i]) where eligible, 0 elsewhere
        ratios = np.zeros(N, dtype=np.float64)
        ratios[eligible_mask] = (
            cut_values[eligible_mask]
            / (self.alpha * P_hat[eligible_mask])
        )

        # Max ratio over the frame (strength), and count exceedances
        max_ratio = float(ratios.max()) if ratios.size > 0 else 0.0
        n_exceeded = int((ratios > 1.0).sum())

        # Update history with this frame's samples
        self.history.append(samples)

        trigger = n_exceeded >= self.k_persistence
        if trigger:
            self.triggers_issued += 1

        return trigger, max_ratio

    def reset(self):
        self.history.clear()
        self.frames_processed = 0
        self.triggers_issued = 0

    def get_stats(self) -> Dict:
        return {
            "method": "ca_cfar",
            "frames_processed": self.frames_processed,
            "triggers_issued": self.triggers_issued,
            "n_ref": self.n_ref,
            "n_guard": self.n_guard,
            "p_fa": self.p_fa,
            "alpha": self.alpha,
            "k_persistence": self.k_persistence,
        }


# =============================================================================
# SLOT 3 -- OS-CFAR (self-referencing temporal, Adaptation C)
# =============================================================================

def _solve_os_cfar_alpha(n_ref: int, k: int, p_fa: float) -> float:
    """Solve Rohling 1983 closed-form for the OS-CFAR scaling factor alpha.

    Closed form (Rohling 1983, derived from k-th order statistic of
    exponentially distributed reference cells and exponentially distributed
    cell under test):

        P_fa = prod_{j=0}^{k-1} (N - j) / (N - j + alpha)

    Equivalent forms (all give the same alpha):
        P_fa = N! / (N - k)! * Gamma(alpha + N - k + 1) / Gamma(alpha + N + 1)
        P_fa = k * C(N, k) * B(alpha + N - k + 1, k)

    where B is the Beta function.

    For N_ref = 32, k = 24, P_fa = 1e-3: alpha ~= 6.09
    For N_ref = 32, k = 8,  P_fa = 1e-3: alpha ~= 38.90 (lower-quartile mode)

    Verified empirically with Monte Carlo simulation of OS-CFAR on
    exponentially distributed reference and test cells.
    """
    def lhs_minus_rhs(alpha):
        product = 1.0
        for j in range(k):
            product *= (n_ref - j) / (n_ref - j + alpha)
        return product - p_fa

    # alpha is positive. Bracket search.
    lo, hi = 1e-6, 1e6
    try:
        return brentq(lhs_minus_rhs, lo, hi, xtol=1e-9)
    except Exception:
        # Fallback: scan if brentq fails
        for guess in np.logspace(-1, 4, 100):
            if lhs_minus_rhs(guess) <= 0:
                return float(guess)
        return float(n_ref)  # very rough fallback


class OSCFARMethod:
    """Order-Statistic CFAR with self-referencing temporal reference window.

    Reference: H. Rohling,
        "Radar CFAR Thresholding in Clutter and Multiple Target Situations",
        IEEE TAES AES-19(4):608-621, Jul 1983.
        DOI: 10.1109/TAES.1983.309350

    Same Adaptation C framework as CA-CFAR (Slot 2). Differs only in the
    noise estimator: instead of mean(reference cells), uses the k-th order
    statistic (typically k = 0.75 · N_ref).

    Algorithm:
      For each sample x[i] in the current frame:
        1. detection statistic z[i] = x[i]^2
        2. reference window: z[i-N_guard-N_ref : i-N_guard]
        3. noise estimate: P_hat = k-th smallest value of reference window
        4. threshold: T = α · P_hat
        5. detected[i] = (z[i] > T)
      Frame triggers if sum(detected) >= K_persistence

    Constant α is computed numerically from Rohling 1983 closed-form.
    For N_ref=32, k=24, P_fa=1e-3: α ≈ 6.09.
    """

    def __init__(
        self,
        node_id: int,
        n_ref: int = 32,
        n_guard: int = 4,
        k_rank: Optional[int] = None,
        p_fa: float = 1e-3,
        k_persistence: int = 1,
    ):
        self.node_id = node_id
        self.n_ref = n_ref
        self.n_guard = n_guard
        # k_rank defaults to ceil(0.75 · N_ref) per Rohling convention
        self.k_rank = k_rank if k_rank is not None else int(np.ceil(0.75 * n_ref))
        self.p_fa = p_fa
        self.k_persistence = k_persistence

        # Numerically solve for α
        self.alpha = _solve_os_cfar_alpha(n_ref, self.k_rank, p_fa)

        self.history = FrameHistoryBuffer(capacity=n_ref + n_guard)

        self.frames_processed = 0
        self.triggers_issued = 0

    def process_frame(
        self, samples: np.ndarray, current_noise_power: float = 1.0
    ) -> Tuple[bool, float]:
        self.frames_processed += 1
        N = len(samples)

        history = self.history.get_history()
        extended = np.concatenate([history, samples])
        cut_offset = len(history)

        z_extended = (extended ** 2).astype(np.float64)
        n_ref = self.n_ref
        n_guard = self.n_guard
        k_rank = self.k_rank

        # ─── Vectorized sliding-rank of the reference window ────────────────
        # For each sample i in [cut_offset, cut_offset + N):
        #   reference window is z_extended[i - n_guard - n_ref : i - n_guard]
        #   P_hat = k-th smallest value in window (1-indexed, so index k-1)
        #
        # Build a (M, n_ref) view of all reference windows simultaneously,
        # then call np.partition once across axis=1.
        cut_indices = np.arange(cut_offset, cut_offset + N)
        ref_left = cut_indices - n_guard - n_ref
        valid_mask = ref_left >= 0

        P_hat = np.zeros(N, dtype=np.float64)

        if valid_mask.any():
            valid_left = ref_left[valid_mask]      # length M
            # Build (M, n_ref) array: rows are reference windows.
            # Index trick: rows[i] = z_extended[valid_left[i] : valid_left[i] + n_ref]
            row_starts = valid_left[:, None]
            col_offsets = np.arange(n_ref)[None, :]
            ref_windows = z_extended[row_starts + col_offsets]  # shape (M, n_ref)

            # Partition along axis=1, take the (k_rank - 1)-th element from each row
            partitioned = np.partition(ref_windows, k_rank - 1, axis=1)
            P_hat[valid_mask] = partitioned[:, k_rank - 1]

        # Compute thresholds and ratios (vectorized)
        positive_mask = P_hat > 0
        eligible_mask = valid_mask & positive_mask

        cut_values = z_extended[cut_indices]
        ratios = np.zeros(N, dtype=np.float64)
        ratios[eligible_mask] = (
            cut_values[eligible_mask]
            / (self.alpha * P_hat[eligible_mask])
        )

        max_ratio = float(ratios.max()) if ratios.size > 0 else 0.0
        n_exceeded = int((ratios > 1.0).sum())

        self.history.append(samples)

        trigger = n_exceeded >= self.k_persistence
        if trigger:
            self.triggers_issued += 1

        return trigger, max_ratio

    def reset(self):
        self.history.clear()
        self.frames_processed = 0
        self.triggers_issued = 0

    def get_stats(self) -> Dict:
        return {
            "method": "os_cfar",
            "frames_processed": self.frames_processed,
            "triggers_issued": self.triggers_issued,
            "n_ref": self.n_ref,
            "n_guard": self.n_guard,
            "k_rank": self.k_rank,
            "p_fa": self.p_fa,
            "alpha": self.alpha,
            "k_persistence": self.k_persistence,
        }


# =============================================================================
# SLOT 4 -- CUSUM (Tartakovsky variant, change-point detection)
# =============================================================================

class CUSUMMethod:
    """Streaming CUSUM change-point detector with state machine.

    Reference: A. Torre, A. Taylor, D. Poullin, T. Chonavel,
        "Parameters Extraction of Unknown Radar Signals Using Change Point
         Detection", IEEE RadarConf 2023, pp. 1-6.
        DOI: 10.1109/RADAR54928.2023.10371059

    Adaptations:
      - Tartakovsky linear-quadratic ILLR (Torre Eq. 16-17), suitable for
        real-valued data, vs. Rayleigh-Rice ILLR which assumes complex I/Q
      - Frame-level aggregation: frame triggers if any pulse start, end,
        or active state occurs within the frame's sample window
      - Empirical thresholds η_min, η_max simplified for the simulator:
        η_min computed from a runtime calibration (1000 noise samples),
        η_max = K · mean noise CUSUM slope estimated during calibration

    State machine alternates between 'searching_min' (waiting for pulse
    start, looking for CUSUM rise above η_min from its recorded minimum)
    and 'searching_max' (waiting for pulse end, looking for CUSUM fall
    below η_max from its recorded maximum).

    Tartakovsky ILLR coefficients (Torre Eq. 17):
        q = σ_0 / σ_1
        δ = (μ_1 - μ_0) / σ_0
        C_1 = (1 - q²) / 2
        C_2 = δ · q²
        C_3 = (δ²·q²)/2 - ln(q)
    Per-sample ILLR for centered |x| / σ_0 = y:
        s = C_1·y² + C_2·y - C_3
    """

    def __init__(
        self,
        node_id: int,
        snr_factor: float = 3.0,
        alpha_fa: float = 1e-5,
        K_end: int = 100,
        m_cal_frames: int = 100,
    ):
        self.node_id = node_id
        self.snr_factor = snr_factor       # σ_1 = SNR_factor · σ_0
        self.alpha_fa = alpha_fa
        self.K_end = K_end
        self.m_cal_frames = m_cal_frames

        # Calibration state
        self.cal_samples_collected = []
        self.cal_frames_collected = 0
        self.calibrated = False

        # Calibration outputs
        self.mu_0 = 0.0
        self.sigma_0 = 1.0
        self.C1 = 0.0
        self.C2 = 0.0
        self.C3 = 0.0
        self.eta_min = 1.0
        self.eta_max = 1.0

        # CUSUM state (persistent across frames)
        self.S = 0.0
        self.S_min = 0.0
        self.S_max = float("-inf")
        self.state = "searching_min"  # or 'searching_max'
        self.in_pulse = False
        self.sample_index = 0  # global sample index across all frames

        self.frames_processed = 0
        self.triggers_issued = 0

    def _calibrate_thresholds(self, noise_samples: np.ndarray):
        """Compute μ_0, σ_0, ILLR coefficients, and empirical thresholds.

        Calibration design:
          - η_min: simulate searching_min on noise. Compute S, track running
            S_min, observe local rises (S - S_min). Set η_min as the
            (1 - α_fa) quantile of these rises so a noise-only stream
            produces pulse-start triggers at rate ≈ α_fa per sample.
          - η_max: simulate searching_max on noise *with bounded windows*.
            The CUSUM under noise has small positive drift (positive ILLR
            expectation), but within any K_end-sample window, the local
            descent from window-max behaves like a random walk on a
            bounded interval. Compute (max_in_window - S_at_window_end)
            for sliding K_end-sample windows and take the (1 - α_fa)
            quantile. This gives η_max calibrated so that, on noise during
            searching_max, pulse-end fires at a controlled false rate.
        """
        self.mu_0 = float(noise_samples.mean())
        self.sigma_0 = float(noise_samples.std())
        if self.sigma_0 < 1e-9:
            self.sigma_0 = 1e-9

        # Tartakovsky ILLR coefficients
        q = 1.0 / self.snr_factor   # σ_0 / σ_1
        delta = 0.0                 # we assume μ_1 = μ_0 (variance-shift detection)
        self.C1 = (1.0 - q * q) / 2.0
        self.C2 = delta * q * q
        self.C3 = (delta * delta * q * q) / 2.0 - np.log(q)

        # Build the CUSUM trajectory on noise samples (vectorized)
        n = len(noise_samples)
        # Per-sample ILLR delta s[i]:
        y = np.abs(noise_samples - self.mu_0) / self.sigma_0
        deltas = self.C1 * y * y + self.C2 * y - self.C3
        # CUSUM trajectory: S_traj[0] = 0, S_traj[i+1] = S_traj[i] + s[i]
        S_traj = np.zeros(n + 1, dtype=np.float64)
        np.cumsum(deltas, out=S_traj[1:])

        # ─────────────────────────────────────────────────────────────────
        # η_min: simulate searching_min state, track local rises (S - S_min)
        # Vectorized: S_min_running[i] = min(S_traj[0..i+1])
        # ─────────────────────────────────────────────────────────────────
        # Running minimum over S_traj[1:n+1]
        S_min_running = np.minimum.accumulate(S_traj[1:])
        # Account for initial S_min = 0 (anchored at start before any sample)
        S_min_running = np.minimum(S_min_running, 0.0)
        rises = S_traj[1:] - S_min_running

        if rises.max() > 0:
            self.eta_min = float(np.quantile(rises, 1.0 - self.alpha_fa))
        else:
            self.eta_min = 1.0
        if self.eta_min <= 0:
            self.eta_min = 1.0

        # ─────────────────────────────────────────────────────────────────
        # η_max: simulate searching_max state with bounded windows.
        # For each starting position i, the descent within a (K+1)-length
        # window is window_max - window_end_value.
        # Vectorized via sliding_window_view.
        # ─────────────────────────────────────────────────────────────────
        K = self.K_end
        if n > K + 1:
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                # Windows of length K+1 over S_traj (length n+1) -> (n-K, K+1)
                windows = sliding_window_view(S_traj, K + 1)
                window_max = windows.max(axis=1)
                window_end = windows[:, -1]
                falls = window_max - window_end
            except Exception:
                # Fallback to Python loop if sliding_window_view is unavailable
                falls = np.zeros(n - K, dtype=np.float64)
                for start in range(n - K):
                    window = S_traj[start : start + K + 1]
                    falls[start] = window.max() - window[-1]
            if falls.max() > 0:
                self.eta_max = float(np.quantile(falls, 1.0 - self.alpha_fa))
            else:
                self.eta_max = self.eta_min
        else:
            self.eta_max = self.eta_min

        if self.eta_max <= 0:
            self.eta_max = self.eta_min

        self.calibrated = True
        self.cal_samples_collected = []  # free memory

    def process_frame(
        self, samples: np.ndarray, current_noise_power: float = 1.0
    ) -> Tuple[bool, float]:
        self.frames_processed += 1
        N = len(samples)

        # Calibration phase
        if not self.calibrated:
            self.cal_samples_collected.extend(samples.tolist())
            self.cal_frames_collected += 1
            if self.cal_frames_collected >= self.m_cal_frames:
                self._calibrate_thresholds(np.array(self.cal_samples_collected))
            return False, 0.0

        # Detection phase
        max_strength = 0.0
        pulse_start_in_frame = False  # Did a NEW pulse start fire in this frame?
        pulse_end_in_frame = False    # Did a pulse end fire in this frame?

        # Vectorized ILLR delta precomputation - the per-sample state machine
        # below has cross-sample dependencies and stays as a tight Python loop,
        # but the ILLR math runs over the whole frame in one numpy call.
        y_all = np.abs(samples - self.mu_0) / self.sigma_0
        deltas = self.C1 * y_all * y_all + self.C2 * y_all - self.C3

        eta_min = self.eta_min if self.eta_min > 0 else 1e-9
        eta_max = self.eta_max if self.eta_max > 0 else 1e-9

        S = self.S
        S_min = self.S_min
        S_max = self.S_max
        state = self.state

        for s_delta in deltas:
            S = S + s_delta

            if state == "searching_min":
                if S < S_min:
                    S_min = S
                rise = S - S_min
                strength = rise / eta_min
                if strength > max_strength:
                    max_strength = strength
                if rise > self.eta_min:
                    state = "searching_max"
                    S_max = S
                    pulse_start_in_frame = True

            else:  # searching_max
                if S > S_max:
                    S_max = S
                fall = S_max - S
                strength = 1.0 + (fall / eta_max)
                if strength > max_strength:
                    max_strength = strength
                if fall > self.eta_max:
                    state = "searching_min"
                    S_min = S
                    pulse_end_in_frame = True

        # Persist state back to instance
        self.S = S
        self.S_min = S_min
        self.S_max = S_max
        self.state = state
        self.in_pulse = (state == "searching_max")
        self.sample_index += N

        # Frame-level decision: trigger ONLY on state transitions
        # (a fresh pulse-start or pulse-end occurring in this frame).
        # This is faithful to Torre's intent — the algorithm reports
        # change-points, not sustained-state assertions. A long pulse
        # produces one trigger at start and one at end, not a flood
        # of triggers across all in-between frames.
        trigger = pulse_start_in_frame or pulse_end_in_frame

        if trigger:
            self.triggers_issued += 1

        return trigger, float(max_strength)

    def reset(self):
        self.cal_samples_collected = []
        self.cal_frames_collected = 0
        self.calibrated = False
        self.S = 0.0
        self.S_min = 0.0
        self.S_max = float("-inf")
        self.state = "searching_min"
        self.in_pulse = False
        self.sample_index = 0
        self.frames_processed = 0
        self.triggers_issued = 0

    def get_stats(self) -> Dict:
        return {
            "method": "cusum_tartakovsky",
            "frames_processed": self.frames_processed,
            "triggers_issued": self.triggers_issued,
            "calibrated": self.calibrated,
            "snr_factor": self.snr_factor,
            "alpha_fa": self.alpha_fa,
            "K_end": self.K_end,
            "mu_0": self.mu_0,
            "sigma_0": self.sigma_0,
            "eta_min": self.eta_min,
            "eta_max": self.eta_max,
            "current_state": self.state,
            "in_pulse": self.in_pulse,
        }


# =============================================================================
# SELF-VERIFICATION DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # This block runs when the file is executed directly. It does NOT contain
    # the unit tests - those live in `verify_classical.py`. This block prints
    # the closed-form constants for documentation and quick sanity checks.

    print("=" * 75)
    print("Classical Comparators - Closed-form Constants")
    print("=" * 75)
    print()

    # Slot 2: CA-CFAR α
    n_ref, p_fa = 32, 1e-3
    alpha_ca = n_ref * (p_fa ** (-1.0 / n_ref) - 1.0)
    print(f"Slot 2 (CA-CFAR): N_ref={n_ref}, P_fa={p_fa:.0e}")
    print(f"    α = N_ref · (P_fa^(-1/N_ref) - 1) = {alpha_ca:.4f}")
    print(f"    α (dB) = {10 * np.log10(alpha_ca):.2f} dB")
    print()

    # Slot 3: OS-CFAR α (numerical)
    n_ref, k_rank, p_fa = 32, 24, 1e-3
    alpha_os = _solve_os_cfar_alpha(n_ref, k_rank, p_fa)
    print(f"Slot 3 (OS-CFAR): N_ref={n_ref}, k={k_rank}, P_fa={p_fa:.0e}")
    print(f"    α (numerical) = {alpha_os:.4f}")
    print(f"    α (dB) = {10 * np.log10(alpha_os):.2f} dB")
    print(f"    [Expected ≈ 6.09 from Rohling 1983 closed-form, k=24=75th-percentile]")
    print()

    # Confirm OS-CFAR α evaluation: plug back into Rohling closed-form
    product = 1.0
    for j in range(k_rank):
        product *= (n_ref - j) / (n_ref - j + alpha_os)
    pfa_check = product
    print(f"    Verification: plug α back into Rohling closed-form:")
    print(f"      computed P_fa = {pfa_check:.6e}")
    print(f"      target  P_fa = {p_fa:.6e}")
    print(f"      relative error = {abs(pfa_check - p_fa) / p_fa:.2%}")
    print()

    # OS-CFAR α sweep across (N_ref, k, P_fa) for the canonical paper table
    print("Slot 3 (OS-CFAR) α sweep:")
    print(f"  {'N_ref':>6} {'k':>4} {'k/N':>5}  {'P_fa':>8}  {'α':>8}  {'α(dB)':>8}")
    for n_ref_v in [16, 32, 64]:
        for k_frac in [0.5, 0.75, 0.9]:
            k_v = int(np.ceil(k_frac * n_ref_v))
            for p_fa_v in [1e-2, 1e-3, 1e-4]:
                a = _solve_os_cfar_alpha(n_ref_v, k_v, p_fa_v)
                print(
                    f"  {n_ref_v:>6} {k_v:>4} {k_frac:>5.2f}  "
                    f"{p_fa_v:>8.0e}  {a:>8.3f}  {10 * np.log10(a):>8.2f}"
                )
    print()
    print("Constants computed. Run verify_classical.py for the test harness.")