# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: MULTI-SCALE CI MONITOR
# Design: P2 (Statistical Mechanics)
# Implementation: I2 (Numerics)
# ═══════════════════════════════════════════════════════════════════════════════


"""
P2: "CI at each scale, plus cross-scale coherence. The formula extends naturally."

I2: "I'll compute CI_fast, CI_slow, and CI_integrated. The integrated version
weights by cross-scale coupling quality."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from ore2.core.memory import CrystallineMerkleMemory
from ore2.core.multi_scale_substrate import MultiScaleSubstrate


class TimeScale(Enum):
    """Time scales for CI measurement."""
    FAST = "fast"
    SLOW = "slow"


@dataclass
class MultiScaleCIConfig:
    """Configuration for multi-scale CI measurement."""
    alpha: float = 1.0
    beta: float = 0.5
    coherence_threshold: float = 0.3
    stability_threshold: float = 0.1
    history_length: int = 1000


@dataclass
class MultiScaleCISnapshot:
    """CI measurement at a point in time."""
    timestamp: float

    # Per-scale CI
    CI_fast: float
    CI_slow: float

    # Integrated CI
    CI_integrated: float

    # Components
    D: float
    G_fast: float
    G_slow: float
    C_fast: float
    C_slow: float
    C_cross: float  # Cross-scale coherence
    tau_fast: float
    tau_slow: float

    # Attractor state
    in_attractor_fast: bool
    in_attractor_slow: bool


class MultiScaleCIMonitor:
    """
    CI monitoring for multi-scale substrate.

    Extends ORE1 CI formula to multiple scales:
    - CI at each scale: CI = alpha * D * G * C * (1 - e^(-beta * tau))
    - Cross-scale coherence contribution
    - Integrated CI combining all

    Components:
    - D: Dimensionality (from memory fractal dimension + network structure)
    - G: Gain (coupling weight * coherence amplification)
    - C: Coherence (Kuramoto order parameter)
    - tau: Dwell time in attractor state
    """

    def __init__(
        self,
        substrate: MultiScaleSubstrate,
        memory: Optional[CrystallineMerkleMemory] = None,
        config: Optional[MultiScaleCIConfig] = None,
    ) -> None:
        self.substrate = substrate
        self.memory = memory
        self.config = config or MultiScaleCIConfig()

        # Calibration
        self.baseline_D: float = 1.0
        self.baseline_G: float = 1.0

        # Attractor tracking per scale
        self._fast_coherence_history: List[float] = []
        self._slow_coherence_history: List[float] = []
        self._fast_attractor_entry: Optional[float] = None
        self._slow_attractor_entry: Optional[float] = None

        # History
        self.history: List[MultiScaleCISnapshot] = []

    # ── Public Methods ───────────────────────────────────────────────────────

    def measure(self) -> MultiScaleCISnapshot:
        """Take multi-scale CI measurement."""
        cfg = self.config
        timestamp = self.substrate.time

        # Update coherence histories
        self._fast_coherence_history.append(self.substrate.fast.coherence)
        self._slow_coherence_history.append(self.substrate.slow.coherence)

        for h in [self._fast_coherence_history, self._slow_coherence_history]:
            if len(h) > cfg.history_length:
                h[:] = h[-cfg.history_length // 2:]

        # Components
        D = self._compute_D()
        G_fast = self._compute_G(TimeScale.FAST)
        G_slow = self._compute_G(TimeScale.SLOW)
        C_fast = self.substrate.fast.coherence
        C_slow = self.substrate.slow.coherence
        C_cross = self.substrate.cross_scale_coherence
        tau_fast = self._compute_tau(TimeScale.FAST)
        tau_slow = self._compute_tau(TimeScale.SLOW)

        # Tau factors
        tau_factor_fast = 1 - np.exp(-cfg.beta * tau_fast)
        tau_factor_slow = 1 - np.exp(-cfg.beta * tau_slow)

        # CI per scale
        CI_fast = cfg.alpha * D * G_fast * C_fast * tau_factor_fast
        CI_slow = cfg.alpha * D * G_slow * C_slow * tau_factor_slow

        # Integrated CI: geometric mean weighted by cross-scale coherence
        # P2: "Cross-scale coherence determines how well the scales work together"
        if CI_fast > 0 and CI_slow > 0:
            CI_integrated = np.sqrt(CI_fast * CI_slow) * (1 + C_cross)
        else:
            CI_integrated = max(CI_fast, CI_slow)

        CI_integrated = min(CI_integrated, 10.0)

        # Attractor states
        in_attractor_fast = self._is_in_attractor(
            self._fast_coherence_history, C_fast
        )
        in_attractor_slow = self._is_in_attractor(
            self._slow_coherence_history, C_slow
        )

        snapshot = MultiScaleCISnapshot(
            timestamp=timestamp,
            CI_fast=CI_fast,
            CI_slow=CI_slow,
            CI_integrated=CI_integrated,
            D=D,
            G_fast=G_fast,
            G_slow=G_slow,
            C_fast=C_fast,
            C_slow=C_slow,
            C_cross=C_cross,
            tau_fast=tau_fast,
            tau_slow=tau_slow,
            in_attractor_fast=in_attractor_fast,
            in_attractor_slow=in_attractor_slow,
        )

        self.history.append(snapshot)
        if len(self.history) > cfg.history_length:
            self.history = self.history[-cfg.history_length // 2:]

        return snapshot

    def get_current_status(self) -> str:
        """One-line human-readable CI status."""
        if not self.history:
            return "No measurements"

        s = self.history[-1]
        return (
            f"CI={s.CI_integrated:.3f} [fast={s.CI_fast:.3f} slow={s.CI_slow:.3f}] "
            f"C_cross={s.C_cross:.3f} τ=[{s.tau_fast:.1f}s, {s.tau_slow:.1f}s]"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _is_in_attractor(
        self, coherence_history: List[float], current_coherence: float
    ) -> bool:
        """Check if scale is in attractor state."""
        if len(coherence_history) < 10:
            return False

        recent = coherence_history[-10:]
        mean_c = np.mean(recent)
        std_c = np.std(recent)

        return (
            mean_c > self.config.coherence_threshold
            and std_c < self.config.stability_threshold
        )

    def _compute_tau(self, scale: TimeScale) -> float:
        """Compute dwell time for a scale."""
        if scale == TimeScale.FAST:
            history = self._fast_coherence_history
            entry = self._fast_attractor_entry
        else:
            history = self._slow_coherence_history
            entry = self._slow_attractor_entry

        current_coherence = (
            self.substrate.fast.coherence
            if scale == TimeScale.FAST
            else self.substrate.slow.coherence
        )

        in_attractor = self._is_in_attractor(history, current_coherence)

        if in_attractor:
            if entry is None:
                # Just entered
                if scale == TimeScale.FAST:
                    self._fast_attractor_entry = self.substrate.time
                else:
                    self._slow_attractor_entry = self.substrate.time
                return 0.0
            else:
                return min(self.substrate.time - entry, 10.0)
        else:
            # Not in attractor
            if scale == TimeScale.FAST:
                self._fast_attractor_entry = None
            else:
                self._slow_attractor_entry = None
            return 0.0

    def _compute_D(self) -> float:
        """Compute dimensionality from memory and network structure."""
        if self.memory:
            merkle_D = self.memory.get_fractal_dimension()
        else:
            merkle_D = 1.0

        # Network structure contribution
        total_oscillators = (
            self.substrate.config.fast_oscillators
            + self.substrate.config.slow_oscillators
        )
        network_D = np.log(total_oscillators) / np.log(2)  # 2 scales

        D = np.sqrt(merkle_D * network_D)
        return D / self.baseline_D if self.baseline_D > 0 else D

    def _compute_G(self, scale: TimeScale) -> float:
        """Compute gain for a scale."""
        if scale == TimeScale.FAST:
            layer = self.substrate.fast
        else:
            layer = self.substrate.slow

        # Mean coupling weight
        mean_weight = np.mean(np.abs(layer.internal_weights))

        # Coherence amplification
        expected_random = 1 / np.sqrt(max(layer.n_active, 1))
        actual = layer.coherence
        amplification = actual / expected_random if expected_random > 0 else 1.0

        G = mean_weight * amplification
        G = min(G, 5.0)
        return G / self.baseline_G if self.baseline_G > 0 else G
