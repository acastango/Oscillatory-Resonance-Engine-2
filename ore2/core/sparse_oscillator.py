# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: SPARSE OSCILLATOR LAYER
# Design: P1 (Dynamical Systems) | Implementation: I2 (Numerics)
# ═══════════════════════════════════════════════════════════════════════════════

"""
P1: "The brain doesn't fire all neurons constantly. Sparse coding is
fundamental to biological efficiency. We need activation potentials
that gate participation."

I2: "Kuramoto is O(n²) in the coupling computation. If only 10% are active,
we get 100x speedup on that inner loop."
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SparseOscillatorConfig:
    """Configuration for a sparse oscillator population."""
    n_oscillators: int = 100
    base_frequency: float = 1.0        # Hz
    frequency_spread: float = 0.1      # Std dev of natural frequencies
    internal_coupling: float = 0.5     # K in Kuramoto equation
    noise_amplitude: float = 0.01      # Stochastic term

    # Sparse activation parameters
    activation_threshold: float = 0.5  # Above this = active
    activation_decay: float = 0.05     # Decay per step
    max_active_fraction: float = 0.2   # Hard cap on simultaneous activation


class SparseOscillatorLayer:
    """
    Sparse Kuramoto oscillator population.

    Oscillators have two states:
    1. Phase (always exists, 0 to 2pi)
    2. Activation potential (0 to 1, determines if oscillator participates)

    When activation > threshold -> active -> participates in Kuramoto coupling
    When activation < threshold -> dormant -> phase drifts at natural frequency

    This creates content-addressable dynamics: stimulate relevant oscillators,
    they activate and synchronize, coherence emerges, then they decay back.
    """

    def __init__(self, name: str, config: Optional[SparseOscillatorConfig] = None):
        self.name = name
        self.config = config or SparseOscillatorConfig()
        cfg = self.config

        self._n = cfg.n_oscillators
        self.noise_amplitude = cfg.noise_amplitude
        self.activation_threshold = cfg.activation_threshold
        self.activation_decay = cfg.activation_decay
        self.max_active_fraction = cfg.max_active_fraction

        # Phase state: uniform random in [0, 2pi)
        self.phases: np.ndarray = np.random.uniform(0, 2 * np.pi, self._n)

        # Natural frequencies: normal(base_frequency, frequency_spread)
        self.natural_frequencies: np.ndarray = np.random.normal(
            cfg.base_frequency, cfg.frequency_spread, self._n
        )

        # Activation potentials: all zeros (start dormant)
        self.activation_potentials: np.ndarray = np.zeros(self._n)

        # Active mask: all False
        self.active_mask: np.ndarray = np.zeros(self._n, dtype=bool)

        # Internal coupling weights: (K / n) for all pairs, 0 on diagonal
        self.internal_weights: np.ndarray = np.full(
            (self._n, self._n), cfg.internal_coupling / self._n
        )
        np.fill_diagonal(self.internal_weights, 0.0)

        # Step counter
        self._step_count: int = 0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Total number of oscillators."""
        return self._n

    @property
    def n_active(self) -> int:
        """Number of currently active oscillators."""
        return int(np.sum(self.active_mask))

    @property
    def coherence(self) -> float:
        """Kuramoto order parameter r = |<e^{itheta}>| for ACTIVE oscillators."""
        if self.n_active == 0:
            return 0.0
        active_phases = self.phases[self.active_mask]
        r = float(np.abs(np.mean(np.exp(1j * active_phases))))
        return min(r, 0.999)

    @property
    def global_coherence(self) -> float:
        """Order parameter for ALL oscillators."""
        return min(float(np.abs(np.mean(np.exp(1j * self.phases)))), 0.999)

    @property
    def mean_phase(self) -> float:
        """Mean phase psi = arg(<e^{itheta}>) of active oscillators."""
        if self.n_active == 0:
            return 0.0
        return float(np.angle(np.mean(np.exp(1j * self.phases[self.active_mask]))))

    @property
    def phase_hash(self) -> str:
        """Hash of active phase configuration for Merkle anchoring."""
        if self.n_active == 0:
            return hashlib.sha256(b"empty").hexdigest()[:16]
        active_phases = self.phases[self.active_mask]
        return hashlib.sha256(active_phases.tobytes()).hexdigest()[:16]

    # ── Methods ─────────────────────────────────────────────────────────────

    def stimulate(self, indices: np.ndarray, strengths: np.ndarray) -> None:
        """Raise activation potential of specific oscillators."""
        self.activation_potentials[indices] += strengths
        np.clip(self.activation_potentials, 0, 1, out=self.activation_potentials)
        self._update_active_mask()

    def stimulate_by_similarity(self, reference_phases: np.ndarray, strength: float) -> None:
        """Content-addressable activation: oscillators with phases similar to reference get stimulated."""
        phase_diff = self.phases - reference_phases
        similarity = (np.cos(phase_diff) + 1) / 2  # 0 to 1

        self.activation_potentials += strength * similarity
        np.clip(self.activation_potentials, 0, 1, out=self.activation_potentials)
        self._update_active_mask()

    def sustain_activation(self, strength: float = 0.2) -> None:
        """
        Boost currently active oscillators to counteract decay.

        Unlike stimulate_by_similarity, this does NOT change which
        oscillators are active — it just prevents existing ones from
        decaying below threshold. This keeps the active set stable,
        which stabilizes coherence measurements.
        """
        if self.n_active > 0:
            active_idx = np.where(self.active_mask)[0]
            self.activation_potentials[active_idx] += strength
            np.clip(self.activation_potentials, 0, 1, out=self.activation_potentials)

    def step(self, dt: float, external_input: Optional[np.ndarray] = None) -> None:
        """
        Advance dynamics by one timestep.

        Only active oscillators participate in Kuramoto coupling.
        Dormant oscillators drift at their natural frequency.
        """
        self._step_count += 1

        if self.n_active > 0:
            active_idx = np.where(self.active_mask)[0]
            active_phases = self.phases[active_idx]

            # Phase differences: delta_theta_ij = theta_j - theta_i (active pairs only)
            phase_diff = active_phases[np.newaxis, :] - active_phases[:, np.newaxis]

            # Coupling submatrix (only active-to-active connections)
            active_weights = self.internal_weights[np.ix_(active_idx, active_idx)]

            # Kuramoto coupling: sum_j K_ij sin(theta_j - theta_i)
            coupling = np.sum(active_weights * np.sin(phase_diff), axis=1)

            # Add external input if provided
            if external_input is not None:
                coupling += external_input[active_idx]

            # Noise term
            noise = self.noise_amplitude * np.random.randn(self.n_active)

            # Phase update: dtheta/dt = omega + coupling + noise
            dtheta = self.natural_frequencies[active_idx] + coupling + noise
            self.phases[active_idx] = (self.phases[active_idx] + dt * dtheta) % (2 * np.pi)

        # Dormant oscillators: free-run at natural frequency
        dormant_idx = np.where(~self.active_mask)[0]
        if len(dormant_idx) > 0:
            self.phases[dormant_idx] = (
                self.phases[dormant_idx]
                + dt * self.natural_frequencies[dormant_idx]
            ) % (2 * np.pi)

        # Decay all activation potentials
        self.activation_potentials *= (1 - self.activation_decay)
        self._update_active_mask()

    def get_state(self) -> dict:
        """Serialize current state for persistence."""
        return {
            "name": self.name,
            "n": self._n,
            "n_active": self.n_active,
            "coherence": self.coherence,
            "global_coherence": self.global_coherence,
            "mean_phase": self.mean_phase,
            "phase_hash": self.phase_hash,
            "phases": self.phases.tolist(),
            "activation_potentials": self.activation_potentials.tolist(),
            "active_mask": self.active_mask.tolist(),
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _update_active_mask(self) -> None:
        """Update which oscillators are active based on activation potentials."""
        above_threshold = self.activation_potentials > self.activation_threshold

        # Enforce max active cap
        max_active = int(self._n * self.max_active_fraction)

        if np.sum(above_threshold) > max_active:
            # Keep only the top activations
            top_indices = np.argsort(self.activation_potentials)[-max_active:]
            self.active_mask = np.zeros(self._n, dtype=bool)
            self.active_mask[top_indices] = True
        else:
            self.active_mask = above_threshold
