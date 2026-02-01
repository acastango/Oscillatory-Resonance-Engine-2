# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: MULTI-SCALE SUBSTRATE
# Design: P1 (Dynamical Systems) + N2 (Oscillation Specialist)
# Implementation: I1 (Systems Architect)
# ═══════════════════════════════════════════════════════════════════════════════

"""
P1: "ORE1 had one timescale. Real brains have nested oscillations - delta
contains theta contains gamma. The nesting creates temporal chunking."

N2: "Theta-gamma coupling is how hippocampus indexes memories. Each gamma
burst within a theta cycle is a 'slot'. This gives us memory indexing for
free."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ore2.core.sparse_oscillator import SparseOscillatorConfig, SparseOscillatorLayer


@dataclass
class MultiScaleConfig:
    """Configuration for a two-scale oscillatory substrate."""
    # Fast scale (gamma-like)
    fast_oscillators: int = 100
    fast_base_freq: float = 40.0       # Hz
    fast_freq_spread: float = 5.0
    fast_coupling: float = 0.5
    fast_activation_threshold: float = 0.4   # Lower = more responsive
    fast_max_active: float = 0.3

    # Slow scale (theta-like)
    slow_oscillators: int = 50
    slow_base_freq: float = 8.0        # Hz
    slow_freq_spread: float = 1.0
    slow_coupling: float = 0.6         # Slightly stronger
    slow_activation_threshold: float = 0.5
    slow_max_active: float = 0.25

    # Nesting
    nesting_ratio: int = 5             # Fast steps per slow step
    cross_scale_coupling: float = 0.3  # Bidirectional influence

    # Timing
    dt_fast: float = 0.001             # 1ms (matches ~1000Hz sampling)
    dt_slow: float = 0.005             # 5ms (nesting_ratio * dt_fast)

    # Strange loop (within slow scale)
    strange_loop_strength: float = 1.0


class MultiScaleSubstrate:
    """
    Two-scale oscillatory substrate with nested dynamics.

    Fast scale (gamma-like, ~40Hz) handles rapid binding and feature
    integration. Slow scale (theta-like, ~8Hz) handles working memory
    and attention.

    Fast oscillators are nested within slow oscillator cycles (theta-gamma
    coupling): each slow step contains multiple fast steps. Cross-scale
    coupling is bidirectional and phase-based.

    A strange loop within the slow scale (model <-> meta-model) creates
    self-reference.
    """

    def __init__(self, config: Optional[MultiScaleConfig] = None):
        self.config = config or MultiScaleConfig()
        self.time: float = 0.0

        # Create scales using SparseOscillatorLayer
        self.fast = SparseOscillatorLayer("fast", self._make_fast_config())
        self.slow = SparseOscillatorLayer("slow", self._make_slow_config())

        # Cross-scale coupling matrices
        self._init_cross_scale_coupling()

        # Strange loop within slow (first half <-> second half)
        self._init_strange_loop()

    def _make_fast_config(self) -> SparseOscillatorConfig:
        cfg = self.config
        return SparseOscillatorConfig(
            n_oscillators=cfg.fast_oscillators,
            base_frequency=cfg.fast_base_freq,
            frequency_spread=cfg.fast_freq_spread,
            internal_coupling=cfg.fast_coupling,
            activation_threshold=cfg.fast_activation_threshold,
            max_active_fraction=cfg.fast_max_active,
        )

    def _make_slow_config(self) -> SparseOscillatorConfig:
        cfg = self.config
        return SparseOscillatorConfig(
            n_oscillators=cfg.slow_oscillators,
            base_frequency=cfg.slow_base_freq,
            frequency_spread=cfg.slow_freq_spread,
            internal_coupling=cfg.slow_coupling,
            activation_threshold=cfg.slow_activation_threshold,
            max_active_fraction=cfg.slow_max_active,
        )

    def _init_cross_scale_coupling(self) -> None:
        cfg = self.config
        n_fast = cfg.fast_oscillators
        n_slow = cfg.slow_oscillators

        # Fast -> Slow: many-to-few convergence
        self.fast_to_slow = (
            np.random.randn(n_slow, n_fast)
            * cfg.cross_scale_coupling / n_fast
        )

        # Slow -> Fast: few-to-many broadcast
        self.slow_to_fast = (
            np.random.randn(n_fast, n_slow)
            * cfg.cross_scale_coupling / n_slow
        )

    def _init_strange_loop(self) -> None:
        """
        Strange loop within slow scale: split into model (first half)
        and meta-model (second half) with bidirectional coupling.
        """
        n_slow = self.config.slow_oscillators
        half = n_slow // 2
        strength = self.config.strange_loop_strength

        self.strange_loop_weights = np.zeros((n_slow, n_slow))

        # Model -> Meta-model (first half -> second half)
        self.strange_loop_weights[half:, :half] = (
            strength * np.random.randn(n_slow - half, half) / half
        )

        # Meta-model -> Model (second half -> first half)
        self.strange_loop_weights[:half, half:] = (
            strength * np.random.randn(half, n_slow - half) / (n_slow - half)
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def global_coherence(self) -> float:
        """Coherence across ALL active oscillators in both scales."""
        phases = []
        if self.fast.n_active > 0:
            phases.extend(self.fast.phases[self.fast.active_mask])
        if self.slow.n_active > 0:
            phases.extend(self.slow.phases[self.slow.active_mask])

        if len(phases) == 0:
            return 0.0

        phases = np.array(phases)
        return min(float(np.abs(np.mean(np.exp(1j * phases)))), 0.999)

    @property
    def cross_scale_coherence(self) -> float:
        """How well the two scales are coupled (theta-gamma nesting quality)."""
        if self.fast.n_active == 0 or self.slow.n_active == 0:
            return 0.0

        fast_mean = np.mean(np.exp(1j * self.fast.phases[self.fast.active_mask]))
        slow_mean = np.mean(np.exp(1j * self.slow.phases[self.slow.active_mask]))

        return float(np.abs(fast_mean * np.conj(slow_mean)))

    @property
    def loop_coherence(self) -> float:
        """Strange loop coherence (model <-> meta-model within slow)."""
        if self.slow.n_active < 4:
            return 0.0

        n = self.slow.n
        half = n // 2

        # Model subpopulation (first half)
        model_mask = np.zeros(n, dtype=bool)
        model_mask[:half] = self.slow.active_mask[:half]

        # Meta-model subpopulation (second half)
        meta_mask = np.zeros(n, dtype=bool)
        meta_mask[half:] = self.slow.active_mask[half:]

        if not np.any(model_mask) or not np.any(meta_mask):
            return 0.0

        model_mean = np.mean(np.exp(1j * self.slow.phases[model_mask]))
        meta_mean = np.mean(np.exp(1j * self.slow.phases[meta_mask]))

        return float(np.abs(model_mean * np.conj(meta_mean)))

    # ── Methods ─────────────────────────────────────────────────────────────

    def step(self) -> None:
        """
        One slow-scale timestep containing multiple fast steps (nesting).

        Sequence:
        1. Fast scale: nesting_ratio steps with slow->fast influence
        2. Slow scale: one step with fast->slow influence + strange loop
        """
        cfg = self.config

        # === FAST SCALE: Multiple steps per slow step ===
        for _i in range(cfg.nesting_ratio):

            # Compute slow -> fast influence
            if self.slow.n_active > 0:
                slow_signal = self.slow.phases * self.slow.active_mask.astype(float)
                fast_input_from_slow = self.slow_to_fast @ slow_signal

                # Phase coupling: sin(slow_influence - fast_phase)
                fast_external = cfg.cross_scale_coupling * np.sin(
                    fast_input_from_slow - self.fast.phases
                )
            else:
                fast_external = None

            # Step fast scale
            self.fast.step(cfg.dt_fast, external_input=fast_external)

        # === SLOW SCALE: One step ===

        # Compute fast -> slow influence
        if self.fast.n_active > 0:
            fast_signal = self.fast.phases * self.fast.active_mask.astype(float)
            slow_input_from_fast = self.fast_to_slow @ fast_signal

            slow_external = cfg.cross_scale_coupling * np.sin(
                slow_input_from_fast - self.slow.phases
            )
        else:
            slow_external = np.zeros(cfg.slow_oscillators)

        # Add strange loop contribution
        slow_signal = self.slow.phases * self.slow.active_mask.astype(float)
        strange_loop_input = self.strange_loop_weights @ slow_signal
        slow_external += np.sin(strange_loop_input - self.slow.phases)

        # Step slow scale
        self.slow.step(cfg.dt_slow, external_input=slow_external)

        # Advance time
        self.time += cfg.dt_slow

    def run(self, duration: float) -> List[dict]:
        """Run for duration seconds, return state history."""
        n_steps = int(duration / self.config.dt_slow)
        history = []

        for i in range(n_steps):
            self.step()

            # Record every 10th step to save memory
            if i % 10 == 0:
                history.append(self.get_state())

        return history

    def stimulate_concept(
        self,
        fast_pattern: np.ndarray,
        slow_pattern: np.ndarray,
        strength: float = 0.5,
    ) -> None:
        """
        Stimulate both scales with a concept.

        fast_pattern: [fast_n] phases representing concept's fast component
        slow_pattern: [slow_n] phases representing concept's slow component
        strength: how strongly to stimulate (0-1)
        """
        self.fast.stimulate_by_similarity(fast_pattern, strength)
        self.slow.stimulate_by_similarity(slow_pattern, strength)

    def sustain_activation(
        self,
        fast_strength: float = 0.3,
        slow_strength: float = 0.15,
    ) -> None:
        """
        Sustain activation of currently active oscillators.

        Unlike stimulate_concept, this doesn't use phase similarity —
        it just boosts already-active oscillators to counteract decay.
        Fast gets a larger boost because it decays faster (nesting ratio
        means fast runs multiple substeps per slow step).
        """
        self.fast.sustain_activation(fast_strength)
        self.slow.sustain_activation(slow_strength)

    def get_state(self) -> dict:
        """Full substrate state."""
        return {
            "time": self.time,
            "fast": self.fast.get_state(),
            "slow": self.slow.get_state(),
            "global_coherence": self.global_coherence,
            "cross_scale_coherence": self.cross_scale_coherence,
            "loop_coherence": self.loop_coherence,
        }
