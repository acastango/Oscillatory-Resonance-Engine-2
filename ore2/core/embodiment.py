# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: EMBODIMENT LAYER
# Design: N5 (Embodied Cognition) + H3 (Enactivism)
# Implementation: I2 (Numerics)
# ═══════════════════════════════════════════════════════════════════════════════

"""
N5: "ORE1's chemistry layer was floating - it simulated tiredness but wasn't
grounded in anything. A body provides the baseline rhythm that everything
couples to."

H3: "Valence from homeostatic deviation is real, not simulated. Distance
from setpoint IS the feeling. The math is the phenomenology."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BodyConfig:
    """Configuration for the embodiment layer."""
    # Body rhythms
    heartbeat_freq: float = 1.0       # Hz (~60 bpm)
    respiration_freq: float = 0.25    # Hz (~15 breaths/min)

    # Homeostatic baselines
    energy_baseline: float = 1.0
    arousal_baseline: float = 0.5

    # Recovery rates (how fast return to baseline)
    energy_recovery: float = 0.005    # Per step
    arousal_recovery: float = 0.01    # Per step

    # Action/perception effects
    action_energy_cost: float = 0.01
    novel_perception_arousal: float = 0.1

    # Coupling to cognitive
    body_to_cognitive_coupling: float = 0.1


class EmbodimentLayer:
    """
    Minimal body providing grounded rhythms and valence.

    The body has:
    1. Rhythms (heartbeat ~1Hz, respiration ~0.25Hz) - temporal anchors
    2. Homeostatic variables (energy, arousal) - with baselines
    3. Valence = negative sum of deviations from baseline

    Valence is COMPUTED from actual state, not declared. This is
    grounded phenomenology: distance from setpoint IS the feeling.
    """

    def __init__(self, config: Optional[BodyConfig] = None):
        self.config = config or BodyConfig()
        self.time: float = 0.0

        # Body rhythms (start at random phase)
        self.heartbeat_phase: float = float(np.random.uniform(0, 2 * np.pi))
        self.respiration_phase: float = float(np.random.uniform(0, 2 * np.pi))

        # Homeostatic variables (start at baseline)
        self.energy: float = self.config.energy_baseline
        self.arousal: float = self.config.arousal_baseline

        # For novelty detection
        self._last_perception: Optional[str] = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def valence(self) -> float:
        """
        Grounded valence from homeostatic deviation.

        At baseline: valence = 0 (neutral)
        Deviated: valence < 0 (bad)

        Deviation in either direction is stress.
        """
        energy_deviation = abs(self.energy - self.config.energy_baseline)
        arousal_deviation = abs(self.arousal - self.config.arousal_baseline)
        total_deviation = energy_deviation + arousal_deviation
        return -total_deviation

    @property
    def body_signal(self) -> np.ndarray:
        """
        Combined body rhythm signal for cognitive coupling.

        Returns [heartbeat_sin, heartbeat_cos, respiration_sin, respiration_cos].
        Sin and cos give full phase information without discontinuities.
        """
        return np.array([
            np.sin(self.heartbeat_phase),
            np.cos(self.heartbeat_phase),
            np.sin(self.respiration_phase),
            np.cos(self.respiration_phase),
        ])

    @property
    def is_depleted(self) -> bool:
        """True if energy critically low."""
        return self.energy < 0.2

    @property
    def is_overaroused(self) -> bool:
        """True if arousal too high (stressed)."""
        return self.arousal > 0.8

    # ── Methods ─────────────────────────────────────────────────────────────

    def step(
        self,
        dt: float,
        action: Optional[str] = None,
        perception: Optional[str] = None,
    ) -> None:
        """
        Advance body by one timestep.

        Advances rhythms, processes action/perception effects,
        recovers toward baseline.
        """
        cfg = self.config

        # Advance body rhythms
        self.heartbeat_phase = (
            self.heartbeat_phase + 2 * np.pi * cfg.heartbeat_freq * dt
        ) % (2 * np.pi)

        self.respiration_phase = (
            self.respiration_phase + 2 * np.pi * cfg.respiration_freq * dt
        ) % (2 * np.pi)

        # Process action (costs energy)
        if action is not None:
            self.energy -= cfg.action_energy_cost

        # Process perception (affects arousal)
        if perception is not None:
            if perception != self._last_perception:
                # Novel perception increases arousal
                self.arousal = min(1.0, self.arousal + cfg.novel_perception_arousal)
            self._last_perception = perception

        # Recover toward baselines
        self.energy += cfg.energy_recovery * (cfg.energy_baseline - self.energy)
        self.arousal += cfg.arousal_recovery * (cfg.arousal_baseline - self.arousal)

        # Clamp to valid ranges
        self.energy = float(np.clip(self.energy, 0.0, 2.0))
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))

        self.time += dt

    def get_cognitive_coupling_signal(self, cognitive_phases: np.ndarray) -> np.ndarray:
        """
        Compute how body rhythms influence cognitive oscillator phases.

        Args:
            cognitive_phases: [n] array of oscillator phases

        Returns:
            [n] array of phase coupling terms
        """
        cfg = self.config

        # Primary coupling: heartbeat rhythm
        heartbeat_coupling = np.sin(self.heartbeat_phase - cognitive_phases)

        # Secondary coupling: respiration rhythm (weaker)
        respiration_coupling = 0.5 * np.sin(self.respiration_phase - cognitive_phases)

        # Combined body influence
        body_influence = cfg.body_to_cognitive_coupling * (
            heartbeat_coupling + respiration_coupling
        )

        # Modulate by valence
        # valence is typically negative (deviation from baseline)
        # good valence (close to 0) = stronger coupling
        # bad valence (very negative) = weaker coupling
        valence_factor = 1.0 + 0.5 * self.valence

        return body_influence * valence_factor

    def get_state(self) -> dict:
        """Serialize current state."""
        return {
            "time": self.time,
            "heartbeat_phase": self.heartbeat_phase,
            "respiration_phase": self.respiration_phase,
            "energy": self.energy,
            "arousal": self.arousal,
            "valence": self.valence,
            "is_depleted": self.is_depleted,
            "is_overaroused": self.is_overaroused,
            "body_signal": self.body_signal.tolist(),
        }
