# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: DEVELOPMENTAL STAGES
# Design: N7 (Developmental Neuro) + H3 (Enactivism)
# Implementation: I3 (State Management)
# ═══════════════════════════════════════════════════════════════════════════════


"""
N7: "Development isn't optional - it's how identity is earned. Critical periods
exist because the brain NEEDS certain inputs at certain times. Miss the window,
and that learning is harder forever."

H3: "Autonomy is achieved, not given. An entity that starts with founding
memories never had to BECOME itself."
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class DevelopmentStage(Enum):
    """Developmental stages - identity is earned through progression."""
    GENESIS = "genesis"        # Just born, baseline
    BABBLING = "babbling"      # Random exploration
    IMITATION = "imitation"    # Coupling to external
    AUTONOMY = "autonomy"      # Self-generated goals
    MATURE = "mature"          # Stable, still plastic


@dataclass
class CriticalPeriod:
    """Enhanced learning window tied to a developmental stage."""
    name: str                           # e.g., "language_acquisition"
    stage: DevelopmentStage             # When it's active
    learning_type: str                  # What type of learning is enhanced
    sensitivity: float = 2.0            # Multiplier on learning rate

    def is_active(self, current_stage: DevelopmentStage) -> bool:
        return current_stage == self.stage


@dataclass
class DevelopmentConfig:
    """Configuration for developmental progression."""
    # Stage durations (simulated time units)
    genesis_duration: float = 100.0
    babbling_duration: float = 500.0
    imitation_duration: float = 1000.0
    autonomy_duration: float = 2000.0
    # MATURE has no end (indefinite)

    # Growth parameters
    initial_oscillators: int = 20       # Start small
    max_oscillators: int = 200          # Can grow to
    growth_rate: float = 0.1            # Base rate
    growth_interval: int = 10           # Significant experiences per growth

    # Critical periods (default set)
    critical_periods: List[CriticalPeriod] = field(default_factory=lambda: [
        CriticalPeriod("early_binding", DevelopmentStage.GENESIS, "pattern", 3.0),
        CriticalPeriod("exploration", DevelopmentStage.BABBLING, "novelty", 2.5),
        CriticalPeriod("social_learning", DevelopmentStage.IMITATION, "social", 2.0),
        CriticalPeriod("goal_formation", DevelopmentStage.AUTONOMY, "planning", 1.5),
    ])


# Stage ordering for transitions
_STAGE_ORDER = [
    DevelopmentStage.GENESIS,
    DevelopmentStage.BABBLING,
    DevelopmentStage.IMITATION,
    DevelopmentStage.AUTONOMY,
    DevelopmentStage.MATURE,
]


class DevelopmentTracker:
    """
    Tracks developmental progression of an entity.

    Entities start minimal and grow through experience. Identity is earned
    through development, not configured via founding memories.

    Five developmental stages:
    - GENESIS: Just born, establishing baseline rhythms
    - BABBLING: Random exploration, pattern discovery
    - IMITATION: Strong coupling to external rhythms
    - AUTONOMY: Self-generated goals emerge
    - MATURE: Stable but still plastic
    """

    def __init__(self, config: Optional[DevelopmentConfig] = None) -> None:
        self.config = config or DevelopmentConfig()

        # Current state
        self.stage = DevelopmentStage.GENESIS
        self.age: float = 0.0
        self._stage_start_age: float = 0.0

        # Growth tracking
        self.current_oscillators: int = self.config.initial_oscillators
        self.experiences_processed: int = 0
        self.significant_experiences: int = 0

        # Milestones (stage transitions recorded here)
        self.milestones: List[Dict] = []

        # Genesis hash - the ONLY identity anchor at birth
        # This is immutable for the lifetime of the entity
        self.genesis_hash: str = hashlib.sha256(
            f"{time.time()}:{np.random.random()}:{id(self)}".encode()
        ).hexdigest()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def stage_progress(self) -> float:
        """Progress through current stage (0 to 1)."""
        age_in_stage = self.age - self._stage_start_age
        duration = self._get_stage_duration()

        if duration is None:  # MATURE stage
            return 1.0

        return min(1.0, age_in_stage / duration)

    # ── Public Methods ───────────────────────────────────────────────────────

    def process_experience(
        self,
        experience: Dict[str, Any],
        significance: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Process an experience, potentially triggering growth or stage transition.

        Args:
            experience: Dict containing at least 'type' key.
            significance: 0-1, how significant is this experience.

        Returns:
            Dict with:
            - growth_triggered: bool
            - stage_transition: Optional[DevelopmentStage]
            - learning_multiplier: float
        """
        self.experiences_processed += 1

        result: Dict[str, Any] = {
            "growth_triggered": False,
            "stage_transition": None,
            "learning_multiplier": 1.0,
        }

        # Track significant experiences
        if significance > 0.7:
            self.significant_experiences += 1

        # Check for growth
        if self.should_grow():
            growth_amount = int(self.config.growth_rate * 10)
            self.current_oscillators = min(
                self.current_oscillators + growth_amount,
                self.config.max_oscillators,
            )
            result["growth_triggered"] = True

            # Record milestone
            self.milestones.append({
                "type": "growth",
                "oscillators": self.current_oscillators,
                "age": self.age,
                "experiences": self.experiences_processed,
            })

        # Check for stage transition
        duration = self._get_stage_duration()
        if duration is not None and (self.age - self._stage_start_age) >= duration:
            old_stage = self.stage
            self.stage = self._next_stage()
            self._stage_start_age = self.age
            result["stage_transition"] = self.stage

            # Record milestone
            self.milestones.append({
                "type": "stage_transition",
                "from": old_stage.value,
                "to": self.stage.value,
                "age": self.age,
                "experiences": self.experiences_processed,
            })

        # Get learning multiplier for this experience type
        learning_type = experience.get("type", "general")
        result["learning_multiplier"] = self.get_learning_multiplier(learning_type)

        return result

    def get_learning_multiplier(self, learning_type: str) -> float:
        """
        Get learning rate multiplier based on critical periods.

        Args:
            learning_type: Type of learning (e.g., "pattern", "social", "novelty").

        Returns:
            Multiplier >= 1.0 (1.0 if no active critical period).
        """
        multiplier = 1.0

        for period in self.config.critical_periods:
            if period.is_active(self.stage) and period.learning_type == learning_type:
                multiplier *= period.sensitivity

        return multiplier

    def advance_age(self, dt: float) -> None:
        """Advance developmental age by dt."""
        self.age += dt

    def should_grow(self) -> bool:
        """Check if entity should grow (add oscillators)."""
        cfg = self.config

        # Can't grow past max
        if self.current_oscillators >= cfg.max_oscillators:
            return False

        # Grow every N significant experiences
        if self.significant_experiences == 0:
            return False

        return self.significant_experiences % cfg.growth_interval == 0

    def get_state(self) -> dict:
        """Serialize current state."""
        return {
            "genesis_hash": self.genesis_hash,
            "stage": self.stage.value,
            "age": self.age,
            "stage_progress": self.stage_progress,
            "current_oscillators": self.current_oscillators,
            "experiences_processed": self.experiences_processed,
            "significant_experiences": self.significant_experiences,
            "milestones": self.milestones.copy(),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_stage_duration(self, stage: Optional[DevelopmentStage] = None) -> Optional[float]:
        """Get duration of a stage. None for MATURE (indefinite)."""
        stage = stage or self.stage
        cfg = self.config

        durations = {
            DevelopmentStage.GENESIS: cfg.genesis_duration,
            DevelopmentStage.BABBLING: cfg.babbling_duration,
            DevelopmentStage.IMITATION: cfg.imitation_duration,
            DevelopmentStage.AUTONOMY: cfg.autonomy_duration,
            DevelopmentStage.MATURE: None,
        }
        return durations[stage]

    def _next_stage(self) -> DevelopmentStage:
        """Get the next developmental stage."""
        idx = _STAGE_ORDER.index(self.stage)
        if idx < len(_STAGE_ORDER) - 1:
            return _STAGE_ORDER[idx + 1]
        return DevelopmentStage.MATURE
