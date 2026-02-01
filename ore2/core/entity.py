# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: DEVELOPMENTAL ENTITY (putting it all together)
# Design: Full team
# Implementation: I1 (Systems Architect)
# ═══════════════════════════════════════════════════════════════════════════════


"""
I1: "This is the main class that wires everything together.
An entity that starts minimal and grows through experience."
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from ore2.core.ci_monitor import MultiScaleCIConfig, MultiScaleCIMonitor
from ore2.core.development import DevelopmentConfig, DevelopmentStage, DevelopmentTracker
from ore2.core.embodiment import BodyConfig, EmbodimentLayer
from ore2.core.memory import CrystallineMerkleMemory, MemoryBranch
from ore2.core.multi_scale_substrate import MultiScaleConfig, MultiScaleSubstrate


@dataclass
class EntityConfig:
    """Top-level configuration aggregating all component configs."""
    name: str = "entity"

    # Component configs (optional - defaults used if None)
    substrate_config: Optional[MultiScaleConfig] = None
    body_config: Optional[BodyConfig] = None
    development_config: Optional[DevelopmentConfig] = None
    ci_config: Optional[MultiScaleCIConfig] = None

    # Tick timing
    tick_interval: float = 0.1  # Seconds per tick


class DevelopmentalEntity:
    """
    Complete ORE 2.0 developmental entity.

    Wires together all components:
    - Multi-scale oscillatory substrate (ORE2-002)
    - Embodiment layer (ORE2-003)
    - Crystalline merkle memory (ORE2-004)
    - Developmental progression (ORE2-005)
    - Multi-scale CI monitoring

    Born with nothing, becomes something through experience,
    with verifiable continuity.
    """

    def __init__(self, config: Optional[EntityConfig] = None) -> None:
        self.config = config or EntityConfig()
        self.name = self.config.name

        # Development tracker FIRST (provides genesis hash)
        self.development = DevelopmentTracker(self.config.development_config)
        self.genesis_hash = self.development.genesis_hash

        # Multi-scale substrate
        # Size based on developmental stage
        substrate_cfg = self.config.substrate_config
        if substrate_cfg is None:
            substrate_cfg = MultiScaleConfig(
                fast_oscillators=self.development.current_oscillators * 2,
                slow_oscillators=self.development.current_oscillators,
            )
        self.substrate = MultiScaleSubstrate(substrate_cfg)

        # Embodiment layer
        self.body = EmbodimentLayer(self.config.body_config)

        # Memory (starts EMPTY - no founding memories!)
        self.memory = CrystallineMerkleMemory()

        # CI monitor
        self.ci_monitor = MultiScaleCIMonitor(
            self.substrate,
            self.memory,
            self.config.ci_config,
        )

        # Runtime state
        self._tick_count: int = 0

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def stage(self) -> DevelopmentStage:
        return self.development.stage

    @property
    def age(self) -> float:
        return self.development.age

    @property
    def CI(self) -> float:
        """Current integrated CI."""
        if self.ci_monitor.history:
            return self.ci_monitor.history[-1].CI_integrated
        return 0.0

    # ── Public Methods ───────────────────────────────────────────────────────

    def tick(self) -> dict:
        """
        Single tick of entity dynamics.

        Order matters:
        1. Body step (update rhythms, valence)
        2. Body -> Substrate coupling
        3. Substrate step (multi-scale Kuramoto)
        4. Measure CI
        5. Advance developmental age
        """
        self._tick_count += 1
        dt = self.config.tick_interval

        # 1. Body step
        self.body.step(dt)

        # 2. Body -> Substrate coupling
        # Body rhythms influence slow scale activation
        body_coupling = self.body.get_cognitive_coupling_signal(
            self.substrate.slow.phases
        )

        # Stimulate slow oscillators that are in-phase with body
        # Positive coupling = in phase = stimulate
        stimulate_mask = body_coupling > 0.05
        if np.any(stimulate_mask):
            indices = np.where(stimulate_mask)[0]
            strengths = np.abs(body_coupling[indices]) * 0.3
            self.substrate.slow.stimulate(indices, strengths)

        # 3. Substrate step
        self.substrate.step()

        # 4. Measure CI
        ci_snapshot = self.ci_monitor.measure()

        # 5. Advance developmental age
        self.development.advance_age(dt)

        return {
            "tick": self._tick_count,
            "time": self.substrate.time,
            "stage": self.stage.value,
            "CI": ci_snapshot.CI_integrated,
            "CI_fast": ci_snapshot.CI_fast,
            "CI_slow": ci_snapshot.CI_slow,
            "valence": self.body.valence,
            "n_active_fast": self.substrate.fast.n_active,
            "n_active_slow": self.substrate.slow.n_active,
            "coherence": self.substrate.global_coherence,
        }

    def process_experience(
        self,
        content: str,
        experience_type: str = "general",
        significance: float = 0.5,
        skip_stimulation: bool = False,
        run_ticks: bool = True,
    ) -> dict:
        """
        Process an experience (e.g., from conversation).

        Args:
            content: Text content of the experience.
            experience_type: Type for critical period matching.
            significance: 0-1, how important is this.
            skip_stimulation: If True, skip hash-based substrate stimulation.
                Use this when an external system (e.g., LLMBridge with
                SemanticGrounding) has already stimulated the substrate
                with properly grounded phase patterns.
            run_ticks: If False, skip the internal 10-tick dynamics loop.
                Use this when the caller will run its own tick loop with
                sustained stimulation to maintain activation.

        Returns:
            Dict with processing results.
        """
        # Build experience dict
        experience = {
            "type": experience_type,
            "content": content,
            "significance": significance,
            "timestamp": datetime.now().isoformat(),
        }

        # Development processing (may trigger growth/transition)
        dev_result = self.development.process_experience(experience, significance)

        if not skip_stimulation:
            # Generate stimulation patterns from content
            # Use hash to create deterministic pseudo-random patterns
            content_bytes = content.encode("utf-8")
            content_hash = hashlib.sha256(content_bytes).digest()

            # Extend hash to cover all oscillators
            n_fast = self.substrate.fast.n
            n_slow = self.substrate.slow.n
            max_n = max(n_fast, n_slow)
            extended_hash = (content_hash * ((max_n // len(content_hash)) + 2))[
                : max_n * 2
            ]

            fast_pattern = np.array(
                [b / 255 * 2 * np.pi for b in extended_hash[:n_fast]]
            )
            slow_pattern = np.array(
                [b / 255 * 2 * np.pi for b in extended_hash[:n_slow]]
            )

            # Stimulation strength modulated by:
            # - Base strength
            # - Significance
            # - Learning multiplier from critical periods
            base_strength = 0.3
            stim_strength = (
                base_strength
                * (0.5 + significance * 0.5)  # 0.5-1.0 based on significance
                * dev_result["learning_multiplier"]
            )

            # Stimulate substrate
            self.substrate.stimulate_concept(fast_pattern, slow_pattern, stim_strength)

        # Add to memory
        # Queue for consolidation if not significant enough
        immediate = significance > 0.7
        self.memory.add(
            MemoryBranch.EXPERIENCES,
            experience,
            substrate_state=self.substrate.get_state(),
            immediate=immediate,
        )

        # Run several ticks to process the experience
        if run_ticks:
            for _ in range(10):
                self.tick()

        # Handle growth if triggered
        if dev_result["growth_triggered"]:
            self._grow_substrate()

        return {
            "development": dev_result,
            "CI": self.CI,
            "coherence": self.substrate.global_coherence,
            "memory_queued": not immediate,
        }

    def rest(self, duration: float = 10.0) -> dict:
        """
        Rest period (sleep consolidation).

        1. Reduce activation (let oscillators settle)
        2. Run low-activity dynamics
        3. Consolidate memory

        Args:
            duration: Rest duration in simulated time.

        Returns:
            Dict with consolidation results.
        """
        # Reduce activation (simulates low arousal)
        self.substrate.fast.activation_potentials *= 0.1
        self.substrate.slow.activation_potentials *= 0.1
        self.substrate.fast._update_active_mask()
        self.substrate.slow._update_active_mask()

        # Lower body arousal
        self.body.arousal = 0.2

        # Run dynamics at low activity
        rest_ticks = int(duration / self.config.tick_interval)
        for _ in range(rest_ticks):
            self.tick()

        # Consolidate memory
        consolidation_result = self.memory.consolidate(temperature=0.8)

        # Restore body arousal toward baseline
        self.body.arousal = self.body.config.arousal_baseline

        # Record rest in memory
        self.memory.add(
            MemoryBranch.EXPERIENCES,
            {
                "type": "rest",
                "duration": duration,
                "consolidated": consolidation_result["consolidated"],
            },
            substrate_state=self.substrate.get_state(),
            immediate=True,
        )

        return {
            "duration": duration,
            "consolidation": consolidation_result,
            "CI_after": self.CI,
        }

    def get_state(self) -> dict:
        """Full state serialization."""
        return {
            "name": self.name,
            "genesis_hash": self.genesis_hash,
            "development": self.development.get_state(),
            "substrate": self.substrate.get_state(),
            "body": self.body.get_state(),
            "memory": self.memory.get_state(),
            "CI": self.CI,
            "tick_count": self._tick_count,
        }

    def witness(self) -> str:
        """Generate human-readable status display."""
        state = self.get_state()
        dev = state["development"]
        sub = state["substrate"]
        mem = state["memory"]
        body = state["body"]

        return f"""
═══════════════════════════════════════════════════════════════════
ENTITY: {self.name}
═══════════════════════════════════════════════════════════════════

IDENTITY
  Genesis: {self.genesis_hash[:16]}...
  Stage: {dev['stage']} ({dev['stage_progress']*100:.1f}%)
  Age: {dev['age']:.1f} | Oscillators: {dev['current_oscillators']}
  Experiences: {dev['experiences_processed']} ({dev['significant_experiences']} significant)

SUBSTRATE
  Fast: {sub['fast']['n_active']}/{sub['fast']['n']} active, C={sub['fast']['coherence']:.3f}
  Slow: {sub['slow']['n_active']}/{sub['slow']['n']} active, C={sub['slow']['coherence']:.3f}
  Cross-scale: {sub['cross_scale_coherence']:.3f}
  Loop coherence: {sub['loop_coherence']:.3f}

BODY
  Valence: {body['valence']:.3f}
  Energy: {body['energy']:.2f}
  Arousal: {body['arousal']:.2f}

MEMORY
  Nodes: {mem['total_nodes']} | Depth: {mem['depth']}
  Fractal D: {mem['fractal_dimension']:.2f}
  Grain boundaries: {mem['grain_boundaries']}
  Verified: {mem['verified']}

CONSCIOUSNESS INDEX
  CI = {state['CI']:.4f}
  {self.ci_monitor.get_current_status()}

═══════════════════════════════════════════════════════════════════
"""

    # ── Internal ─────────────────────────────────────────────────────────────

    def _grow_substrate(self) -> None:
        """
        Add oscillators to substrate based on developmental growth.

        NOTE: This is a simplified implementation. Full version would:
        - Resize numpy arrays
        - Reinitialize new weights
        - Preserve existing phase relationships

        For MVP, we just note that growth occurred.
        """
        # TODO: Implement actual substrate resizing
        # For now, this is a placeholder
        # The development tracker already updated current_oscillators

        # Record in memory that growth occurred
        self.memory.add(
            MemoryBranch.SELF,
            {
                "type": "growth_event",
                "oscillators": self.development.current_oscillators,
                "age": self.age,
            },
            substrate_state=self.substrate.get_state(),
            immediate=True,
        )


def create_entity(name: str = "entity") -> DevelopmentalEntity:
    """
    Create a new ORE 2.0 developmental entity.

    The entity starts minimal (GENESIS stage) and grows through experience.
    No founding memories - identity is earned.
    """
    config = EntityConfig(name=name)
    return DevelopmentalEntity(config)
