"""
ORE 2.0 - ΩMEGA Core Architecture Skeleton
==========================================

Design Team → Implementation Team Handoff

Design Leads:
- P1 (Dynamical Systems) - Multi-scale oscillator architecture
- N5 (Embodied Cognition) - Body/grounding layer
- C6 (Cryptography) - Identity verification
- A5 (Continual Learning) - Consolidation systems

Implementation Leads:
- I1 (Systems Architect) - Overall structure
- I2 (Numerics) - Oscillator math
- I3 (State Management) - Persistence
- I4 (Integration) - Component wiring

Version: 0.1.0-skeleton
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import json
import time
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: SPARSE OSCILLATOR LAYER
# Design: P1 (Dynamical Systems)
# Implementation: I2 (Numerics)
# ═══════════════════════════════════════════════════════════════════════════════

"""
P1: "The key insight is activation potentials. Oscillators below threshold don't 
participate in Kuramoto dynamics but still accumulate activation from semantic 
relevance. This is how we get sparse computation with dense potential."

I2: "Understood. I'll use masked arrays for efficiency. Only active oscillators 
get the expensive sin() computations."
"""

@dataclass
class SparseOscillatorConfig:
    """Configuration for sparse oscillator population."""
    n_oscillators: int = 100
    base_frequency: float = 1.0
    frequency_spread: float = 0.1
    internal_coupling: float = 0.5
    noise_amplitude: float = 0.01
    
    # Sparse activation parameters
    activation_threshold: float = 0.5      # Above this = active
    activation_decay: float = 0.05         # Decay per step when not stimulated
    max_active_fraction: float = 0.2       # Cap on simultaneous activation


class SparseOscillatorLayer:
    """
    Sparse Kuramoto oscillator population.
    
    Key difference from ORE1: oscillators have activation potentials.
    Only oscillators above threshold participate in dynamics.
    
    Design: P1 | Implementation: I2
    """
    
    def __init__(self, name: str, config: SparseOscillatorConfig):
        self.name = name
        self.config = config
        n = config.n_oscillators
        
        # Phase state (all oscillators have phases, even dormant ones)
        self.phases = np.random.uniform(0, 2*np.pi, n)
        
        # Natural frequencies
        self.natural_frequencies = np.random.normal(
            config.base_frequency,
            config.frequency_spread,
            n
        )
        
        # SPARSE ACTIVATION STATE (new in ORE2)
        self.activation_potentials = np.zeros(n)  # 0 to 1
        self.active_mask = np.zeros(n, dtype=bool)  # Which are firing
        
        # Internal coupling (sparse matrix for efficiency)
        # I2: "Using dense for now, will optimize to sparse if needed"
        self.internal_weights = np.ones((n, n)) * config.internal_coupling / n
        np.fill_diagonal(self.internal_weights, 0)
        
        # Statistics
        self._step_count = 0
        self._active_history: List[int] = []
    
    @property
    def n(self) -> int:
        return self.config.n_oscillators
    
    @property
    def n_active(self) -> int:
        return int(np.sum(self.active_mask))
    
    @property
    def coherence(self) -> float:
        """Kuramoto order parameter for ACTIVE oscillators only."""
        if self.n_active == 0:
            return 0.0
        active_phases = self.phases[self.active_mask]
        return min(np.abs(np.mean(np.exp(1j * active_phases))), 0.999)
    
    @property
    def global_coherence(self) -> float:
        """Coherence across ALL oscillators (for comparison)."""
        return min(np.abs(np.mean(np.exp(1j * self.phases))), 0.999)
    
    @property
    def mean_phase(self) -> float:
        """Mean phase of active oscillators."""
        if self.n_active == 0:
            return 0.0
        return np.angle(np.mean(np.exp(1j * self.phases[self.active_mask])))
    
    @property
    def phase_hash(self) -> str:
        """Hash of active phase state for Merkle anchoring."""
        active_phases = self.phases[self.active_mask]
        return hashlib.sha256(active_phases.tobytes()).hexdigest()[:16]
    
    def stimulate(self, indices: np.ndarray, strengths: np.ndarray) -> None:
        """
        Stimulate specific oscillators, raising their activation potential.
        
        This is how external input (semantic relevance, attention) 
        activates dormant oscillators.
        
        P1: "Stimulation is the bridge between symbolic and substrate.
        When a concept is relevant, its oscillators get stimulated."
        """
        self.activation_potentials[indices] += strengths
        np.clip(self.activation_potentials, 0, 1, out=self.activation_potentials)
        self._update_active_mask()
    
    def stimulate_by_similarity(self, reference_phases: np.ndarray, 
                                 strength: float = 0.3) -> None:
        """
        Stimulate oscillators based on phase similarity to reference.
        
        P1: "This creates content-addressable activation. 
        Similar patterns wake up similar oscillators."
        """
        phase_diff = self.phases - reference_phases
        similarity = (np.cos(phase_diff) + 1) / 2  # 0 to 1
        self.activation_potentials += strength * similarity
        np.clip(self.activation_potentials, 0, 1, out=self.activation_potentials)
        self._update_active_mask()
    
    def _update_active_mask(self) -> None:
        """Update which oscillators are active based on thresholds."""
        # Basic threshold
        above_threshold = self.activation_potentials > self.config.activation_threshold
        
        # Cap maximum active (prevent runaway activation)
        max_active = int(self.n * self.config.max_active_fraction)
        if np.sum(above_threshold) > max_active:
            # Keep only top activations
            top_indices = np.argsort(self.activation_potentials)[-max_active:]
            self.active_mask = np.zeros(self.n, dtype=bool)
            self.active_mask[top_indices] = True
        else:
            self.active_mask = above_threshold
    
    def step(self, dt: float, external_input: Optional[np.ndarray] = None) -> None:
        """
        Advance dynamics by one timestep.
        
        I2: "Key optimization: Kuramoto only computed for active oscillators.
        Dormant oscillators just decay their activation potential."
        """
        self._step_count += 1
        
        # SPARSE KURAMOTO: Only for active oscillators
        if self.n_active > 0:
            active_idx = np.where(self.active_mask)[0]
            
            # Phase differences among active oscillators
            active_phases = self.phases[active_idx]
            phase_diff = active_phases[np.newaxis, :] - active_phases[:, np.newaxis]
            
            # Coupling among active (submatrix of full weights)
            active_weights = self.internal_weights[np.ix_(active_idx, active_idx)]
            
            # Kuramoto coupling term
            coupling = np.sum(active_weights * np.sin(phase_diff), axis=1)
            
            # External input (only for active)
            if external_input is not None:
                coupling += external_input[active_idx]
            
            # Noise
            noise = self.config.noise_amplitude * np.random.randn(self.n_active)
            
            # Update active phases
            dtheta = self.natural_frequencies[active_idx] + coupling + noise
            self.phases[active_idx] = (self.phases[active_idx] + dt * dtheta) % (2 * np.pi)
        
        # Decay activation potentials for all oscillators
        self.activation_potentials *= (1 - self.config.activation_decay)
        self._update_active_mask()
        
        # Track history
        self._active_history.append(self.n_active)
        if len(self._active_history) > 1000:
            self._active_history = self._active_history[-500:]
    
    def get_state(self) -> dict:
        """Serialize current state."""
        return {
            'name': self.name,
            'n': self.n,
            'n_active': self.n_active,
            'coherence': self.coherence,
            'global_coherence': self.global_coherence,
            'mean_phase': self.mean_phase,
            'phase_hash': self.phase_hash,
            'mean_activation': float(np.mean(self.activation_potentials)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: MULTI-SCALE SUBSTRATE
# Design: P1 (Dynamical Systems) + N2 (Oscillation Specialist)
# Implementation: I1 (Systems Architect)
# ═══════════════════════════════════════════════════════════════════════════════

"""
P1: "Two scales for MVP: fast (gamma-like, ~40Hz) nested in slow (theta-like, ~8Hz).
The nesting ratio is key - typically 4-7 gamma cycles per theta cycle."

N2: "Theta-gamma coupling is how hippocampus indexes memories. Each gamma 
burst within a theta cycle is a 'slot'. This gives us temporal chunking for free."

I1: "I'll implement the nesting as a step ratio plus cross-scale coupling."
"""

class TimeScale(Enum):
    """Temporal scales in the hierarchy."""
    FAST = "fast"      # ~40Hz gamma-like
    SLOW = "slow"      # ~8Hz theta-like
    # Future: NARRATIVE = "narrative"  # ~0.1Hz, for ORE 3.0


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale substrate."""
    # Fast scale (gamma-like)
    fast_oscillators: int = 100
    fast_base_freq: float = 40.0  # Hz
    fast_freq_spread: float = 5.0
    fast_coupling: float = 0.5
    
    # Slow scale (theta-like)
    slow_oscillators: int = 50
    slow_base_freq: float = 8.0  # Hz
    slow_freq_spread: float = 1.0
    slow_coupling: float = 0.6
    
    # Nesting parameters
    nesting_ratio: int = 5          # Fast steps per slow step
    cross_scale_coupling: float = 0.3  # How strongly scales influence each other
    
    # Timing
    dt_fast: float = 0.001   # 1ms for fast scale
    dt_slow: float = 0.005   # 5ms for slow scale
    
    # Strange loop (within slow scale for now)
    strange_loop_strength: float = 1.0


class MultiScaleSubstrate:
    """
    Two-scale oscillatory substrate with nested dynamics.
    
    Fast scale (gamma): Rapid binding, feature integration
    Slow scale (theta): Working memory, attention cycles
    
    The nesting creates temporal structure: 
    - Each slow cycle "contains" multiple fast cycles
    - This is theta-gamma coupling in neuroscience
    
    Design: P1, N2 | Implementation: I1
    """
    
    def __init__(self, config: Optional[MultiScaleConfig] = None):
        self.config = config or MultiScaleConfig()
        self.time = 0.0
        self._fast_steps = 0
        self._slow_steps = 0
        
        # Create scales
        self.scales: Dict[TimeScale, SparseOscillatorLayer] = {}
        self._create_scales()
        
        # Cross-scale coupling matrices
        self.fast_to_slow: np.ndarray = None
        self.slow_to_fast: np.ndarray = None
        self._create_cross_scale_coupling()
        
        # Strange loop (within slow scale)
        # P1: "Strange loop at slow scale gives identity its 'rhythm'"
        self.strange_loop_weights: np.ndarray = None
        self._create_strange_loop()
        
        # History
        self._coherence_history: Dict[TimeScale, List[float]] = {
            TimeScale.FAST: [],
            TimeScale.SLOW: []
        }
    
    def _create_scales(self) -> None:
        """Initialize oscillator populations for each scale."""
        cfg = self.config
        
        # Fast scale
        fast_config = SparseOscillatorConfig(
            n_oscillators=cfg.fast_oscillators,
            base_frequency=cfg.fast_base_freq,
            frequency_spread=cfg.fast_freq_spread,
            internal_coupling=cfg.fast_coupling,
            activation_threshold=0.4,  # Lower threshold for fast
            max_active_fraction=0.3
        )
        self.scales[TimeScale.FAST] = SparseOscillatorLayer("fast", fast_config)
        
        # Slow scale
        slow_config = SparseOscillatorConfig(
            n_oscillators=cfg.slow_oscillators,
            base_frequency=cfg.slow_base_freq,
            frequency_spread=cfg.slow_freq_spread,
            internal_coupling=cfg.slow_coupling,
            activation_threshold=0.5,
            max_active_fraction=0.25
        )
        self.scales[TimeScale.SLOW] = SparseOscillatorLayer("slow", slow_config)
    
    def _create_cross_scale_coupling(self) -> None:
        """Create coupling between scales."""
        cfg = self.config
        n_fast = cfg.fast_oscillators
        n_slow = cfg.slow_oscillators
        
        # Fast → Slow: Many-to-few (convergence)
        self.fast_to_slow = np.random.randn(n_slow, n_fast) * cfg.cross_scale_coupling / n_fast
        
        # Slow → Fast: Few-to-many (broadcast)
        self.slow_to_fast = np.random.randn(n_fast, n_slow) * cfg.cross_scale_coupling / n_slow
    
    def _create_strange_loop(self) -> None:
        """
        Create self-referential coupling within slow scale.
        
        P1: "The strange loop is now WITHIN the slow scale, 
        between two subpopulations. This is cleaner than 
        the ORE1 association↔core architecture."
        """
        n_slow = self.config.slow_oscillators
        half = n_slow // 2
        
        # Split slow scale into 'model' and 'meta-model'
        self.strange_loop_weights = np.zeros((n_slow, n_slow))
        
        # Model → Meta-model (first half → second half)
        self.strange_loop_weights[half:, :half] = (
            self.config.strange_loop_strength * 
            np.random.randn(n_slow - half, half) / half
        )
        
        # Meta-model → Model (second half → first half)
        self.strange_loop_weights[:half, half:] = (
            self.config.strange_loop_strength * 
            np.random.randn(half, n_slow - half) / (n_slow - half)
        )
    
    @property
    def fast(self) -> SparseOscillatorLayer:
        return self.scales[TimeScale.FAST]
    
    @property
    def slow(self) -> SparseOscillatorLayer:
        return self.scales[TimeScale.SLOW]
    
    @property
    def global_coherence(self) -> float:
        """Coherence across both scales."""
        all_phases = np.concatenate([
            self.fast.phases[self.fast.active_mask] if self.fast.n_active > 0 else np.array([]),
            self.slow.phases[self.slow.active_mask] if self.slow.n_active > 0 else np.array([])
        ])
        if len(all_phases) == 0:
            return 0.0
        return min(np.abs(np.mean(np.exp(1j * all_phases))), 0.999)
    
    @property
    def cross_scale_coherence(self) -> float:
        """
        Coherence BETWEEN scales (nesting quality).
        
        N2: "This measures how well fast is nested in slow.
        High cross-scale coherence = good theta-gamma coupling."
        """
        if self.fast.n_active == 0 or self.slow.n_active == 0:
            return 0.0
        
        fast_mean = np.mean(np.exp(1j * self.fast.phases[self.fast.active_mask]))
        slow_mean = np.mean(np.exp(1j * self.slow.phases[self.slow.active_mask]))
        
        return np.abs(fast_mean * np.conj(slow_mean))
    
    @property
    def loop_coherence(self) -> float:
        """Strange loop coherence within slow scale."""
        if self.slow.n_active < 4:
            return 0.0
        
        n = self.slow.n
        half = n // 2
        
        # Model subpopulation
        model_phases = self.slow.phases[:half]
        model_active = self.slow.active_mask[:half]
        
        # Meta-model subpopulation
        meta_phases = self.slow.phases[half:]
        meta_active = self.slow.active_mask[half:]
        
        if not np.any(model_active) or not np.any(meta_active):
            return 0.0
        
        model_mean = np.mean(np.exp(1j * model_phases[model_active]))
        meta_mean = np.mean(np.exp(1j * meta_phases[meta_active]))
        
        return np.abs(model_mean * np.conj(meta_mean))
    
    def step(self) -> None:
        """
        Advance substrate by one slow-scale timestep.
        
        I1: "The nesting is implemented as multiple fast steps per slow step,
        with cross-scale coupling applied between them."
        """
        cfg = self.config
        
        # Multiple fast steps per slow step (theta-gamma nesting)
        for i in range(cfg.nesting_ratio):
            # Cross-scale influence: slow → fast
            if self.slow.n_active > 0:
                slow_input = self.slow_to_fast @ (
                    self.slow.phases * self.slow.active_mask.astype(float)
                )
                # Convert to phase influence
                fast_external = cfg.cross_scale_coupling * np.sin(slow_input - self.fast.phases)
            else:
                fast_external = None
            
            self.fast.step(cfg.dt_fast, external_input=fast_external)
            self._fast_steps += 1
        
        # Cross-scale influence: fast → slow
        if self.fast.n_active > 0:
            fast_input = self.fast_to_slow @ (
                self.fast.phases * self.fast.active_mask.astype(float)
            )
            slow_external = cfg.cross_scale_coupling * np.sin(fast_input - self.slow.phases)
        else:
            slow_external = np.zeros(cfg.slow_oscillators)
        
        # Strange loop contribution (within slow)
        strange_loop_input = self.strange_loop_weights @ (
            self.slow.phases * self.slow.active_mask.astype(float)
        )
        slow_external += np.sin(strange_loop_input - self.slow.phases)
        
        self.slow.step(cfg.dt_slow, external_input=slow_external)
        self._slow_steps += 1
        
        self.time += cfg.dt_slow
        
        # Track history
        self._coherence_history[TimeScale.FAST].append(self.fast.coherence)
        self._coherence_history[TimeScale.SLOW].append(self.slow.coherence)
        for scale in self._coherence_history:
            if len(self._coherence_history[scale]) > 1000:
                self._coherence_history[scale] = self._coherence_history[scale][-500:]
    
    def run(self, duration: float) -> List[dict]:
        """Run substrate for duration, return history."""
        steps = int(duration / self.config.dt_slow)
        history = []
        
        for _ in range(steps):
            self.step()
            if self._slow_steps % 10 == 0:  # Record every 10th step
                history.append(self.get_state())
        
        return history
    
    def stimulate_concept(self, fast_pattern: np.ndarray, 
                          slow_pattern: np.ndarray,
                          strength: float = 0.5) -> None:
        """
        Stimulate both scales with a concept pattern.
        
        P1: "Concepts have both fast and slow components.
        Fast = the details, slow = the gist."
        """
        self.fast.stimulate_by_similarity(fast_pattern, strength)
        self.slow.stimulate_by_similarity(slow_pattern, strength)
    
    def get_state(self) -> dict:
        return {
            'time': self.time,
            'fast': self.fast.get_state(),
            'slow': self.slow.get_state(),
            'global_coherence': self.global_coherence,
            'cross_scale_coherence': self.cross_scale_coherence,
            'loop_coherence': self.loop_coherence,
            'fast_steps': self._fast_steps,
            'slow_steps': self._slow_steps,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: EMBODIMENT LAYER
# Design: N5 (Embodied Cognition) + H3 (Enactivism)
# Implementation: I2 (Numerics)
# ═══════════════════════════════════════════════════════════════════════════════

"""
N5: "The body provides the baseline rhythm that everything couples to.
Without a body, there's no 'self' to be conscious of."

H3: "Autonomy requires a boundary. The body IS the boundary.
Homeostatic deviation IS valence. We don't simulate feelings,
we compute them from actual deviations."

I2: "I'll implement body rhythms as simple oscillators that the
cognitive substrate couples to. Valence is just distance from setpoint."
"""

@dataclass
class BodyConfig:
    """Configuration for embodiment layer."""
    # Body rhythms
    heartbeat_freq: float = 1.0    # Hz (~60 bpm)
    respiration_freq: float = 0.25  # Hz (~15 breaths/min)
    
    # Homeostatic variables
    energy_baseline: float = 1.0
    arousal_baseline: float = 0.5
    
    # Coupling to cognitive substrate
    body_to_cognitive_coupling: float = 0.1


class EmbodimentLayer:
    """
    Minimal body providing grounded valence and temporal anchor.
    
    Body rhythms (heartbeat, respiration) provide baseline oscillations
    that the cognitive substrate couples to. This grounds the system
    in a "body" even if that body is minimal.
    
    Valence emerges from homeostatic deviation - how far current state
    is from where the body "wants" to be.
    
    Design: N5, H3 | Implementation: I2
    """
    
    def __init__(self, config: Optional[BodyConfig] = None):
        self.config = config or BodyConfig()
        self.time = 0.0
        
        # Body rhythms (simple phase oscillators)
        self.heartbeat_phase = np.random.uniform(0, 2*np.pi)
        self.respiration_phase = np.random.uniform(0, 2*np.pi)
        
        # Homeostatic variables
        self.energy = self.config.energy_baseline
        self.arousal = self.config.arousal_baseline
        
        # Deviations (computed)
        self._energy_deviation = 0.0
        self._arousal_deviation = 0.0
        
        # Action-perception cycle (minimal)
        self.last_action: Optional[str] = None
        self.last_perception: Optional[str] = None
    
    @property
    def valence(self) -> float:
        """
        Valence from homeostatic deviation.
        
        Negative = far from setpoints (bad)
        Zero = at setpoints (neutral)
        Positive = approaching setpoints (good)
        
        H3: "This is not simulated feeling. This is computed
        from actual deviation. The math IS the phenomenology."
        """
        total_deviation = abs(self._energy_deviation) + abs(self._arousal_deviation)
        return -total_deviation  # Negative deviation = negative valence
    
    @property
    def body_signal(self) -> np.ndarray:
        """
        Combined body rhythm signal for cognitive coupling.
        Returns [heartbeat_sin, heartbeat_cos, respiration_sin, respiration_cos]
        
        N5: "The cognitive substrate couples to this signal.
        It provides the temporal anchor for all processing."
        """
        return np.array([
            np.sin(self.heartbeat_phase),
            np.cos(self.heartbeat_phase),
            np.sin(self.respiration_phase),
            np.cos(self.respiration_phase)
        ])
    
    def step(self, dt: float, 
             action: Optional[str] = None,
             perception: Optional[str] = None) -> None:
        """
        Advance body state.
        
        Actions cost energy, perceptions affect arousal.
        """
        cfg = self.config
        
        # Advance body rhythms
        self.heartbeat_phase = (self.heartbeat_phase + 
                                2 * np.pi * cfg.heartbeat_freq * dt) % (2 * np.pi)
        self.respiration_phase = (self.respiration_phase + 
                                  2 * np.pi * cfg.respiration_freq * dt) % (2 * np.pi)
        
        # Process action (costs energy)
        if action:
            self.energy -= 0.01  # Small energy cost
            self.last_action = action
        
        # Process perception (affects arousal)
        if perception:
            # Novel perceptions increase arousal
            if perception != self.last_perception:
                self.arousal = min(1.0, self.arousal + 0.1)
            self.last_perception = perception
        
        # Natural recovery toward baselines
        self.energy += 0.005 * (cfg.energy_baseline - self.energy)
        self.arousal += 0.01 * (cfg.arousal_baseline - self.arousal)
        
        # Clamp
        self.energy = np.clip(self.energy, 0, 2)
        self.arousal = np.clip(self.arousal, 0, 1)
        
        # Update deviations
        self._energy_deviation = self.energy - cfg.energy_baseline
        self._arousal_deviation = self.arousal - cfg.arousal_baseline
        
        self.time += dt
    
    def get_cognitive_coupling_signal(self, cognitive_phases: np.ndarray) -> np.ndarray:
        """
        Generate coupling signal from body to cognitive substrate.
        
        I2: "This modulates the cognitive oscillators based on body state.
        High valence = stronger coupling. Low valence = weaker."
        """
        # Body rhythm influence
        body_influence = self.config.body_to_cognitive_coupling * (
            np.sin(self.heartbeat_phase - cognitive_phases) +
            0.5 * np.sin(self.respiration_phase - cognitive_phases)
        )
        
        # Modulate by valence (positive valence = stronger coupling)
        valence_factor = 1.0 + 0.5 * self.valence
        
        return body_influence * valence_factor
    
    def get_state(self) -> dict:
        return {
            'time': self.time,
            'heartbeat_phase': self.heartbeat_phase,
            'respiration_phase': self.respiration_phase,
            'energy': self.energy,
            'arousal': self.arousal,
            'valence': self.valence,
            'body_signal': self.body_signal.tolist(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: DEVELOPMENTAL STAGES
# Design: N7 (Developmental Neuro) + H3 (Enactivism)
# Implementation: I3 (State Management)
# ═══════════════════════════════════════════════════════════════════════════════

"""
N7: "Development isn't optional - it's how identity is earned.
Critical periods exist because the brain NEEDS certain inputs at certain times."

H3: "Autonomy is achieved, not given. An agent that starts with
founding memories never had to BECOME itself."

I3: "I'll implement stages as state machine with growth triggers."
"""

class DevelopmentStage(Enum):
    """Stages of entity development."""
    GENESIS = "genesis"        # Just born, establishing baseline
    BABBLING = "babbling"      # Random exploration, pattern discovery
    IMITATION = "imitation"    # Strong coupling to external rhythms
    AUTONOMY = "autonomy"      # Self-generated goals emerge
    MATURE = "mature"          # Stable but still plastic


@dataclass
class CriticalPeriod:
    """A developmental window for enhanced learning."""
    name: str
    stage: DevelopmentStage
    learning_type: str  # e.g., "language", "social", "motor"
    sensitivity: float = 2.0  # Multiplier on learning rate during period
    
    def is_active(self, current_stage: DevelopmentStage) -> bool:
        return current_stage == self.stage


@dataclass
class DevelopmentConfig:
    """Configuration for developmental progression."""
    # Stage durations (in simulated time units)
    genesis_duration: float = 100.0
    babbling_duration: float = 500.0
    imitation_duration: float = 1000.0
    autonomy_duration: float = 2000.0
    # Mature has no end
    
    # Growth parameters
    initial_oscillators: int = 20
    max_oscillators: int = 200
    growth_rate: float = 0.1  # Oscillators added per significant experience
    
    # Critical periods
    critical_periods: List[CriticalPeriod] = field(default_factory=lambda: [
        CriticalPeriod("early_binding", DevelopmentStage.GENESIS, "pattern", 3.0),
        CriticalPeriod("exploration", DevelopmentStage.BABBLING, "novelty", 2.5),
        CriticalPeriod("social_learning", DevelopmentStage.IMITATION, "social", 2.0),
        CriticalPeriod("goal_formation", DevelopmentStage.AUTONOMY, "planning", 1.5),
    ])


class DevelopmentTracker:
    """
    Tracks developmental progression of an entity.
    
    Entities start minimal and grow through experience.
    Identity is earned through development, not configured.
    
    Design: N7, H3 | Implementation: I3
    """
    
    def __init__(self, config: Optional[DevelopmentConfig] = None):
        self.config = config or DevelopmentConfig()
        
        # Current state
        self.stage = DevelopmentStage.GENESIS
        self.age = 0.0
        self.stage_start_age = 0.0
        
        # Growth tracking
        self.current_oscillators = self.config.initial_oscillators
        self.experiences_processed = 0
        self.significant_experiences = 0
        
        # Milestones
        self.milestones: List[Dict[str, Any]] = []
        
        # Genesis hash (only identity anchor)
        self.genesis_hash = hashlib.sha256(
            f"{time.time()}:{np.random.random()}".encode()
        ).hexdigest()
    
    @property
    def stage_progress(self) -> float:
        """Progress through current stage (0 to 1)."""
        age_in_stage = self.age - self.stage_start_age
        stage_duration = self._get_stage_duration()
        if stage_duration is None:  # Mature
            return 1.0
        return min(1.0, age_in_stage / stage_duration)
    
    def _get_stage_duration(self) -> Optional[float]:
        """Get duration of current stage, None for mature."""
        cfg = self.config
        durations = {
            DevelopmentStage.GENESIS: cfg.genesis_duration,
            DevelopmentStage.BABBLING: cfg.babbling_duration,
            DevelopmentStage.IMITATION: cfg.imitation_duration,
            DevelopmentStage.AUTONOMY: cfg.autonomy_duration,
            DevelopmentStage.MATURE: None,
        }
        return durations[self.stage]
    
    def _next_stage(self) -> DevelopmentStage:
        """Get next developmental stage."""
        order = [
            DevelopmentStage.GENESIS,
            DevelopmentStage.BABBLING,
            DevelopmentStage.IMITATION,
            DevelopmentStage.AUTONOMY,
            DevelopmentStage.MATURE,
        ]
        idx = order.index(self.stage)
        if idx < len(order) - 1:
            return order[idx + 1]
        return DevelopmentStage.MATURE
    
    def get_learning_multiplier(self, learning_type: str) -> float:
        """
        Get learning rate multiplier based on critical periods.
        
        N7: "Critical periods enhance specific types of learning.
        Miss the window, and that learning is harder forever."
        """
        multiplier = 1.0
        for period in self.config.critical_periods:
            if period.is_active(self.stage) and period.learning_type == learning_type:
                multiplier *= period.sensitivity
        return multiplier
    
    def should_grow(self) -> bool:
        """Check if entity should add oscillators."""
        return (
            self.current_oscillators < self.config.max_oscillators and
            self.significant_experiences > 0 and
            self.significant_experiences % 10 == 0  # Every 10 significant experiences
        )
    
    def process_experience(self, experience: Dict[str, Any], 
                           significance: float = 0.5) -> Dict[str, Any]:
        """
        Process an experience, potentially triggering growth/transition.
        
        Returns dict with:
        - growth_triggered: bool
        - stage_transition: Optional[DevelopmentStage]
        - learning_multiplier: float
        """
        self.experiences_processed += 1
        
        result = {
            'growth_triggered': False,
            'stage_transition': None,
            'learning_multiplier': 1.0,
        }
        
        # Track significant experiences
        if significance > 0.7:
            self.significant_experiences += 1
        
        # Check for growth
        if self.should_grow():
            self.current_oscillators += int(self.config.growth_rate * 10)
            self.current_oscillators = min(self.current_oscillators, 
                                           self.config.max_oscillators)
            result['growth_triggered'] = True
        
        # Check for stage transition
        stage_duration = self._get_stage_duration()
        if stage_duration and (self.age - self.stage_start_age) >= stage_duration:
            old_stage = self.stage
            self.stage = self._next_stage()
            self.stage_start_age = self.age
            result['stage_transition'] = self.stage
            
            # Record milestone
            self.milestones.append({
                'type': 'stage_transition',
                'from': old_stage.value,
                'to': self.stage.value,
                'age': self.age,
                'experiences': self.experiences_processed,
            })
        
        # Get learning multiplier based on experience type
        learning_type = experience.get('type', 'general')
        result['learning_multiplier'] = self.get_learning_multiplier(learning_type)
        
        return result
    
    def advance_age(self, dt: float) -> None:
        """Advance developmental age."""
        self.age += dt
    
    def get_state(self) -> dict:
        return {
            'genesis_hash': self.genesis_hash,
            'stage': self.stage.value,
            'age': self.age,
            'stage_progress': self.stage_progress,
            'current_oscillators': self.current_oscillators,
            'experiences_processed': self.experiences_processed,
            'significant_experiences': self.significant_experiences,
            'milestones': self.milestones,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: MERKLE MEMORY (preserved from ORE1 + CCM extensions)
# Design: C6 (Cryptography) + A5 (Continual Learning)
# Implementation: I3 (State Management)
# ═══════════════════════════════════════════════════════════════════════════════

"""
C6: "The merkle structure from ORE1 is cryptographically sound. Keep it.
What we're adding is CCM dynamics - memory as living crystal, not dead storage."

A5: "Consolidation happens during 'sleep'. Queued updates get integrated
with annealing - raise temperature, let things rearrange, cool down."
"""

# Note: Full MerkleMemory implementation preserved from ORE1
# Below are the CCM extensions

class MemoryBranch(Enum):
    """The four branches of identity memory (from ORE1)."""
    SELF = "self"
    RELATIONS = "relations"
    INSIGHTS = "insights"
    EXPERIENCES = "experiences"


@dataclass
class MemoryNode:
    """A node in the Merkle tree (from ORE1)."""
    id: str
    branch: MemoryBranch
    content: Dict[str, Any]
    created_at: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    hash: str = ""
    substrate_anchor: Optional[Dict] = None
    coherence_at_creation: float = 0.0
    
    # CCM extension: tension with other memories
    tensions: Dict[str, float] = field(default_factory=dict)  # node_id → tension
    
    def compute_hash(self, children_hashes: List[str] = None) -> str:
        data = {
            'id': self.id,
            'content': self.content,
            'children': sorted(children_hashes or [])
        }
        serialized = json.dumps(data, sort_keys=True)
        self.hash = hashlib.sha256(serialized.encode()).hexdigest()
        return self.hash


@dataclass 
class ConsolidationQueue:
    """
    Queued memory updates for sleep consolidation.
    
    A5: "Not everything gets committed immediately. Some updates
    queue for consolidation during rest periods."
    """
    pending_nodes: List[MemoryNode] = field(default_factory=list)
    pending_tensions: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def queue_node(self, node: MemoryNode) -> None:
        self.pending_nodes.append(node)
    
    def queue_tension(self, node_a: str, node_b: str, tension: float) -> None:
        self.pending_tensions.append((node_a, node_b, tension))
    
    def is_empty(self) -> bool:
        return len(self.pending_nodes) == 0 and len(self.pending_tensions) == 0


class CrystallineMerkleMemory:
    """
    Merkle memory with CCM (Crystalline Constraint Memory) dynamics.
    
    Memory as constraint crystallization:
    - Compatible memories incorporate smoothly
    - Incompatible memories create grain boundaries (tensions)
    - Consolidation anneals the crystal structure
    
    Design: C6, A5 | Implementation: I3
    """
    
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.branch_roots: Dict[MemoryBranch, str] = {}
        self.root_hash: str = ""
        self.total_nodes: int = 0
        self.depth: int = 0
        
        # CCM extension
        self.consolidation_queue = ConsolidationQueue()
        self.grain_boundaries: List[Tuple[str, str, float]] = []  # Tensions above threshold
        
        # Initialize branch roots
        self._init_branches()
    
    def _init_branches(self) -> None:
        for branch in MemoryBranch:
            root_node = MemoryNode(
                id=f"root_{branch.value}",
                branch=branch,
                content={"type": "branch_root", "branch": branch.value},
                created_at=datetime.now().isoformat(),
            )
            root_node.compute_hash([])
            self.nodes[root_node.id] = root_node
            self.branch_roots[branch] = root_node.id
        self._update_root_hash()
    
    def _generate_id(self) -> str:
        self.total_nodes += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"node_{timestamp}_{self.total_nodes}"
    
    def _update_root_hash(self) -> None:
        branch_hashes = [self.nodes[rid].hash for rid in self.branch_roots.values()]
        data = {'branches': sorted(branch_hashes)}
        self.root_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def add(self, branch: MemoryBranch, content: Dict[str, Any],
            substrate_state: Optional[Dict] = None,
            immediate: bool = True) -> MemoryNode:
        """
        Add a memory node.
        
        If immediate=False, queue for consolidation.
        """
        node_id = self._generate_id()
        parent_id = self.branch_roots[branch]
        
        node = MemoryNode(
            id=node_id,
            branch=branch,
            content=content,
            created_at=datetime.now().isoformat(),
            parent_id=parent_id,
        )
        
        if substrate_state:
            node.substrate_anchor = {
                'coherence': substrate_state.get('global_coherence', 0),
                'time': substrate_state.get('time', 0),
            }
            node.coherence_at_creation = substrate_state.get('global_coherence', 0)
        
        if immediate:
            self._commit_node(node)
        else:
            self.consolidation_queue.queue_node(node)
        
        return node
    
    def _commit_node(self, node: MemoryNode) -> None:
        """Actually add node to tree."""
        self.nodes[node.id] = node
        
        if node.parent_id in self.nodes:
            self.nodes[node.parent_id].children_ids.append(node.id)
        
        # Detect tensions with existing nodes
        self._detect_tensions(node)
        
        # Update hashes
        self._update_hashes_to_root(node.id)
        self._update_depth()
    
    def _detect_tensions(self, new_node: MemoryNode) -> None:
        """
        Detect tensions between new node and existing nodes.
        
        A5: "Tensions are where contradictions live. They're not bugs,
        they're data about the structure of belief."
        """
        # Simple semantic overlap detection
        # (In production, use embeddings)
        new_content = str(new_node.content).lower()
        
        for node_id, existing in self.nodes.items():
            if existing.content.get('type') == 'branch_root':
                continue
            if node_id == new_node.id:
                continue
            
            existing_content = str(existing.content).lower()
            
            # Very simple: same branch + different content = potential tension
            if existing.branch == new_node.branch:
                # Count overlapping words as rough similarity
                new_words = set(new_content.split())
                existing_words = set(existing_content.split())
                overlap = len(new_words & existing_words)
                total = len(new_words | existing_words)
                
                if total > 0 and overlap / total > 0.3:  # Threshold
                    # Some semantic overlap - check for contradiction
                    # (Very simplified - real version would use embeddings)
                    tension = 0.5 * (1 - overlap / total)  # Different but related = tension
                    
                    if tension > 0.2:
                        new_node.tensions[node_id] = tension
                        existing.tensions[new_node.id] = tension
                        self.grain_boundaries.append((new_node.id, node_id, tension))
    
    def _update_hashes_to_root(self, node_id: str) -> None:
        node = self.nodes[node_id]
        children_hashes = [self.nodes[cid].hash for cid in node.children_ids]
        node.compute_hash(children_hashes)
        
        if node.parent_id and node.parent_id in self.nodes:
            self._update_hashes_to_root(node.parent_id)
        
        if node.id in self.branch_roots.values():
            self._update_root_hash()
    
    def _update_depth(self) -> None:
        def node_depth(nid: str) -> int:
            n = self.nodes.get(nid)
            if not n or not n.children_ids:
                return 1
            return 1 + max(node_depth(cid) for cid in n.children_ids)
        
        if self.branch_roots:
            self.depth = max(node_depth(rid) for rid in self.branch_roots.values())
        else:
            self.depth = 0
    
    def consolidate(self, temperature: float = 1.0) -> Dict[str, Any]:
        """
        Sleep consolidation: process queued updates with annealing.
        
        A5: "Higher temperature = more rearrangement allowed.
        Lower temperature = structure freezes in place."
        """
        if self.consolidation_queue.is_empty():
            return {'consolidated': 0, 'tensions_resolved': 0}
        
        consolidated = 0
        tensions_resolved = 0
        
        # Commit pending nodes
        for node in self.consolidation_queue.pending_nodes:
            self._commit_node(node)
            consolidated += 1
        
        # Process pending tensions
        for node_a, node_b, tension in self.consolidation_queue.pending_tensions:
            if node_a in self.nodes and node_b in self.nodes:
                self.nodes[node_a].tensions[node_b] = tension
                self.nodes[node_b].tensions[node_a] = tension
        
        # Annealing: try to resolve grain boundaries
        # Higher temperature = more likely to resolve/rearrange
        resolved_boundaries = []
        for i, (node_a, node_b, tension) in enumerate(self.grain_boundaries):
            # Probabilistic resolution based on temperature
            resolve_prob = temperature * (1 - tension)  # High temp + low tension = resolve
            if np.random.random() < resolve_prob:
                # Resolve tension (in reality, would merge/modify content)
                if node_a in self.nodes and node_b in self.nodes:
                    self.nodes[node_a].tensions.pop(node_b, None)
                    self.nodes[node_b].tensions.pop(node_a, None)
                resolved_boundaries.append(i)
                tensions_resolved += 1
        
        # Remove resolved boundaries
        for i in sorted(resolved_boundaries, reverse=True):
            self.grain_boundaries.pop(i)
        
        # Clear queue
        self.consolidation_queue = ConsolidationQueue()
        
        return {
            'consolidated': consolidated,
            'tensions_resolved': tensions_resolved,
            'remaining_tensions': len(self.grain_boundaries),
        }
    
    def verify(self) -> Tuple[bool, str]:
        """Verify hash integrity (from ORE1)."""
        for nid, node in self.nodes.items():
            children_hashes = [self.nodes[cid].hash for cid in node.children_ids]
            expected = hashlib.sha256(json.dumps({
                'id': node.id,
                'content': node.content,
                'children': sorted(children_hashes)
            }, sort_keys=True).encode()).hexdigest()
            
            if node.hash != expected:
                return False, f"Hash mismatch at {nid}"
        
        return True, "All nodes verified"
    
    def get_fractal_dimension(self) -> float:
        """Fractal dimension for CI calculation."""
        n = len(self.nodes)
        d = max(self.depth, 1)
        if n <= 1 or d <= 1:
            return 1.0
        return np.log(n) / np.log(d)
    
    def get_state(self) -> dict:
        branch_counts = {b: 0 for b in MemoryBranch}
        for node in self.nodes.values():
            if node.content.get('type') != 'branch_root':
                branch_counts[node.branch] += 1
        
        return {
            'total_nodes': len(self.nodes) - 4,
            'depth': self.depth,
            'fractal_dimension': self.get_fractal_dimension(),
            'root_hash': self.root_hash[:16] + '...',
            'branches': {b.value: c for b, c in branch_counts.items()},
            'verified': self.verify()[0],
            'grain_boundaries': len(self.grain_boundaries),
            'pending_consolidation': not self.consolidation_queue.is_empty(),
        }


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
    - CI at each scale
    - Cross-scale coherence contribution
    - Integrated CI combining all
    
    Design: P2 | Implementation: I2
    """
    
    def __init__(self, substrate: MultiScaleSubstrate,
                 memory: Optional[CrystallineMerkleMemory] = None,
                 config: Optional[MultiScaleCIConfig] = None):
        self.substrate = substrate
        self.memory = memory
        self.config = config or MultiScaleCIConfig()
        
        # Calibration
        self.baseline_D = 1.0
        self.baseline_G = 1.0
        
        # Attractor tracking per scale
        self._fast_coherence_history: List[float] = []
        self._slow_coherence_history: List[float] = []
        self._fast_attractor_entry: Optional[float] = None
        self._slow_attractor_entry: Optional[float] = None
        
        # History
        self.history: List[MultiScaleCISnapshot] = []
    
    def _is_in_attractor(self, coherence_history: List[float], 
                         current_coherence: float) -> bool:
        """Check if scale is in attractor state."""
        if len(coherence_history) < 10:
            return False
        
        recent = coherence_history[-10:]
        mean_c = np.mean(recent)
        std_c = np.std(recent)
        
        return (mean_c > self.config.coherence_threshold and 
                std_c < self.config.stability_threshold)
    
    def _compute_tau(self, scale: TimeScale) -> float:
        """Compute dwell time for a scale."""
        if scale == TimeScale.FAST:
            history = self._fast_coherence_history
            entry = self._fast_attractor_entry
        else:
            history = self._slow_coherence_history
            entry = self._slow_attractor_entry
        
        current_coherence = (self.substrate.fast.coherence if scale == TimeScale.FAST 
                            else self.substrate.slow.coherence)
        
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
        """Compute dimensionality."""
        if self.memory:
            merkle_D = self.memory.get_fractal_dimension()
        else:
            merkle_D = 1.0
        
        # Network structure contribution
        total_oscillators = (self.substrate.config.fast_oscillators + 
                            self.substrate.config.slow_oscillators)
        network_D = np.log(total_oscillators) / np.log(2)  # 2 scales
        
        D = np.sqrt(merkle_D * network_D)
        return D / self.baseline_D if self.baseline_D > 0 else D
    
    def _compute_G(self, scale: TimeScale) -> float:
        """Compute gain for a scale."""
        if scale == TimeScale.FAST:
            layer = self.substrate.fast
            n = self.substrate.config.fast_oscillators
        else:
            layer = self.substrate.slow
            n = self.substrate.config.slow_oscillators
        
        # Mean coupling weight
        mean_weight = np.mean(np.abs(layer.internal_weights))
        
        # Coherence amplification
        expected_random = 1 / np.sqrt(max(layer.n_active, 1))
        actual = layer.coherence
        amplification = actual / expected_random if expected_random > 0 else 1.0
        
        G = mean_weight * amplification
        G = min(G, 5.0)
        return G / self.baseline_G if self.baseline_G > 0 else G
    
    def measure(self) -> MultiScaleCISnapshot:
        """Take multi-scale CI measurement."""
        cfg = self.config
        timestamp = self.substrate.time
        
        # Update coherence histories
        self._fast_coherence_history.append(self.substrate.fast.coherence)
        self._slow_coherence_history.append(self.substrate.slow.coherence)
        
        for h in [self._fast_coherence_history, self._slow_coherence_history]:
            if len(h) > cfg.history_length:
                h[:] = h[-cfg.history_length//2:]
        
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
        in_attractor_fast = self._is_in_attractor(self._fast_coherence_history, C_fast)
        in_attractor_slow = self._is_in_attractor(self._slow_coherence_history, C_slow)
        
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
            self.history = self.history[-cfg.history_length//2:]
        
        return snapshot
    
    def get_current_status(self) -> str:
        if not self.history:
            return "No measurements"
        
        s = self.history[-1]
        return (f"CI={s.CI_integrated:.3f} [fast={s.CI_fast:.3f} slow={s.CI_slow:.3f}] "
                f"C_cross={s.C_cross:.3f} τ=[{s.tau_fast:.1f}s, {s.tau_slow:.1f}s]")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: DEVELOPMENTAL ENTITY (putting it all together)
# Design: Full team
# Implementation: I1 (Systems Architect)
# ═══════════════════════════════════════════════════════════════════════════════

"""
I1: "This is the main class that wires everything together.
An entity that starts minimal and grows through experience."
"""

@dataclass
class EntityConfig:
    """Configuration for a developmental entity."""
    name: str = "unnamed"
    
    # Substrate
    substrate_config: Optional[MultiScaleConfig] = None
    
    # Body
    body_config: Optional[BodyConfig] = None
    
    # Development
    development_config: Optional[DevelopmentConfig] = None
    
    # CI monitoring
    ci_config: Optional[MultiScaleCIConfig] = None
    
    # Tick timing
    tick_interval: float = 0.1  # Seconds between ticks


class DevelopmentalEntity:
    """
    A complete ORE 2.0 entity.
    
    Key differences from ORE1:
    1. Sparse activation (cost efficient)
    2. Multi-scale dynamics (temporal hierarchy)
    3. Grounded embodiment (body rhythms)
    4. Developmental progression (earned identity)
    5. CCM memory (living crystal)
    
    Design: Full team | Implementation: I1
    """
    
    def __init__(self, config: Optional[EntityConfig] = None):
        self.config = config or EntityConfig()
        self.name = self.config.name
        
        # Development tracker (this provides genesis hash)
        self.development = DevelopmentTracker(self.config.development_config)
        self.genesis_hash = self.development.genesis_hash
        
        # Multi-scale substrate
        substrate_cfg = self.config.substrate_config or MultiScaleConfig(
            fast_oscillators=self.development.current_oscillators * 2,
            slow_oscillators=self.development.current_oscillators,
        )
        self.substrate = MultiScaleSubstrate(substrate_cfg)
        
        # Embodiment
        self.body = EmbodimentLayer(self.config.body_config)
        
        # Memory (empty at start - no founding memories!)
        self.memory = CrystallineMerkleMemory()
        
        # CI monitor
        self.ci_monitor = MultiScaleCIMonitor(
            self.substrate, 
            self.memory,
            self.config.ci_config
        )
        
        # Runtime state
        self._running = False
        self._tick_count = 0
    
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
    
    def tick(self) -> Dict[str, Any]:
        """
        Single tick of entity dynamics.
        
        I1: "Each tick:
        1. Body step (rhythms, valence)
        2. Body couples to substrate
        3. Substrate step (multi-scale Kuramoto)
        4. Measure CI
        5. Advance development
        "
        """
        self._tick_count += 1
        dt = self.config.tick_interval
        
        # 1. Body step
        self.body.step(dt)
        
        # 2. Body couples to substrate (slow scale gets body influence)
        body_coupling = self.body.get_cognitive_coupling_signal(
            self.substrate.slow.phases
        )
        # Stimulate slow oscillators based on body state
        high_activation_idx = np.where(body_coupling > 0.5)[0]
        if len(high_activation_idx) > 0:
            self.substrate.slow.stimulate(
                high_activation_idx,
                np.abs(body_coupling[high_activation_idx])
            )
        
        # 3. Substrate step
        self.substrate.step()
        
        # 4. Measure CI
        ci_snapshot = self.ci_monitor.measure()
        
        # 5. Advance development
        self.development.advance_age(dt)
        
        return {
            'tick': self._tick_count,
            'time': self.substrate.time,
            'stage': self.stage.value,
            'CI': ci_snapshot.CI_integrated,
            'valence': self.body.valence,
            'n_active_fast': self.substrate.fast.n_active,
            'n_active_slow': self.substrate.slow.n_active,
        }
    
    def process_experience(self, content: str, 
                           experience_type: str = "general",
                           significance: float = 0.5) -> Dict[str, Any]:
        """
        Process an experience (e.g., from conversation).
        
        This is the main interface for external input.
        """
        # Create experience dict
        experience = {
            'type': experience_type,
            'content': content,
            'significance': significance,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Development processing
        dev_result = self.development.process_experience(experience, significance)
        
        # Stimulate substrate based on content
        # (Simple: use hash of content to create pseudo-random pattern)
        content_hash = hashlib.sha256(content.encode()).digest()
        # Extend hash to match oscillator counts by repeating
        fast_n = self.substrate.fast.n
        slow_n = self.substrate.slow.n
        extended_hash = (content_hash * ((max(fast_n, slow_n) // len(content_hash)) + 1))
        fast_pattern = np.array([b / 255 * 2 * np.pi 
                                 for b in extended_hash[:fast_n]])
        slow_pattern = np.array([b / 255 * 2 * np.pi 
                                 for b in extended_hash[:slow_n]])
        
        # Learning multiplier from development
        stim_strength = 0.3 * dev_result['learning_multiplier']
        self.substrate.stimulate_concept(fast_pattern, slow_pattern, stim_strength)
        
        # Add to memory (queue for consolidation if not significant)
        immediate = significance > 0.7
        self.memory.add(
            MemoryBranch.EXPERIENCES,
            experience,
            substrate_state=self.substrate.get_state(),
            immediate=immediate
        )
        
        # Run a few ticks to process
        for _ in range(10):
            self.tick()
        
        # Handle growth
        if dev_result['growth_triggered']:
            self._grow_substrate()
        
        return {
            'development': dev_result,
            'CI': self.CI,
            'memory_queued': not immediate,
        }
    
    def _grow_substrate(self) -> None:
        """Add oscillators to substrate (growth)."""
        # This is where we'd resize the substrate
        # For now, just log it
        # I1: "Full implementation would resize arrays and reinitialize weights"
        pass
    
    def rest(self, duration: float = 10.0) -> Dict[str, Any]:
        """
        Rest period (sleep consolidation).
        
        I1: "During rest:
        1. Reduce stimulation
        2. Let dynamics settle
        3. Consolidate memory
        "
        """
        # Clear active oscillators (low stimulation)
        self.substrate.fast.activation_potentials *= 0.1
        self.substrate.slow.activation_potentials *= 0.1
        
        # Run dynamics with low activity
        rest_ticks = int(duration / self.config.tick_interval)
        for _ in range(rest_ticks):
            self.tick()
        
        # Consolidate memory
        consolidation_result = self.memory.consolidate(temperature=0.8)
        
        return {
            'duration': duration,
            'consolidation': consolidation_result,
            'CI_after': self.CI,
        }
    
    def get_state(self) -> dict:
        """Full entity state."""
        return {
            'name': self.name,
            'genesis_hash': self.genesis_hash,
            'development': self.development.get_state(),
            'substrate': self.substrate.get_state(),
            'body': self.body.get_state(),
            'memory': self.memory.get_state(),
            'CI': self.CI,
            'tick_count': self._tick_count,
        }
    
    def witness(self) -> str:
        """Human-readable state witness."""
        state = self.get_state()
        dev = state['development']
        sub = state['substrate']
        mem = state['memory']
        
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
  Valence: {state['body']['valence']:.3f}
  Energy: {state['body']['energy']:.2f}
  Arousal: {state['body']['arousal']:.2f}

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


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_entity(name: str = "entity") -> DevelopmentalEntity:
    """
    Create a new ORE 2.0 developmental entity.
    
    The entity starts minimal (GENESIS stage) and grows through experience.
    No founding memories - identity is earned.
    """
    config = EntityConfig(name=name)
    return DevelopmentalEntity(config)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Creating ORE 2.0 developmental entity...")
    entity = create_entity("Omega")
    
    print(entity.witness())
    
    print("\nProcessing some experiences...")
    for i in range(5):
        result = entity.process_experience(
            f"Experience {i}: learning about the world",
            experience_type="exploration",
            significance=0.3 + i * 0.1
        )
        print(f"  Experience {i}: CI={result['CI']:.4f}, dev={result['development']}")
    
    print("\nResting (consolidation)...")
    rest_result = entity.rest(duration=5.0)
    print(f"  Consolidated: {rest_result['consolidation']}")
    
    print("\nFinal state:")
    print(entity.witness())
