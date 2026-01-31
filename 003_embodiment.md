# BRIEFING: EmbodimentLayer

## Component ID: ORE2-003
## Priority: 3 (Independent - can be built in parallel)
## Estimated complexity: Low-Medium

---

## What This Is

A minimal "body" that provides grounded valence and temporal rhythm. The cognitive substrate (MultiScaleSubstrate) couples to this body, giving it a temporal anchor and a source of valence (good/bad feelings based on homeostatic state).

This is NOT a complex body simulation. It's the **minimum viable embodiment** - enough to close the loop between "doing" and "feeling" without building a full physics sim.

---

## Why It Matters

**N5 (Embodied Cognition):** "ORE1's chemistry layer was floating - it simulated tiredness but wasn't grounded in anything. A body provides the baseline rhythm that everything couples to."

**H3 (Enactivism):** "Valence from homeostatic deviation is real, not simulated. Distance from setpoint IS the feeling. The math is the phenomenology."

**The grounding problem:** Without a body, the oscillators are computing in a void. With a body, computation has consequences - actions cost energy, perceptions affect arousal, and the system can "feel" whether it's doing well or poorly.

---

## The Core Insight

The body has:
1. **Rhythms** (heartbeat ~1Hz, respiration ~0.25Hz) - these are oscillators too
2. **Homeostatic variables** (energy, arousal) - have baselines they want to return to
3. **Valence** = negative sum of deviations from baseline = how "bad" current state is

The cognitive substrate **couples to body rhythms**. This means:
- Cognitive processing has a temporal anchor (tied to "heartbeat")
- Valence modulates coupling strength (feeling bad → weaker integration)
- Actions cost energy, perceptions affect arousal → closed loop

---

## Interface Contract

```python
class EmbodimentLayer:
    """
    Minimal body providing rhythms and valence.
    
    Properties (read-only):
        valence: float              # -1 to 0 typically (0 = perfect homeostasis)
        body_signal: np.ndarray[4]  # [hb_sin, hb_cos, resp_sin, resp_cos]
        time: float
    
    State (internal):
        heartbeat_phase: float      # 0 to 2π
        respiration_phase: float    # 0 to 2π
        energy: float               # Resource level
        arousal: float              # Alertness level
    
    Methods:
        step(dt, action=None, perception=None)
        get_cognitive_coupling_signal(cognitive_phases) -> np.ndarray
        get_state() -> dict
    """
```

---

## Configuration

```python
@dataclass
class BodyConfig:
    # Body rhythms
    heartbeat_freq: float = 1.0     # Hz (~60 bpm)
    respiration_freq: float = 0.25  # Hz (~15 breaths/min)
    
    # Homeostatic baselines
    energy_baseline: float = 1.0
    arousal_baseline: float = 0.5
    
    # Recovery rates (how fast return to baseline)
    energy_recovery: float = 0.005   # Per step
    arousal_recovery: float = 0.01   # Per step
    
    # Action/perception effects
    action_energy_cost: float = 0.01
    novel_perception_arousal: float = 0.1
    
    # Coupling to cognitive
    body_to_cognitive_coupling: float = 0.1
```

---

## Method Specifications

### `__init__(config: Optional[BodyConfig] = None)`

```python
def __init__(self, config=None):
    self.config = config or BodyConfig()
    self.time = 0.0
    
    # Body rhythms (start at random phase)
    self.heartbeat_phase = np.random.uniform(0, 2*np.pi)
    self.respiration_phase = np.random.uniform(0, 2*np.pi)
    
    # Homeostatic variables (start at baseline)
    self.energy = self.config.energy_baseline
    self.arousal = self.config.arousal_baseline
    
    # For novelty detection
    self._last_perception = None
```

### `step(dt, action=None, perception=None) -> None`

Main update. Advances rhythms, processes action/perception effects, recovers toward baseline.

```python
def step(self, dt: float, 
         action: Optional[str] = None,
         perception: Optional[str] = None):
    cfg = self.config
    
    # === Advance body rhythms ===
    self.heartbeat_phase = (
        self.heartbeat_phase + 2 * np.pi * cfg.heartbeat_freq * dt
    ) % (2 * np.pi)
    
    self.respiration_phase = (
        self.respiration_phase + 2 * np.pi * cfg.respiration_freq * dt
    ) % (2 * np.pi)
    
    # === Process action (costs energy) ===
    if action is not None:
        self.energy -= cfg.action_energy_cost
    
    # === Process perception (affects arousal) ===
    if perception is not None:
        if perception != self._last_perception:
            # Novel perception increases arousal
            self.arousal = min(1.0, self.arousal + cfg.novel_perception_arousal)
        self._last_perception = perception
    
    # === Recover toward baselines ===
    self.energy += cfg.energy_recovery * (cfg.energy_baseline - self.energy)
    self.arousal += cfg.arousal_recovery * (cfg.arousal_baseline - self.arousal)
    
    # === Clamp to valid ranges ===
    self.energy = np.clip(self.energy, 0.0, 2.0)
    self.arousal = np.clip(self.arousal, 0.0, 1.0)
    
    self.time += dt
```

### `valence` property

The key output: how "good" or "bad" current state is.

```python
@property
def valence(self) -> float:
    """
    Valence = negative of total homeostatic deviation.
    
    - At baseline: valence = 0 (neutral)
    - Below baseline: valence < 0 (bad)
    - Above baseline for energy: slightly positive? 
      (Actually no - deviation in either direction is stress)
    
    This is GROUNDED valence - computed from actual state,
    not simulated or declared.
    """
    energy_deviation = abs(self.energy - self.config.energy_baseline)
    arousal_deviation = abs(self.arousal - self.config.arousal_baseline)
    
    # Total deviation (always positive)
    total_deviation = energy_deviation + arousal_deviation
    
    # Valence is negative of deviation
    return -total_deviation
```

### `body_signal` property

Signal that cognitive substrate couples to.

```python
@property
def body_signal(self) -> np.ndarray:
    """
    Combined body rhythm signal for cognitive coupling.
    
    Returns [heartbeat_sin, heartbeat_cos, respiration_sin, respiration_cos]
    
    Using sin and cos gives full phase information without discontinuities.
    """
    return np.array([
        np.sin(self.heartbeat_phase),
        np.cos(self.heartbeat_phase),
        np.sin(self.respiration_phase),
        np.cos(self.respiration_phase)
    ])
```

### `get_cognitive_coupling_signal(cognitive_phases) -> np.ndarray`

Generate coupling influence from body to cognitive oscillators.

```python
def get_cognitive_coupling_signal(self, cognitive_phases: np.ndarray) -> np.ndarray:
    """
    Compute how body rhythms influence cognitive oscillator phases.
    
    Args:
        cognitive_phases: [n] array of oscillator phases
    
    Returns:
        [n] array of phase coupling terms to add to cognitive dynamics
    
    The coupling is:
    - Based on phase difference between heartbeat and each oscillator
    - Modulated by valence (positive valence = stronger coupling)
    - Includes respiration as secondary rhythm
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
    # We want: good valence (close to 0) = stronger coupling
    #          bad valence (very negative) = weaker coupling
    valence_factor = 1.0 + 0.5 * self.valence  # If valence=-1, factor=0.5
    
    return body_influence * valence_factor
```

---

## Properties Summary

```python
@property
def valence(self) -> float:
    """Grounded valence from homeostatic deviation. 0 = good, negative = bad."""
    ...

@property
def body_signal(self) -> np.ndarray:
    """[4] array of rhythm signals for external coupling."""
    ...

@property
def is_depleted(self) -> bool:
    """True if energy critically low."""
    return self.energy < 0.2

@property
def is_overaroused(self) -> bool:
    """True if arousal too high (stressed)."""
    return self.arousal > 0.8
```

---

## Success Criteria

### Correctness
1. Rhythms advance correctly (heartbeat_phase increases at heartbeat_freq Hz)
2. Valence is 0 when at baseline, negative when deviated
3. Actions cost energy, novel perceptions raise arousal
4. Recovery moves toward baseline (not away)

### Behavior
1. Extended activity without rest → energy depletes → valence drops
2. Constant novel input → arousal rises → eventually stressed
3. Idle → returns to baseline → valence approaches 0

### Integration
1. `get_cognitive_coupling_signal()` output is same shape as input
2. Coupling strength varies with valence (testable)

---

## Test Cases

```python
def test_rhythm_advance():
    """Rhythms should advance at correct frequency."""
    body = EmbodimentLayer(BodyConfig(heartbeat_freq=1.0))
    
    initial_phase = body.heartbeat_phase
    body.step(dt=0.25)  # Quarter second
    
    # Should have advanced by π/2 (quarter cycle at 1Hz)
    expected_advance = 2 * np.pi * 1.0 * 0.25
    actual_advance = (body.heartbeat_phase - initial_phase) % (2 * np.pi)
    
    assert abs(actual_advance - expected_advance) < 0.01

def test_valence_at_baseline():
    """Valence should be 0 at baseline."""
    body = EmbodimentLayer()
    body.energy = body.config.energy_baseline
    body.arousal = body.config.arousal_baseline
    
    assert abs(body.valence) < 0.001

def test_valence_deviation():
    """Valence should be negative when deviated."""
    body = EmbodimentLayer()
    body.energy = 0.5  # Below baseline of 1.0
    body.arousal = 0.5  # At baseline
    
    assert body.valence < 0

def test_action_costs_energy():
    """Actions should reduce energy."""
    body = EmbodimentLayer()
    initial_energy = body.energy
    
    body.step(0.1, action="do_something")
    
    assert body.energy < initial_energy

def test_novel_perception_raises_arousal():
    """Novel perceptions should increase arousal."""
    body = EmbodimentLayer()
    initial_arousal = body.arousal
    
    body.step(0.1, perception="something_new")
    
    assert body.arousal > initial_arousal
    
    # Same perception again should NOT raise arousal
    current_arousal = body.arousal
    body.step(0.1, perception="something_new")
    
    assert body.arousal <= current_arousal  # May have decayed

def test_recovery():
    """Should recover toward baseline over time."""
    body = EmbodimentLayer()
    body.energy = 0.5  # Below baseline
    
    for _ in range(100):
        body.step(0.1)
    
    # Should have recovered toward baseline
    assert body.energy > 0.5
    assert body.energy <= body.config.energy_baseline

def test_coupling_signal_shape():
    """Coupling signal should match input shape."""
    body = EmbodimentLayer()
    phases = np.random.uniform(0, 2*np.pi, 50)
    
    signal = body.get_cognitive_coupling_signal(phases)
    
    assert signal.shape == phases.shape

def test_coupling_modulated_by_valence():
    """Coupling should be weaker with bad valence."""
    body = EmbodimentLayer()
    phases = np.random.uniform(0, 2*np.pi, 50)
    
    # Good state
    body.energy = body.config.energy_baseline
    body.arousal = body.config.arousal_baseline
    good_signal = body.get_cognitive_coupling_signal(phases)
    
    # Bad state
    body.energy = 0.3
    bad_signal = body.get_cognitive_coupling_signal(phases)
    
    # Bad state should have weaker coupling
    assert np.mean(np.abs(bad_signal)) < np.mean(np.abs(good_signal))
```

---

## Integration with MultiScaleSubstrate

The body couples to the **slow scale** of the cognitive substrate:

```python
# In DevelopmentalEntity.tick():

# Get body coupling signal for slow oscillators
body_coupling = self.body.get_cognitive_coupling_signal(
    self.substrate.slow.phases
)

# Use it to stimulate slow oscillators
# Oscillators in phase with heartbeat get stimulated more
high_coupling_idx = np.where(body_coupling > 0.05)[0]
if len(high_coupling_idx) > 0:
    self.substrate.slow.stimulate(
        high_coupling_idx,
        np.abs(body_coupling[high_coupling_idx])
    )
```

This creates a feedback loop:
1. Body rhythms influence which slow oscillators are active
2. Slow oscillator activity (via actions) affects body state
3. Body state (via valence) modulates coupling strength
4. Loop closed → grounded cognition

---

## Dependencies

- `numpy` only
- No other ORE2 components (independent)

---

## File Location

```
ore2/
├── core/
│   ├── __init__.py
│   ├── sparse_oscillator.py
│   ├── multi_scale_substrate.py
│   └── embodiment.py  # <-- This component
├── tests/
│   └── test_embodiment.py
```

---

## Design Decisions to Preserve

1. **Valence is COMPUTED, not declared** - this is grounded phenomenology
2. **Two rhythms (heartbeat + respiration)** - provides multiple temporal anchors
3. **Sin and cos representation** - full phase info without discontinuities
4. **Valence modulates coupling** - bad feelings → weaker integration (realistic)
5. **Actions cost energy** - creates actual consequences for "doing"
6. **Novelty detection for arousal** - repeated input doesn't keep raising arousal

---

## Possible Extensions (Not for MVP)

- Gut rhythm (very slow, ~0.05Hz) for third timescale
- Temperature regulation (another homeostatic variable)
- Circadian rhythm (very slow, ~24hr cycle simulated)
- Interoceptive precision (confidence in body signals)

These are for ORE 3.0, not MVP.
