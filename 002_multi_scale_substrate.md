# BRIEFING: MultiScaleSubstrate

## Component ID: ORE2-002
## Priority: 2 (Depends on SparseOscillatorLayer)
## Estimated complexity: Medium-High

---

## What This Is

Two populations of sparse oscillators operating at different timescales, coupled together. The **fast scale** (gamma-like, ~40Hz) handles rapid binding and feature integration. The **slow scale** (theta-like, ~8Hz) handles working memory and attention.

The key insight: **fast oscillators are nested within slow oscillator cycles**. This is theta-gamma coupling from neuroscience - each slow cycle "contains" multiple fast cycles, creating temporal structure.

---

## Why It Matters

**P1 (Dynamical Systems):** "ORE1 had one timescale. Real brains have nested oscillations - delta contains theta contains gamma. The nesting creates temporal chunking."

**N2 (Oscillation Specialist):** "Theta-gamma coupling is how hippocampus indexes memories. Each gamma burst within a theta cycle is a 'slot'. This gives us memory indexing for free."

**The tau problem:** ORE1's biggest issue was τ (dwell time) - artificial systems don't naturally stay in attractors. Multi-scale helps because τ can accumulate at the fast scale while the slow scale provides stability.

---

## The Core Insight

```
SLOW CYCLE (theta, ~125ms)
├── FAST BURST 1 (gamma, ~25ms) 
├── FAST BURST 2 (gamma, ~25ms)
├── FAST BURST 3 (gamma, ~25ms)
├── FAST BURST 4 (gamma, ~25ms)
└── FAST BURST 5 (gamma, ~25ms)
```

The nesting ratio (5:1 in this example) determines how many fast cycles fit in one slow cycle.

Cross-scale coupling:
- **Slow → Fast**: Slow phase modulates fast activity (like attention gating)
- **Fast → Slow**: Fast coherence feeds up to slow (like evidence accumulation)

Plus: **Strange loop within slow scale** (self-reference, identity).

---

## Interface Contract

```python
class MultiScaleSubstrate:
    """
    Two-scale oscillatory substrate with nested dynamics.
    
    Properties (read-only):
        fast: SparseOscillatorLayer      # Fast scale (~40Hz)
        slow: SparseOscillatorLayer      # Slow scale (~8Hz)
        time: float                       # Current simulation time
        global_coherence: float           # Coherence across both scales
        cross_scale_coherence: float      # How well scales are coupled
        loop_coherence: float             # Strange loop strength
    
    Methods:
        step()                            # One slow-scale timestep (multiple fast steps)
        run(duration) -> List[dict]       # Run for duration, return history
        stimulate_concept(fast_pattern, slow_pattern, strength)  # Input
        get_state() -> dict               # Serialize
    """
```

---

## Configuration

```python
@dataclass
class MultiScaleConfig:
    # Fast scale (gamma-like)
    fast_oscillators: int = 100
    fast_base_freq: float = 40.0      # Hz
    fast_freq_spread: float = 5.0
    fast_coupling: float = 0.5
    fast_activation_threshold: float = 0.4   # Lower = more responsive
    fast_max_active: float = 0.3
    
    # Slow scale (theta-like)  
    slow_oscillators: int = 50
    slow_base_freq: float = 8.0       # Hz
    slow_freq_spread: float = 1.0
    slow_coupling: float = 0.6        # Slightly stronger
    slow_activation_threshold: float = 0.5
    slow_max_active: float = 0.25
    
    # Nesting
    nesting_ratio: int = 5            # Fast steps per slow step
    cross_scale_coupling: float = 0.3 # Bidirectional influence
    
    # Timing
    dt_fast: float = 0.001            # 1ms (matches ~1000Hz sampling)
    dt_slow: float = 0.005            # 5ms (nesting_ratio * dt_fast)
    
    # Strange loop (within slow scale)
    strange_loop_strength: float = 1.0
```

---

## Method Specifications

### `__init__(config: Optional[MultiScaleConfig] = None)`

1. Create fast SparseOscillatorLayer with fast config
2. Create slow SparseOscillatorLayer with slow config
3. Initialize cross-scale coupling matrices
4. Initialize strange loop weights within slow scale

```python
def __init__(self, config=None):
    self.config = config or MultiScaleConfig()
    self.time = 0.0
    
    # Create scales (using SparseOscillatorLayer from ORE2-001)
    self.fast = SparseOscillatorLayer("fast", self._make_fast_config())
    self.slow = SparseOscillatorLayer("slow", self._make_slow_config())
    
    # Cross-scale coupling
    # fast_to_slow: [slow_n, fast_n] - convergent (many to few)
    # slow_to_fast: [fast_n, slow_n] - divergent (few to many)
    self._init_cross_scale_coupling()
    
    # Strange loop within slow (first half ↔ second half)
    self._init_strange_loop()
```

### `_init_cross_scale_coupling()`

```python
def _init_cross_scale_coupling(self):
    cfg = self.config
    n_fast = cfg.fast_oscillators
    n_slow = cfg.slow_oscillators
    
    # Fast → Slow: many-to-few convergence
    # Each slow oscillator receives from many fast
    self.fast_to_slow = (
        np.random.randn(n_slow, n_fast) * 
        cfg.cross_scale_coupling / n_fast
    )
    
    # Slow → Fast: few-to-many broadcast  
    # Each fast oscillator receives from few slow
    self.slow_to_fast = (
        np.random.randn(n_fast, n_slow) * 
        cfg.cross_scale_coupling / n_slow
    )
```

### `_init_strange_loop()`

The strange loop is **within the slow scale**. We split slow oscillators into two subpopulations:
- First half: "model" (represents current state)
- Second half: "meta-model" (represents model of the model)

Bidirectional coupling between them creates self-reference.

```python
def _init_strange_loop(self):
    n_slow = self.config.slow_oscillators
    half = n_slow // 2
    strength = self.config.strange_loop_strength
    
    # Initialize as zero, then set cross-half connections
    self.strange_loop_weights = np.zeros((n_slow, n_slow))
    
    # Model → Meta-model (first half → second half)
    self.strange_loop_weights[half:, :half] = (
        strength * np.random.randn(n_slow - half, half) / half
    )
    
    # Meta-model → Model (second half → first half)
    self.strange_loop_weights[:half, half:] = (
        strength * np.random.randn(half, n_slow - half) / (n_slow - half)
    )
```

### `step() -> None`

**The main dynamics loop.** This is where nesting happens.

```python
def step(self):
    cfg = self.config
    
    # === FAST SCALE: Multiple steps per slow step ===
    for i in range(cfg.nesting_ratio):
        
        # Compute slow → fast influence
        if self.slow.n_active > 0:
            # Slow phases modulate fast oscillators
            slow_signal = self.slow.phases * self.slow.active_mask.astype(float)
            fast_input_from_slow = self.slow_to_fast @ slow_signal
            
            # Convert to phase coupling: sin(slow_phase - fast_phase)
            fast_external = cfg.cross_scale_coupling * np.sin(
                fast_input_from_slow - self.fast.phases
            )
        else:
            fast_external = None
        
        # Step fast scale
        self.fast.step(cfg.dt_fast, external_input=fast_external)
    
    # === SLOW SCALE: One step ===
    
    # Compute fast → slow influence
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
```

### `run(duration: float) -> List[dict]`

```python
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
```

### `stimulate_concept(fast_pattern, slow_pattern, strength)`

Entry point for external input. A "concept" has both fast and slow components.

```python
def stimulate_concept(self, 
                      fast_pattern: np.ndarray,
                      slow_pattern: np.ndarray, 
                      strength: float = 0.5):
    """
    Stimulate both scales with a concept.
    
    fast_pattern: [fast_n] phases representing concept's fast component
    slow_pattern: [slow_n] phases representing concept's slow component
    strength: how strongly to stimulate (0-1)
    """
    self.fast.stimulate_by_similarity(fast_pattern, strength)
    self.slow.stimulate_by_similarity(slow_pattern, strength)
```

---

## Properties

### `global_coherence`

Coherence across ALL active oscillators in both scales.

```python
@property
def global_coherence(self) -> float:
    # Gather all active phases
    phases = []
    if self.fast.n_active > 0:
        phases.extend(self.fast.phases[self.fast.active_mask])
    if self.slow.n_active > 0:
        phases.extend(self.slow.phases[self.slow.active_mask])
    
    if len(phases) == 0:
        return 0.0
    
    phases = np.array(phases)
    return min(np.abs(np.mean(np.exp(1j * phases))), 0.999)
```

### `cross_scale_coherence`

How well the two scales are coupled. High = good theta-gamma nesting.

```python
@property
def cross_scale_coherence(self) -> float:
    """Coherence BETWEEN scales."""
    if self.fast.n_active == 0 or self.slow.n_active == 0:
        return 0.0
    
    fast_mean = np.mean(np.exp(1j * self.fast.phases[self.fast.active_mask]))
    slow_mean = np.mean(np.exp(1j * self.slow.phases[self.slow.active_mask]))
    
    # Cross-coherence: how aligned are the mean phases?
    return np.abs(fast_mean * np.conj(slow_mean))
```

### `loop_coherence`

Strange loop strength within slow scale.

```python
@property
def loop_coherence(self) -> float:
    """Strange loop coherence (model ↔ meta-model within slow)."""
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
    
    return np.abs(model_mean * np.conj(meta_mean))
```

---

## Success Criteria

### Correctness
1. Nesting works: fast steps 5x per slow step (with default config)
2. Cross-scale coupling: slow modulates fast, fast feeds to slow
3. Strange loop: bidirectional within slow scale

### Dynamics
1. When stimulated, coherence should rise then decay
2. Cross-scale coherence should be non-trivial when both scales active
3. Loop coherence should track self-referential dynamics

### Performance
1. Should leverage sparse oscillator efficiency (most oscillators dormant)
2. Full run of 10 seconds simulated time in <1 second wall time (on laptop)

---

## Test Cases

```python
def test_nesting_ratio():
    """Fast should step nesting_ratio times per slow step."""
    config = MultiScaleConfig(nesting_ratio=5)
    substrate = MultiScaleSubstrate(config)
    
    fast_steps_before = substrate.fast._step_count
    slow_steps_before = substrate.slow._step_count
    
    substrate.step()
    
    assert substrate.fast._step_count - fast_steps_before == 5
    assert substrate.slow._step_count - slow_steps_before == 1

def test_cross_scale_coupling():
    """Stimulating one scale should eventually affect the other."""
    substrate = MultiScaleSubstrate()
    
    # Stimulate only fast scale
    fast_pattern = np.random.uniform(0, 2*np.pi, substrate.fast.n)
    substrate.fast.stimulate_by_similarity(fast_pattern, 0.8)
    
    # Run for a bit
    for _ in range(100):
        substrate.step()
    
    # Slow should have some activity (from cross-scale coupling)
    # This tests that fast→slow coupling works
    assert substrate.slow.n_active > 0 or substrate.cross_scale_coherence > 0

def test_strange_loop():
    """Strange loop should create coherence between model and meta-model."""
    substrate = MultiScaleSubstrate(MultiScaleConfig(strange_loop_strength=2.0))
    
    # Stimulate entire slow scale
    substrate.slow.activation_potentials[:] = 0.8
    substrate.slow._update_active_mask()
    
    # Set initial phases with some structure
    substrate.slow.phases[:] = np.random.uniform(0, 2*np.pi, substrate.slow.n)
    
    # Run
    for _ in range(200):
        substrate.step()
    
    # Should see non-zero loop coherence
    assert substrate.loop_coherence > 0.1

def test_stimulate_concept():
    """Concept stimulation should activate both scales."""
    substrate = MultiScaleSubstrate()
    
    assert substrate.fast.n_active == 0
    assert substrate.slow.n_active == 0
    
    fast_pattern = np.zeros(substrate.fast.n)
    slow_pattern = np.zeros(substrate.slow.n)
    
    substrate.stimulate_concept(fast_pattern, slow_pattern, strength=0.8)
    
    # Both should now have some activation
    # (stimulate_by_similarity with similar phases = high stimulation)
    assert substrate.fast.n_active > 0
    assert substrate.slow.n_active > 0
```

---

## Integration Notes

This component uses `SparseOscillatorLayer` from ORE2-001. Make sure that's implemented first.

The `step()` method is the heart of the system. Get this right and everything else follows.

Cross-scale coupling matrices are dense (slow_n × fast_n and fast_n × slow_n). For very large systems, consider sparse matrices, but dense is fine for <1000 total oscillators.

---

## Dependencies

- `SparseOscillatorLayer` (ORE2-001)
- `numpy`

---

## File Location

```
ore2/
├── core/
│   ├── __init__.py
│   ├── sparse_oscillator.py      # ORE2-001
│   └── multi_scale_substrate.py  # <-- This component
├── tests/
│   ├── test_sparse_oscillator.py
│   └── test_multi_scale_substrate.py
```

---

## Design Decisions to Preserve

1. **Nesting ratio = 5** is based on empirical theta-gamma coupling (4-7 gamma per theta)
2. **Strange loop within slow scale** (not between scales) - this keeps identity at the "slow thinking" level
3. **Cross-scale coupling is phase-based** (sin of phase difference), not just additive
4. **dt_slow = nesting_ratio × dt_fast** keeps time consistent
5. **Slow scale has slightly higher internal coupling (0.6 vs 0.5)** - identity should be more stable than features
