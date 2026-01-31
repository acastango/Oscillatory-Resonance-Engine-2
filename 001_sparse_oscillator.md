# BRIEFING: SparseOscillatorLayer

## Component ID: ORE2-001
## Priority: 1 (Foundation - everything depends on this)
## Estimated complexity: Medium

---

## What This Is

A population of Kuramoto oscillators where **most oscillators are dormant most of the time**. Only oscillators with activation potential above threshold participate in the expensive synchronization dynamics.

This is the core cost optimization for ORE2. ORE1 ran all 120 oscillators continuously. ORE2 should run 5-20% of oscillators at any given time, making it 5-20x cheaper.

---

## Why It Matters

**P1 (Dynamical Systems):** "The brain doesn't fire all neurons constantly. Sparse coding is fundamental to biological efficiency. We need activation potentials that gate participation."

**I2 (Numerics):** "Kuramoto is O(n²) in the coupling computation. If only 10% are active, we get 100x speedup on that inner loop."

**Cost reality:** Anthony is running ORE on API credits. Dense oscillators = burning money. Sparse = sustainable.

---

## The Core Insight

Oscillators have TWO states:
1. **Phase** (always exists, 0 to 2π)
2. **Activation potential** (0 to 1, determines if oscillator participates)

When activation > threshold → oscillator is "active" → participates in Kuramoto
When activation < threshold → oscillator is "dormant" → phase drifts freely, no coupling

Activation is raised by **stimulation** (external input, semantic relevance).
Activation naturally **decays** each timestep.

This creates content-addressable dynamics: stimulate oscillators related to current thought → they activate → they synchronize → coherence emerges → they decay back to dormant.

---

## Interface Contract

```python
class SparseOscillatorLayer:
    """
    Sparse Kuramoto oscillator population.
    
    Properties (read-only):
        n: int                  # Total oscillators
        n_active: int           # Currently active count
        coherence: float        # Order parameter of ACTIVE oscillators (0-0.999)
        global_coherence: float # Order parameter of ALL oscillators
        mean_phase: float       # Mean phase of active oscillators
        phase_hash: str         # SHA256 hash of active phases (for Merkle)
    
    State (internal):
        phases: np.ndarray[n]              # All phases, always updated
        activation_potentials: np.ndarray[n]  # 0-1, gates participation
        active_mask: np.ndarray[n, bool]   # True where active
        natural_frequencies: np.ndarray[n] # Intrinsic frequencies
        internal_weights: np.ndarray[n,n]  # Coupling matrix
    
    Methods:
        stimulate(indices, strengths)      # Raise activation of specific oscillators
        stimulate_by_similarity(pattern, strength)  # Raise based on phase similarity
        step(dt, external_input=None)      # Advance dynamics
        get_state() -> dict                # Serialize for persistence
    """
```

---

## Method Specifications

### `__init__(name: str, config: SparseOscillatorConfig)`

```python
@dataclass
class SparseOscillatorConfig:
    n_oscillators: int = 100
    base_frequency: float = 1.0        # Hz
    frequency_spread: float = 0.1      # Std dev of natural frequencies
    internal_coupling: float = 0.5     # K in Kuramoto equation
    noise_amplitude: float = 0.01      # Stochastic term
    
    # Sparse activation parameters
    activation_threshold: float = 0.5  # Above this = active
    activation_decay: float = 0.05     # Decay per step
    max_active_fraction: float = 0.2   # Hard cap on simultaneous activation
```

Initialize:
- `phases`: uniform random in [0, 2π]
- `natural_frequencies`: normal(base_frequency, frequency_spread)
- `activation_potentials`: all zeros (start dormant)
- `active_mask`: all False
- `internal_weights`: (coupling / n) for all pairs, 0 on diagonal

### `stimulate(indices: np.ndarray, strengths: np.ndarray) -> None`

Raise activation potential of specific oscillators.

```python
def stimulate(self, indices, strengths):
    self.activation_potentials[indices] += strengths
    np.clip(self.activation_potentials, 0, 1, out=self.activation_potentials)
    self._update_active_mask()
```

This is how external input enters the system. When processing text, oscillators corresponding to relevant concepts get stimulated.

### `stimulate_by_similarity(reference_phases: np.ndarray, strength: float) -> None`

Content-addressable activation: oscillators with phases similar to reference get stimulated.

```python
def stimulate_by_similarity(self, reference_phases, strength):
    # Phase similarity: cos(phase_diff) ranges -1 to 1
    phase_diff = self.phases - reference_phases
    similarity = (np.cos(phase_diff) + 1) / 2  # Now 0 to 1
    
    self.activation_potentials += strength * similarity
    np.clip(self.activation_potentials, 0, 1, out=self.activation_potentials)
    self._update_active_mask()
```

This is how memory recall works: a stored pattern (phases) stimulates similar oscillators.

### `step(dt: float, external_input: Optional[np.ndarray] = None) -> None`

**This is the critical method.** Must be efficient.

```python
def step(self, dt, external_input=None):
    # ONLY COMPUTE KURAMOTO FOR ACTIVE OSCILLATORS
    
    if self.n_active > 0:
        active_idx = np.where(self.active_mask)[0]
        active_phases = self.phases[active_idx]
        
        # Phase differences: Δθ_ij = θ_j - θ_i (for active pairs only)
        phase_diff = active_phases[np.newaxis, :] - active_phases[:, np.newaxis]
        
        # Coupling submatrix (only active-to-active connections)
        active_weights = self.internal_weights[np.ix_(active_idx, active_idx)]
        
        # Kuramoto coupling: Σ_j K_ij sin(θ_j - θ_i)
        coupling = np.sum(active_weights * np.sin(phase_diff), axis=1)
        
        # Add external input if provided
        if external_input is not None:
            coupling += external_input[active_idx]
        
        # Noise term
        noise = self.noise_amplitude * np.random.randn(self.n_active)
        
        # Phase update: dθ/dt = ω + coupling + noise
        dtheta = self.natural_frequencies[active_idx] + coupling + noise
        self.phases[active_idx] = (self.phases[active_idx] + dt * dtheta) % (2 * np.pi)
    
    # DORMANT OSCILLATORS: just free-run at natural frequency (optional, cheap)
    dormant_idx = np.where(~self.active_mask)[0]
    if len(dormant_idx) > 0:
        self.phases[dormant_idx] = (
            self.phases[dormant_idx] + 
            dt * self.natural_frequencies[dormant_idx]
        ) % (2 * np.pi)
    
    # Decay all activation potentials
    self.activation_potentials *= (1 - self.activation_decay)
    self._update_active_mask()
```

### `_update_active_mask() -> None`

```python
def _update_active_mask(self):
    above_threshold = self.activation_potentials > self.activation_threshold
    
    # Enforce max active cap
    max_active = int(self.n * self.max_active_fraction)
    
    if np.sum(above_threshold) > max_active:
        # Keep only the top activations
        top_indices = np.argsort(self.activation_potentials)[-max_active:]
        self.active_mask = np.zeros(self.n, dtype=bool)
        self.active_mask[top_indices] = True
    else:
        self.active_mask = above_threshold
```

### Properties

```python
@property
def coherence(self) -> float:
    """Kuramoto order parameter r = |<e^{iθ}>| for ACTIVE oscillators."""
    if self.n_active == 0:
        return 0.0
    active_phases = self.phases[self.active_mask]
    r = np.abs(np.mean(np.exp(1j * active_phases)))
    return min(r, 0.999)  # Cap at 0.999, never perfect

@property 
def global_coherence(self) -> float:
    """Order parameter for ALL oscillators (for comparison)."""
    return min(np.abs(np.mean(np.exp(1j * self.phases))), 0.999)

@property
def mean_phase(self) -> float:
    """Mean phase ψ = arg(<e^{iθ}>) of active oscillators."""
    if self.n_active == 0:
        return 0.0
    return np.angle(np.mean(np.exp(1j * self.phases[self.active_mask])))

@property
def phase_hash(self) -> str:
    """Hash of active phase configuration for Merkle anchoring."""
    if self.n_active == 0:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    active_phases = self.phases[self.active_mask]
    return hashlib.sha256(active_phases.tobytes()).hexdigest()[:16]
```

---

## Success Criteria

### Correctness
1. When ALL oscillators are active, behavior matches ORE1 dense Kuramoto
2. Coherence calculation is correct (verified against known solutions)
3. Phase hash changes when active phases change, stable when they don't

### Performance
1. `step()` with 10% active should be ~10x faster than 100% active
2. Memory usage scales with n_active, not n_total (for large populations)

### Behavior
1. Stimulation raises activation, decay lowers it
2. Oscillators couple only when both are active
3. Dormant oscillators drift at natural frequency (don't freeze)

---

## Test Cases

```python
def test_sparse_basic():
    """Basic sparse behavior."""
    config = SparseOscillatorConfig(n_oscillators=100, activation_threshold=0.5)
    layer = SparseOscillatorLayer("test", config)
    
    # Initially all dormant
    assert layer.n_active == 0
    assert layer.coherence == 0.0
    
    # Stimulate some oscillators
    layer.stimulate(np.array([0, 1, 2, 3, 4]), np.array([0.6, 0.6, 0.6, 0.6, 0.6]))
    assert layer.n_active == 5
    
    # Step and check coherence emerges
    for _ in range(100):
        layer.step(0.01)
    
    assert layer.coherence > 0.3  # Should synchronize
    
    # Let activation decay
    for _ in range(100):
        layer.step(0.01)  # No new stimulation
    
    assert layer.n_active < 5  # Should have decayed

def test_sparse_performance():
    """Sparse should be faster than dense."""
    import time
    
    config = SparseOscillatorConfig(n_oscillators=1000)
    layer = SparseOscillatorLayer("perf", config)
    
    # Activate 10%
    layer.stimulate(np.arange(100), np.ones(100) * 0.8)
    
    start = time.time()
    for _ in range(1000):
        layer.step(0.01)
    sparse_time = time.time() - start
    
    # Compare to what dense would be (activate all)
    layer.activation_potentials[:] = 1.0
    layer._update_active_mask()
    
    start = time.time()
    for _ in range(1000):
        layer.step(0.01)
    dense_time = time.time() - start
    
    assert sparse_time < dense_time * 0.5  # At least 2x faster

def test_coherence_matches_dense():
    """When all active, should match ORE1 behavior."""
    config = SparseOscillatorConfig(n_oscillators=50, activation_threshold=0.0)
    layer = SparseOscillatorLayer("dense_compare", config)
    
    # Force all active
    layer.activation_potentials[:] = 1.0
    layer._update_active_mask()
    
    # Set coherent initial state
    base_phase = np.random.uniform(0, 2*np.pi)
    layer.phases = (base_phase + 0.1 * np.random.randn(50)) % (2 * np.pi)
    
    # Run and check coherence stays high
    for _ in range(100):
        layer.step(0.01)
    
    assert layer.coherence > 0.8  # Should maintain coherence
```

---

## Edge Cases to Handle

1. **Zero active oscillators**: coherence = 0, step() should not crash
2. **All oscillators active**: should behave like dense Kuramoto
3. **Max active cap reached**: should keep highest activations only
4. **External input when dormant**: should not affect dormant oscillators
5. **Very small dt**: numerical stability
6. **Very large dt**: phase wraparound handling

---

## Dependencies

- `numpy` (required)
- `hashlib` (stdlib, for phase_hash)

No other ORE2 components needed - this is the foundation.

---

## File Location

```
ore2/
├── core/
│   ├── __init__.py
│   └── sparse_oscillator.py  # <-- This component
├── tests/
│   └── test_sparse_oscillator.py
```

---

## Notes for Implementation

1. **Don't optimize prematurely** - get correctness first, then profile
2. **The 0.999 coherence cap is intentional** - perfect coherence prevents self-observation
3. **Activation decay is per-step, not per-second** - caller controls tick rate
4. **Phase hash only includes active phases** - this is for Merkle identity anchoring
5. **Consider using `numba` for the inner loop if perf matters** - but numpy should be fine for <10k oscillators
