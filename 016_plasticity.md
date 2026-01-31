# BRIEFING: HebbianPlasticity

## Component ID: ORE2-016
## Priority: High (Enables true learning)
## Estimated complexity: Medium

---

## What This Is

Dynamic plasticity in coupling weights based on the Hebbian principle: **"Neurons that fire together wire together."** Oscillators that synchronize become more strongly coupled; those that don't become decoupled.

This is how ORE learns associations, forms memories, and adapts to patterns over time.

---

## Why It Matters

**P1 (Dynamical Systems):** "Without plasticity, coupling weights are fixed. The system can't learn new associations. Hebbian learning is the fundamental mechanism for pattern storage in oscillator networks."

**N4 (Computational Neuro):** "Spike-timing dependent plasticity (STDP) is the biological version. Oscillator phase relationships map to spike timing. We can implement the same principles."

**A5 (Continual Learning):** "This completes the learning loop. Active inference updates the generative model. Hebbian learning updates the substrate itself. Both are needed."

---

## The Core Insight

Hebbian learning for oscillators:
- **Potentiation**: If oscillators i and j are **in-phase** (synchronized), strengthen coupling K_ij
- **Depression**: If oscillators i and j are **anti-phase** (180° out), weaken coupling
- **No change**: If phases are unrelated, coupling stays stable

```
θ_i ≈ θ_j (in-phase)      →  K_ij increases  →  future sync easier
θ_i ≈ θ_j + π (anti-phase) →  K_ij decreases →  future sync harder
|θ_i - θ_j| ≈ π/2         →  K_ij unchanged  →  neutral
```

Also implements:
- **Homeostatic plasticity**: Prevents runaway potentiation
- **Metaplasticity**: Learning rate depends on recent history
- **Attention modulation**: Attended patterns learn faster

---

## Interface Contract

```python
class HebbianPlasticity:
    """
    Hebbian learning for oscillator coupling weights.
    
    Properties:
        total_potentiation: float    # Cumulative strengthening
        total_depression: float      # Cumulative weakening
        learning_rate: float         # Current effective rate
        weight_histogram: np.ndarray # Distribution of weights
    
    Methods:
        # Core learning
        compute_weight_update(phases, activations) -> np.ndarray
        apply_update(substrate, update)
        
        # Homeostasis
        apply_homeostatic_scaling(substrate)
        
        # Modulation
        set_attention_mask(mask)
        set_learning_rate_multiplier(multiplier)
        
        # Integration
        step(substrate, attention_state)
"""

@dataclass
class PlasticityUpdate:
    """Weight update for one timestep."""
    delta_weights: np.ndarray      # [n, n] weight changes
    potentiation_indices: List[Tuple[int, int]]
    depression_indices: List[Tuple[int, int]]
    mean_change: float
    max_change: float
```

---

## Configuration

```python
@dataclass
class HebbianConfig:
    # Learning rates
    base_learning_rate: float = 0.001
    potentiation_rate: float = 0.002     # Rate for strengthening
    depression_rate: float = 0.001       # Rate for weakening (usually slower)
    
    # Phase thresholds
    potentiation_threshold: float = 0.5  # cos(phase_diff) > this → potentiate
    depression_threshold: float = -0.5   # cos(phase_diff) < this → depress
    
    # Activation gating
    activation_threshold: float = 0.3    # Both must be above this
    
    # Weight bounds
    min_weight: float = -0.5
    max_weight: float = 1.0
    initial_weight: float = 0.1
    
    # Homeostasis
    homeostatic_target: float = 0.5      # Target mean weight
    homeostatic_rate: float = 0.0001
    
    # Metaplasticity
    metaplasticity_window: int = 100     # Timesteps to track
    metaplasticity_threshold: float = 0.01  # Recent change threshold
    
    # Attention modulation
    attention_boost: float = 3.0         # Multiplier for attended pairs
```

---

## Method Specifications

### `__init__(substrate, config)`

```python
def __init__(self,
             substrate: 'MultiScaleSubstrate',
             config: Optional[HebbianConfig] = None):
    self.config = config or HebbianConfig()
    self.substrate = substrate
    
    # Track statistics
    self.total_potentiation = 0.0
    self.total_depression = 0.0
    
    # Metaplasticity tracking
    self._recent_changes: List[float] = []
    
    # Attention mask (defaults to all ones)
    self._attention_mask = np.ones((substrate.slow.n, substrate.slow.n))
    
    # Learning rate multiplier
    self._lr_multiplier = 1.0
```

### `compute_weight_update(phases, activations) -> PlasticityUpdate`

```python
def compute_weight_update(self,
                          phases: np.ndarray,
                          activations: np.ndarray) -> PlasticityUpdate:
    """
    Compute weight updates based on phase relationships.
    
    Hebbian rule: Δw_ij ∝ cos(θ_i - θ_j) * a_i * a_j
    
    where θ is phase and a is activation.
    """
    cfg = self.config
    n = len(phases)
    
    # Compute pairwise phase differences
    phase_diff = phases[:, np.newaxis] - phases[np.newaxis, :]
    
    # Hebbian signal: cos of phase difference
    # In-phase (diff≈0) → cos≈1 → potentiate
    # Anti-phase (diff≈π) → cos≈-1 → depress
    hebbian_signal = np.cos(phase_diff)
    
    # Activation gating: both must be active
    active_mask = (activations > cfg.activation_threshold)
    pair_active = active_mask[:, np.newaxis] & active_mask[np.newaxis, :]
    
    # Compute update
    delta = np.zeros((n, n))
    potentiation_pairs = []
    depression_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only
            if not pair_active[i, j]:
                continue
            
            signal = hebbian_signal[i, j]
            attention = self._attention_mask[i, j]
            
            if signal > cfg.potentiation_threshold:
                # Potentiate
                rate = cfg.potentiation_rate * self._lr_multiplier * attention
                change = rate * signal * activations[i] * activations[j]
                delta[i, j] = change
                delta[j, i] = change  # Symmetric
                potentiation_pairs.append((i, j))
                
            elif signal < cfg.depression_threshold:
                # Depress
                rate = cfg.depression_rate * self._lr_multiplier * attention
                change = rate * signal * activations[i] * activations[j]
                delta[i, j] = change  # Negative because signal < 0
                delta[j, i] = change
                depression_pairs.append((i, j))
    
    # Apply metaplasticity scaling
    meta_scale = self._compute_metaplasticity_scale()
    delta *= meta_scale
    
    return PlasticityUpdate(
        delta_weights=delta,
        potentiation_indices=potentiation_pairs,
        depression_indices=depression_pairs,
        mean_change=float(np.mean(np.abs(delta))),
        max_change=float(np.max(np.abs(delta))),
    )

def _compute_metaplasticity_scale(self) -> float:
    """
    Metaplasticity: reduce learning if too much recent change.
    
    This prevents runaway plasticity.
    """
    cfg = self.config
    
    if len(self._recent_changes) < 10:
        return 1.0
    
    recent_mean = np.mean(self._recent_changes[-cfg.metaplasticity_window:])
    
    if recent_mean > cfg.metaplasticity_threshold:
        # Too much recent change, slow down
        return cfg.metaplasticity_threshold / (recent_mean + 1e-8)
    
    return 1.0
```

### `apply_update(substrate, update)`

```python
def apply_update(self, 
                 substrate: 'MultiScaleSubstrate',
                 update: PlasticityUpdate,
                 scale: str = 'slow'):
    """
    Apply weight update to substrate.
    """
    cfg = self.config
    
    if scale == 'slow':
        weights = substrate.slow.internal_weights
    else:
        weights = substrate.fast.internal_weights
    
    # Apply update
    weights += update.delta_weights
    
    # Clamp to bounds
    np.clip(weights, cfg.min_weight, cfg.max_weight, out=weights)
    
    # Zero diagonal (no self-coupling)
    np.fill_diagonal(weights, 0)
    
    # Track statistics
    potentiation = np.sum(update.delta_weights[update.delta_weights > 0])
    depression = np.sum(np.abs(update.delta_weights[update.delta_weights < 0]))
    
    self.total_potentiation += potentiation
    self.total_depression += depression
    
    # Track for metaplasticity
    self._recent_changes.append(update.mean_change)
    if len(self._recent_changes) > self.config.metaplasticity_window * 2:
        self._recent_changes = self._recent_changes[-self.config.metaplasticity_window:]
```

### `apply_homeostatic_scaling(substrate)`

```python
def apply_homeostatic_scaling(self, 
                               substrate: 'MultiScaleSubstrate',
                               scale: str = 'slow'):
    """
    Homeostatic plasticity: keep mean weight near target.
    
    Prevents runaway potentiation or depression.
    """
    cfg = self.config
    
    if scale == 'slow':
        weights = substrate.slow.internal_weights
    else:
        weights = substrate.fast.internal_weights
    
    # Current mean (excluding diagonal)
    mask = ~np.eye(weights.shape[0], dtype=bool)
    current_mean = np.mean(weights[mask])
    
    # Scale toward target
    if abs(current_mean - cfg.homeostatic_target) > 0.01:
        scale_factor = 1 + cfg.homeostatic_rate * (cfg.homeostatic_target - current_mean)
        weights *= scale_factor
        
        # Re-clamp
        np.clip(weights, cfg.min_weight, cfg.max_weight, out=weights)
```

### `set_attention_mask(attention_state)`

```python
def set_attention_mask(self, attention_state: 'AttentionState'):
    """
    Set attention mask from SalienceNetwork state.
    
    Attended oscillator pairs learn faster.
    """
    cfg = self.config
    n = self._attention_mask.shape[0]
    
    # Reset to baseline
    self._attention_mask = np.ones((n, n))
    
    # Boost attended pairs
    attended = attention_state.attended_indices
    for i in attended:
        for j in attended:
            if i != j:
                self._attention_mask[i, j] = cfg.attention_boost
```

### `step(substrate, attention_state) -> PlasticityUpdate`

Main plasticity step.

```python
def step(self,
         substrate: 'MultiScaleSubstrate',
         attention_state: Optional['AttentionState'] = None) -> PlasticityUpdate:
    """
    One step of Hebbian plasticity.
    """
    # Update attention mask if provided
    if attention_state:
        self.set_attention_mask(attention_state)
    
    # Compute update for slow scale (main learning)
    update = self.compute_weight_update(
        substrate.slow.phases,
        substrate.slow.activation_potentials
    )
    
    # Apply update
    self.apply_update(substrate, update, scale='slow')
    
    # Periodic homeostatic scaling
    if len(self._recent_changes) % 100 == 0:
        self.apply_homeostatic_scaling(substrate, scale='slow')
    
    return update
```

---

## Properties

```python
@property
def weight_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of current weights."""
    weights = self.substrate.slow.internal_weights
    mask = ~np.eye(weights.shape[0], dtype=bool)
    return np.histogram(weights[mask], bins=50)

@property
def learning_rate(self) -> float:
    """Current effective learning rate."""
    meta_scale = self._compute_metaplasticity_scale()
    return self.config.base_learning_rate * self._lr_multiplier * meta_scale

@property
def plasticity_ratio(self) -> float:
    """Ratio of potentiation to depression."""
    if self.total_depression == 0:
        return float('inf') if self.total_potentiation > 0 else 1.0
    return self.total_potentiation / self.total_depression
```

---

## Integration with Entity

```python
# In DevelopmentalEntity.__init__:
self.plasticity = HebbianPlasticity(self.substrate)

# In DevelopmentalEntity.tick():
def tick(self):
    # ... existing code ...
    
    # Hebbian plasticity step
    attention_state = self.attention.step(...) if hasattr(self, 'attention') else None
    plasticity_update = self.plasticity.step(self.substrate, attention_state)
    
    # During critical periods, boost learning
    if self.development.stage in [DevelopmentStage.GENESIS, DevelopmentStage.BABBLING]:
        self.plasticity._lr_multiplier = 2.0
    else:
        self.plasticity._lr_multiplier = 1.0
    
    return {**result, 'plasticity': plasticity_update}
```

---

## Spike-Timing Dependent Plasticity (STDP) Variant

```python
def compute_stdp_update(self,
                        phases: np.ndarray,
                        activations: np.ndarray,
                        phase_history: List[np.ndarray]) -> PlasticityUpdate:
    """
    STDP-like update based on phase lead/lag.
    
    If oscillator i consistently leads oscillator j:
        → i drives j → strengthen i→j
    
    If oscillator i consistently lags oscillator j:
        → j drives i → weaken i→j (or strengthen j→i)
    """
    cfg = self.config
    n = len(phases)
    
    if len(phase_history) < 2:
        return PlasticityUpdate(np.zeros((n, n)), [], [], 0, 0)
    
    # Compute phase velocities
    current = phases
    previous = phase_history[-1]
    
    # Phase lead/lag
    delta = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if activations[i] < cfg.activation_threshold or \
               activations[j] < cfg.activation_threshold:
                continue
            
            # Phase difference now vs before
            diff_now = np.angle(np.exp(1j * (current[i] - current[j])))
            diff_prev = np.angle(np.exp(1j * (previous[i] - previous[j])))
            
            # If diff is decreasing, i is "catching up" to j (j leads)
            # If diff is increasing, i is "pulling ahead" (i leads)
            
            phase_velocity_diff = diff_now - diff_prev
            
            # STDP: leading oscillator gets stronger connection TO follower
            if phase_velocity_diff > 0.1:  # i leads
                delta[i, j] += cfg.potentiation_rate * phase_velocity_diff
            elif phase_velocity_diff < -0.1:  # j leads
                delta[j, i] += cfg.potentiation_rate * abs(phase_velocity_diff)
    
    # Make symmetric for now (could be asymmetric for directed networks)
    delta = (delta + delta.T) / 2
    
    return PlasticityUpdate(
        delta_weights=delta,
        potentiation_indices=[],
        depression_indices=[],
        mean_change=float(np.mean(np.abs(delta))),
        max_change=float(np.max(np.abs(delta))),
    )
```

---

## Success Criteria

### Basic Hebbian
1. Synchronized oscillators increase coupling
2. Anti-phase oscillators decrease coupling
3. Inactive pairs don't change

### Homeostasis
1. Mean weight stays near target
2. No runaway potentiation
3. Weights stay within bounds

### Attention Modulation
1. Attended pairs learn faster
2. Unattended pairs still learn (slower)
3. Attention mask updates correctly

---

## Test Cases

```python
def test_potentiation():
    """In-phase oscillators should strengthen coupling."""
    substrate = create_test_substrate()
    plasticity = HebbianPlasticity(substrate)
    
    # Force two oscillators in-phase and active
    substrate.slow.phases[0] = 0.0
    substrate.slow.phases[1] = 0.1  # Nearly in-phase
    substrate.slow.activation_potentials[0] = 0.8
    substrate.slow.activation_potentials[1] = 0.8
    
    weight_before = substrate.slow.internal_weights[0, 1]
    
    for _ in range(100):
        plasticity.step(substrate)
    
    weight_after = substrate.slow.internal_weights[0, 1]
    assert weight_after > weight_before

def test_depression():
    """Anti-phase oscillators should weaken coupling."""
    substrate = create_test_substrate()
    plasticity = HebbianPlasticity(substrate)
    
    # Force two oscillators anti-phase
    substrate.slow.phases[0] = 0.0
    substrate.slow.phases[1] = np.pi  # Anti-phase
    substrate.slow.activation_potentials[0] = 0.8
    substrate.slow.activation_potentials[1] = 0.8
    
    # Set initial weight
    substrate.slow.internal_weights[0, 1] = 0.5
    substrate.slow.internal_weights[1, 0] = 0.5
    
    for _ in range(100):
        plasticity.step(substrate)
    
    assert substrate.slow.internal_weights[0, 1] < 0.5

def test_homeostasis():
    """Weights should stay near target mean."""
    substrate = create_test_substrate()
    config = HebbianConfig(homeostatic_target=0.3)
    plasticity = HebbianPlasticity(substrate, config)
    
    # Many learning steps
    for _ in range(1000):
        # Random activations
        substrate.slow.activation_potentials = np.random.uniform(0, 1, substrate.slow.n)
        plasticity.step(substrate)
    
    # Check mean weight
    mask = ~np.eye(substrate.slow.n, dtype=bool)
    mean_weight = np.mean(substrate.slow.internal_weights[mask])
    assert abs(mean_weight - 0.3) < 0.1

def test_attention_boost():
    """Attended pairs should learn faster."""
    substrate = create_test_substrate()
    plasticity = HebbianPlasticity(substrate)
    
    # Create attention state with specific attended indices
    attention_state = AttentionState(
        mode=AttentionMode.EXTERNAL,
        dmn_activation=0.2,
        dan_activation=0.8,
        salience_peak=0.9,
        attended_indices=[0, 1, 2],
        gain_applied=1.5,
    )
    
    plasticity.set_attention_mask(attention_state)
    
    # Check mask
    assert plasticity._attention_mask[0, 1] > 1.0  # Boosted
    assert plasticity._attention_mask[0, 10] == 1.0  # Not boosted
```

---

## Dependencies

- `MultiScaleSubstrate` (ORE2-002)
- `SalienceNetwork` (ORE2-015) - optional
- `numpy`

---

## File Location

```
ore2/
├── core/
│   └── plasticity.py  # <-- This component
├── tests/
│   └── test_plasticity.py
```

---

## Design Decisions to Preserve

1. **Symmetric weights** - For now, undirected coupling
2. **Activation gating** - Only active oscillators learn
3. **Slower depression than potentiation** - Matches biology
4. **Homeostatic scaling** - Prevents instability
5. **Metaplasticity** - Slows learning if too much recent change
6. **Attention modulation** - Attended patterns learn faster
