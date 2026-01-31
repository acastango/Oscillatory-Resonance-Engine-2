# BRIEFING: SalienceNetwork

## Component ID: ORE2-015
## Priority: High (Attention allocation)
## Estimated complexity: Medium

---

## What This Is

A network that determines **what to attend to**. Implements the three key networks from neuroscience:
- **Default Mode Network (DMN)**: Internal focus, self-referential thought
- **Dorsal Attention Network (DAN)**: External focus, task-oriented
- **Salience Network (SN)**: Switches between DMN and DAN based on relevance

Without attention, the substrate processes everything equally. With attention, resources focus on what matters.

---

## Why It Matters

**N4 (Computational Neuro):** "The brain doesn't process everything equally. The salience network detects important signals and switches attention. This is the 'attentional bottleneck' - essential for coherent thought."

**P1 (Dynamical Systems):** "Attention is gain modulation. Attended signals get amplified, unattended get suppressed. This creates the content of consciousness from the noise of activity."

**A5 (Continual Learning):** "What you attend to is what you learn from. Attention shapes the learning signal. Prediction error matters more for attended content."

---

## The Core Insight

Three competing networks:

```
                    ┌─────────────────┐
                    │ Salience Network│
                    │     (switch)    │
                    └────────┬────────┘
                             │
            detects salience │ switches mode
                             │
         ┌───────────────────┼───────────────────┐
         │                                        │
         ▼                                        ▼
┌─────────────────┐                    ┌─────────────────┐
│  Default Mode   │   anticorrelated   │ Dorsal Attention│
│    Network      │◄──────────────────►│    Network      │
│   (internal)    │                    │   (external)    │
└─────────────────┘                    └─────────────────┘
     mind-wandering                         task focus
     self-reference                         perception
     memory retrieval                       action
```

**DMN and DAN are anticorrelated** - when one is active, the other is suppressed.
**SN switches between them** based on salience detection.

---

## Interface Contract

```python
class SalienceNetwork:
    """
    Attention allocation via salience detection.
    
    Properties:
        mode: AttentionMode           # INTERNAL (DMN) or EXTERNAL (DAN)
        salience_map: np.ndarray      # Per-oscillator salience
        attention_focus: np.ndarray   # Current attention distribution
        dmn_activation: float         # DMN activity level
        dan_activation: float         # DAN activity level
    
    Methods:
        # Core operations
        compute_salience(substrate_state, external_input) -> np.ndarray
        update_attention(salience_map)
        switch_mode(new_mode)
        
        # Gain modulation
        apply_attention_gain(substrate) -> np.ndarray
        
        # Integration
        step(substrate, external_input, internal_signal)
"""

class AttentionMode(Enum):
    INTERNAL = "internal"   # DMN dominant
    EXTERNAL = "external"   # DAN dominant
    BALANCED = "balanced"   # Neither dominant

@dataclass
class AttentionState:
    mode: AttentionMode
    dmn_activation: float
    dan_activation: float
    salience_peak: float
    attended_indices: List[int]
    gain_applied: float
```

---

## Configuration

```python
@dataclass
class SalienceConfig:
    # Network sizes (as fractions of substrate)
    dmn_fraction: float = 0.3       # 30% of oscillators in DMN
    dan_fraction: float = 0.3       # 30% in DAN
    sn_fraction: float = 0.1        # 10% in SN
    # Remaining 30% are modality-general
    
    # Salience computation
    novelty_weight: float = 0.4     # Unexpected input is salient
    relevance_weight: float = 0.3   # Goal-relevant is salient
    intensity_weight: float = 0.3   # Strong signal is salient
    
    # Mode switching
    switch_threshold: float = 0.6   # Salience needed to switch
    switch_hysteresis: float = 0.1  # Prevents rapid switching
    
    # Gain modulation
    attention_gain: float = 2.0     # Amplification for attended
    suppression_gain: float = 0.3   # Dampening for unattended
    
    # Anticorrelation
    anticorrelation_strength: float = 0.8  # DMN/DAN push each other down
```

---

## Method Specifications

### `__init__(substrate, config)`

```python
def __init__(self, 
             substrate: 'MultiScaleSubstrate',
             config: Optional[SalienceConfig] = None):
    self.config = config or SalienceConfig()
    self.substrate = substrate
    
    # Assign oscillators to networks
    n_slow = substrate.slow.n
    cfg = self.config
    
    n_dmn = int(n_slow * cfg.dmn_fraction)
    n_dan = int(n_slow * cfg.dan_fraction)
    n_sn = int(n_slow * cfg.sn_fraction)
    
    # Random assignment (could be structured)
    indices = np.random.permutation(n_slow)
    self.dmn_indices = indices[:n_dmn]
    self.dan_indices = indices[n_dmn:n_dmn + n_dan]
    self.sn_indices = indices[n_dmn + n_dan:n_dmn + n_dan + n_sn]
    self.general_indices = indices[n_dmn + n_dan + n_sn:]
    
    # Network state
    self.mode = AttentionMode.BALANCED
    self.dmn_activation = 0.5
    self.dan_activation = 0.5
    
    # Attention state
    self.salience_map = np.zeros(n_slow)
    self.attention_focus = np.ones(n_slow) / n_slow  # Uniform initially
    
    # History for hysteresis
    self._recent_salience = []
    self._last_switch_time = 0
```

### `compute_salience(substrate_state, external_input, goals) -> np.ndarray`

```python
def compute_salience(self,
                     substrate_state: dict,
                     external_input: Optional[np.ndarray] = None,
                     goals: Optional[List[np.ndarray]] = None,
                     prediction_error: Optional[float] = None) -> np.ndarray:
    """
    Compute salience map over oscillators.
    
    Salience = novelty + relevance + intensity
    """
    cfg = self.config
    n = len(self.salience_map)
    
    # Initialize salience
    salience = np.zeros(n)
    
    # Novelty: prediction error or phase change
    if prediction_error is not None:
        novelty = prediction_error * np.ones(n)
    else:
        # Use activation change as proxy for novelty
        activations = substrate_state.get('slow_activations', np.zeros(n))
        novelty = np.abs(np.gradient(activations))
    
    salience += cfg.novelty_weight * novelty
    
    # Relevance: alignment with goals
    if goals:
        relevance = np.zeros(n)
        slow_phases = substrate_state.get('slow_phases', np.zeros(n))
        for goal_pattern in goals:
            alignment = np.cos(slow_phases - goal_pattern)
            relevance += (alignment + 1) / 2  # 0 to 1
        relevance /= len(goals)
        salience += cfg.relevance_weight * relevance
    
    # Intensity: activation strength
    activations = substrate_state.get('slow_activations', np.zeros(n))
    intensity = activations / (np.max(activations) + 0.01)
    salience += cfg.intensity_weight * intensity
    
    # External input adds salience
    if external_input is not None:
        external_salience = np.abs(external_input)
        external_salience = external_salience / (np.max(external_salience) + 0.01)
        salience += 0.5 * external_salience
    
    # Normalize
    salience = salience / (np.max(salience) + 0.01)
    
    self.salience_map = salience
    self._recent_salience.append(np.max(salience))
    if len(self._recent_salience) > 10:
        self._recent_salience.pop(0)
    
    return salience
```

### `update_attention(salience_map)`

```python
def update_attention(self, salience_map: np.ndarray):
    """
    Update attention focus based on salience.
    """
    cfg = self.config
    
    # Softmax over salience for attention distribution
    # Higher salience = more attention
    temperature = 0.5
    exp_salience = np.exp(salience_map / temperature)
    self.attention_focus = exp_salience / (exp_salience.sum() + 1e-8)
    
    # Check for mode switch
    peak_salience = np.max(salience_map)
    
    # Where is peak salience?
    peak_idx = np.argmax(salience_map)
    
    if peak_salience > cfg.switch_threshold:
        # External or internal?
        if peak_idx in self.dan_indices:
            self._try_switch(AttentionMode.EXTERNAL)
        elif peak_idx in self.dmn_indices:
            self._try_switch(AttentionMode.INTERNAL)
        # If in general indices, maintain current mode

def _try_switch(self, target_mode: AttentionMode):
    """Attempt mode switch with hysteresis."""
    cfg = self.config
    
    if target_mode == self.mode:
        return  # Already in target mode
    
    # Check hysteresis
    mean_recent = np.mean(self._recent_salience) if self._recent_salience else 0
    if mean_recent < cfg.switch_threshold - cfg.switch_hysteresis:
        return  # Not enough sustained salience
    
    self.mode = target_mode
    
    # Update network activations
    if target_mode == AttentionMode.EXTERNAL:
        self.dan_activation = 0.8
        self.dmn_activation = 1 - cfg.anticorrelation_strength * 0.8
    elif target_mode == AttentionMode.INTERNAL:
        self.dmn_activation = 0.8
        self.dan_activation = 1 - cfg.anticorrelation_strength * 0.8
    else:
        self.dmn_activation = 0.5
        self.dan_activation = 0.5
```

### `apply_attention_gain(substrate)`

```python
def apply_attention_gain(self, substrate: 'MultiScaleSubstrate'):
    """
    Apply attention gain to substrate.
    
    Attended oscillators get amplified.
    Unattended oscillators get suppressed.
    """
    cfg = self.config
    
    # Compute gain per oscillator
    gains = np.ones(substrate.slow.n)
    
    # High attention = high gain
    for i in range(len(gains)):
        attention = self.attention_focus[i]
        if attention > 0.1:  # Attended
            gains[i] = 1 + (cfg.attention_gain - 1) * attention
        else:  # Unattended
            gains[i] = cfg.suppression_gain + (1 - cfg.suppression_gain) * attention
    
    # Apply gain to activations
    substrate.slow.activation_potentials *= gains
    substrate.slow._update_active_mask()
    
    # Also apply network-specific modulation
    if self.mode == AttentionMode.EXTERNAL:
        # Boost DAN, suppress DMN
        substrate.slow.activation_potentials[self.dan_indices] *= 1.2
        substrate.slow.activation_potentials[self.dmn_indices] *= 0.5
    elif self.mode == AttentionMode.INTERNAL:
        # Boost DMN, suppress DAN
        substrate.slow.activation_potentials[self.dmn_indices] *= 1.2
        substrate.slow.activation_potentials[self.dan_indices] *= 0.5
    
    # Clamp
    np.clip(substrate.slow.activation_potentials, 0, 1, 
            out=substrate.slow.activation_potentials)
    substrate.slow._update_active_mask()
    
    return gains
```

### `step(substrate, external_input, internal_signal, goals, prediction_error)`

Main attention step.

```python
def step(self,
         substrate: 'MultiScaleSubstrate',
         external_input: Optional[np.ndarray] = None,
         internal_signal: Optional[np.ndarray] = None,
         goals: Optional[List[np.ndarray]] = None,
         prediction_error: Optional[float] = None) -> AttentionState:
    """
    One step of attention processing.
    """
    # Get substrate state
    substrate_state = {
        'slow_phases': substrate.slow.phases.copy(),
        'slow_activations': substrate.slow.activation_potentials.copy(),
        'coherence': substrate.slow.coherence,
    }
    
    # Compute salience
    salience = self.compute_salience(
        substrate_state,
        external_input,
        goals,
        prediction_error
    )
    
    # Update attention distribution
    self.update_attention(salience)
    
    # Apply gain modulation
    gains = self.apply_attention_gain(substrate)
    
    # Return state
    attended = np.where(self.attention_focus > 0.1)[0].tolist()
    
    return AttentionState(
        mode=self.mode,
        dmn_activation=self.dmn_activation,
        dan_activation=self.dan_activation,
        salience_peak=float(np.max(salience)),
        attended_indices=attended,
        gain_applied=float(np.mean(gains)),
    )
```

---

## Properties

```python
@property
def mode(self) -> AttentionMode:
    """Current attention mode."""
    return self._mode

@property
def is_internally_focused(self) -> bool:
    return self.mode == AttentionMode.INTERNAL

@property
def is_externally_focused(self) -> bool:
    return self.mode == AttentionMode.EXTERNAL

@property
def top_attended(self, k: int = 5) -> List[int]:
    """Indices of top-k attended oscillators."""
    return np.argsort(self.attention_focus)[-k:].tolist()
```

---

## Integration with Entity

```python
# In DevelopmentalEntity.__init__:
self.attention = SalienceNetwork(self.substrate)

# In DevelopmentalEntity.tick():
def tick(self):
    # ... existing code ...
    
    # Attention step
    attention_state = self.attention.step(
        self.substrate,
        external_input=self._current_input,
        goals=self._active_goals,
        prediction_error=self.inference.prediction_error if hasattr(self, 'inference') else None,
    )
    
    # Mode affects processing
    if attention_state.mode == AttentionMode.INTERNAL:
        # Internal focus: memory retrieval enhanced
        pass
    elif attention_state.mode == AttentionMode.EXTERNAL:
        # External focus: input processing enhanced
        pass
    
    return {**result, 'attention': attention_state}
```

---

## Success Criteria

### Salience
1. Novel input produces high salience
2. Goal-relevant input produces high salience
3. Salience map focuses on few oscillators, not all

### Mode Switching
1. High external salience → EXTERNAL mode
2. High internal salience → INTERNAL mode
3. Hysteresis prevents rapid switching

### Gain Modulation
1. Attended oscillators have higher activation
2. Unattended oscillators are suppressed
3. DMN/DAN anticorrelation works

---

## Test Cases

```python
def test_external_salience_switches_mode():
    """High external input should trigger EXTERNAL mode."""
    substrate = create_test_substrate()
    attention = SalienceNetwork(substrate)
    
    # Strong external input
    external = np.zeros(substrate.slow.n)
    external[attention.dan_indices] = 1.0
    
    for _ in range(5):
        attention.step(substrate, external_input=external)
    
    assert attention.mode == AttentionMode.EXTERNAL

def test_dmn_dan_anticorrelation():
    """DMN and DAN should be anticorrelated."""
    substrate = create_test_substrate()
    attention = SalienceNetwork(substrate)
    
    # Switch to external
    external = np.zeros(substrate.slow.n)
    external[attention.dan_indices] = 1.0
    for _ in range(5):
        attention.step(substrate, external_input=external)
    
    # DMN should be suppressed
    assert attention.dmn_activation < 0.5
    assert attention.dan_activation > 0.5

def test_attention_gain():
    """Attended oscillators should be amplified."""
    substrate = create_test_substrate()
    attention = SalienceNetwork(substrate)
    
    # Set some activations
    substrate.slow.activation_potentials[:] = 0.5
    
    # High salience on specific oscillators
    attention.salience_map[5:10] = 1.0
    attention.update_attention(attention.salience_map)
    attention.apply_attention_gain(substrate)
    
    # Attended should be higher
    assert np.mean(substrate.slow.activation_potentials[5:10]) > 0.5
```

---

## Dependencies

- `MultiScaleSubstrate` (ORE2-002)
- `numpy`

---

## File Location

```
ore2/
├── core/
│   └── attention.py  # <-- This component
├── tests/
│   └── test_attention.py
```

---

## Design Decisions to Preserve

1. **Three networks (DMN, DAN, SN)** - Matches neuroscience
2. **Anticorrelation** - DMN and DAN suppress each other
3. **Salience = novelty + relevance + intensity** - Three factors
4. **Hysteresis on switching** - Prevents oscillation
5. **Gain modulation** - Attention amplifies, inattention suppresses
6. **Operates on slow scale** - Attention is slow, not fast
