# BRIEFING: DevelopmentTracker

## Component ID: ORE2-005
## Priority: 5 (Independent - can parallel)
## Estimated complexity: Low

---

## What This Is

Tracks developmental progression of an entity. Entities start minimal and **grow through experience**. Identity is **earned through development**, not configured via founding memories.

This replaces ORE1's "founding memories" pattern with developmental stages and critical periods.

---

## Why It Matters

**N7 (Developmental Neuroscience):** "Development isn't optional - it's how identity is earned. Critical periods exist because the brain NEEDS certain inputs at certain times. Miss the window, and that learning is harder forever."

**H3 (Enactivism):** "Autonomy is achieved, not given. An entity that starts with founding memories never had to BECOME itself."

**Trust:** An entity that develops its identity has a verifiable history. Founding memories could be anything - development creates proof of becoming.

---

## The Core Insight

Five developmental stages, each with characteristics:

1. **GENESIS** - Just born, establishing baseline rhythms
2. **BABBLING** - Random exploration, pattern discovery  
3. **IMITATION** - Strong coupling to external rhythms
4. **AUTONOMY** - Self-generated goals emerge
5. **MATURE** - Stable but still plastic

Critical periods enhance specific types of learning during specific stages.

The genesis hash is the **only** identity anchor at birth. Everything else is earned.

---

## Interface Contract

```python
class DevelopmentTracker:
    """
    Tracks developmental progression.
    
    Properties:
        genesis_hash: str                  # Identity anchor (immutable)
        stage: DevelopmentStage            # Current stage
        age: float                         # Developmental age
        stage_progress: float              # 0-1 through current stage
        current_oscillators: int           # How many oscillators earned
    
    Methods:
        process_experience(experience, significance) -> dict
        get_learning_multiplier(learning_type) -> float
        advance_age(dt)
        should_grow() -> bool
        get_state() -> dict
    """
```

---

## Data Structures

### DevelopmentStage

```python
class DevelopmentStage(Enum):
    GENESIS = "genesis"        # Just born, baseline
    BABBLING = "babbling"      # Random exploration
    IMITATION = "imitation"    # Coupling to external
    AUTONOMY = "autonomy"      # Self-generated goals
    MATURE = "mature"          # Stable, still plastic
```

### CriticalPeriod

```python
@dataclass
class CriticalPeriod:
    name: str                           # e.g., "language_acquisition"
    stage: DevelopmentStage             # When it's active
    learning_type: str                  # What type of learning is enhanced
    sensitivity: float = 2.0            # Multiplier on learning rate
    
    def is_active(self, current_stage: DevelopmentStage) -> bool:
        return current_stage == self.stage
```

### DevelopmentConfig

```python
@dataclass
class DevelopmentConfig:
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
```

---

## Method Specifications

### `__init__(config=None)`

```python
def __init__(self, config: Optional[DevelopmentConfig] = None):
    self.config = config or DevelopmentConfig()
    
    # Current state
    self.stage = DevelopmentStage.GENESIS
    self.age = 0.0
    self._stage_start_age = 0.0
    
    # Growth tracking
    self.current_oscillators = self.config.initial_oscillators
    self.experiences_processed = 0
    self.significant_experiences = 0
    
    # Milestones (stage transitions recorded here)
    self.milestones: List[Dict] = []
    
    # Genesis hash - the ONLY identity anchor at birth
    # This is immutable for the lifetime of the entity
    self.genesis_hash = hashlib.sha256(
        f"{time.time()}:{np.random.random()}:{id(self)}".encode()
    ).hexdigest()
```

### `stage_progress` property

```python
@property
def stage_progress(self) -> float:
    """Progress through current stage (0 to 1)."""
    age_in_stage = self.age - self._stage_start_age
    duration = self._get_stage_duration()
    
    if duration is None:  # MATURE stage
        return 1.0
    
    return min(1.0, age_in_stage / duration)
```

### `_get_stage_duration(stage=None) -> Optional[float]`

```python
def _get_stage_duration(self, stage: DevelopmentStage = None) -> Optional[float]:
    """Get duration of a stage. None for MATURE (indefinite)."""
    stage = stage or self.stage
    cfg = self.config
    
    durations = {
        DevelopmentStage.GENESIS: cfg.genesis_duration,
        DevelopmentStage.BABBLING: cfg.babbling_duration,
        DevelopmentStage.IMITATION: cfg.imitation_duration,
        DevelopmentStage.AUTONOMY: cfg.autonomy_duration,
        DevelopmentStage.MATURE: None,  # Indefinite
    }
    return durations[stage]
```

### `_next_stage() -> DevelopmentStage`

```python
def _next_stage(self) -> DevelopmentStage:
    """Get the next developmental stage."""
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
```

### `get_learning_multiplier(learning_type) -> float`

```python
def get_learning_multiplier(self, learning_type: str) -> float:
    """
    Get learning rate multiplier based on critical periods.
    
    Args:
        learning_type: Type of learning (e.g., "pattern", "social", "novelty")
    
    Returns:
        Multiplier >= 1.0 (1.0 if no active critical period)
    """
    multiplier = 1.0
    
    for period in self.config.critical_periods:
        if period.is_active(self.stage) and period.learning_type == learning_type:
            multiplier *= period.sensitivity
    
    return multiplier
```

### `should_grow() -> bool`

```python
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
```

### `process_experience(experience, significance) -> dict`

Main entry point for processing experiences.

```python
def process_experience(self, 
                       experience: Dict[str, Any],
                       significance: float = 0.5) -> Dict[str, Any]:
    """
    Process an experience, potentially triggering growth or stage transition.
    
    Args:
        experience: Dict containing at least 'type' key
        significance: 0-1, how significant is this experience
    
    Returns:
        Dict with:
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
        growth_amount = int(self.config.growth_rate * 10)
        self.current_oscillators = min(
            self.current_oscillators + growth_amount,
            self.config.max_oscillators
        )
        result['growth_triggered'] = True
        
        # Record milestone
        self.milestones.append({
            'type': 'growth',
            'oscillators': self.current_oscillators,
            'age': self.age,
            'experiences': self.experiences_processed,
        })
    
    # Check for stage transition
    duration = self._get_stage_duration()
    if duration is not None and (self.age - self._stage_start_age) >= duration:
        old_stage = self.stage
        self.stage = self._next_stage()
        self._stage_start_age = self.age
        result['stage_transition'] = self.stage
        
        # Record milestone
        self.milestones.append({
            'type': 'stage_transition',
            'from': old_stage.value,
            'to': self.stage.value,
            'age': self.age,
            'experiences': self.experiences_processed,
        })
    
    # Get learning multiplier for this experience type
    learning_type = experience.get('type', 'general')
    result['learning_multiplier'] = self.get_learning_multiplier(learning_type)
    
    return result
```

### `advance_age(dt)`

```python
def advance_age(self, dt: float):
    """Advance developmental age by dt."""
    self.age += dt
```

### `get_state() -> dict`

```python
def get_state(self) -> dict:
    return {
        'genesis_hash': self.genesis_hash,
        'stage': self.stage.value,
        'age': self.age,
        'stage_progress': self.stage_progress,
        'current_oscillators': self.current_oscillators,
        'experiences_processed': self.experiences_processed,
        'significant_experiences': self.significant_experiences,
        'milestones': self.milestones.copy(),
    }
```

---

## Success Criteria

### Correctness
1. Genesis hash is unique and immutable
2. Stage transitions occur at correct ages
3. Critical period multipliers apply correctly
4. Growth happens at correct intervals

### Behavior
1. New entity starts in GENESIS with minimal oscillators
2. Processing experiences advances toward growth/transition
3. Significance threshold works (only sig > 0.7 counts)
4. MATURE stage never transitions further

---

## Test Cases

```python
def test_genesis_hash_unique():
    """Each entity should have unique genesis hash."""
    t1 = DevelopmentTracker()
    t2 = DevelopmentTracker()
    
    assert t1.genesis_hash != t2.genesis_hash

def test_initial_state():
    """Should start in GENESIS with minimal oscillators."""
    t = DevelopmentTracker()
    
    assert t.stage == DevelopmentStage.GENESIS
    assert t.current_oscillators == t.config.initial_oscillators
    assert t.age == 0.0

def test_stage_transition():
    """Should transition stages at correct ages."""
    config = DevelopmentConfig(genesis_duration=10.0)
    t = DevelopmentTracker(config)
    
    # Advance past genesis duration
    t.advance_age(15.0)
    result = t.process_experience({'type': 'test'}, 0.5)
    
    assert result['stage_transition'] == DevelopmentStage.BABBLING
    assert t.stage == DevelopmentStage.BABBLING

def test_growth_trigger():
    """Should grow after N significant experiences."""
    config = DevelopmentConfig(growth_interval=5, initial_oscillators=10)
    t = DevelopmentTracker(config)
    
    # Process 4 significant experiences (not enough)
    for i in range(4):
        result = t.process_experience({'type': 'test'}, 0.8)
        assert not result['growth_triggered']
    
    # 5th should trigger growth
    result = t.process_experience({'type': 'test'}, 0.8)
    assert result['growth_triggered']
    assert t.current_oscillators > 10

def test_significance_threshold():
    """Only experiences with significance > 0.7 should count."""
    config = DevelopmentConfig(growth_interval=3)
    t = DevelopmentTracker(config)
    
    # Low significance - shouldn't count
    t.process_experience({'type': 'test'}, 0.5)
    t.process_experience({'type': 'test'}, 0.6)
    t.process_experience({'type': 'test'}, 0.7)  # Exactly 0.7 = not > 0.7
    
    assert t.significant_experiences == 0
    
    # High significance
    t.process_experience({'type': 'test'}, 0.8)
    assert t.significant_experiences == 1

def test_critical_period_multiplier():
    """Critical periods should enhance learning."""
    t = DevelopmentTracker()
    t.stage = DevelopmentStage.GENESIS
    
    # "pattern" learning has critical period in GENESIS
    mult = t.get_learning_multiplier("pattern")
    assert mult > 1.0
    
    # "social" learning doesn't have critical period in GENESIS
    mult = t.get_learning_multiplier("social")
    assert mult == 1.0
    
    # Move to IMITATION stage
    t.stage = DevelopmentStage.IMITATION
    mult = t.get_learning_multiplier("social")
    assert mult > 1.0

def test_mature_no_transition():
    """MATURE stage should never transition."""
    t = DevelopmentTracker()
    t.stage = DevelopmentStage.MATURE
    t._stage_start_age = 0.0
    
    t.advance_age(100000.0)  # Huge age
    result = t.process_experience({'type': 'test'}, 0.5)
    
    assert result['stage_transition'] is None
    assert t.stage == DevelopmentStage.MATURE

def test_milestones_recorded():
    """Stage transitions and growth should be recorded."""
    config = DevelopmentConfig(genesis_duration=10.0, growth_interval=2)
    t = DevelopmentTracker(config)
    
    # Trigger growth
    t.process_experience({'type': 'test'}, 0.8)
    t.process_experience({'type': 'test'}, 0.8)
    
    # Trigger transition
    t.advance_age(15.0)
    t.process_experience({'type': 'test'}, 0.5)
    
    assert len(t.milestones) >= 2
    types = [m['type'] for m in t.milestones]
    assert 'growth' in types
    assert 'stage_transition' in types
```

---

## Dependencies

- `numpy` (for random in genesis hash)
- `hashlib` (for genesis hash)
- `time` (for genesis hash)
- `dataclasses`
- `enum`

No other ORE2 components.

---

## File Location

```
ore2/
├── core/
│   ├── __init__.py
│   ├── sparse_oscillator.py
│   ├── multi_scale_substrate.py
│   ├── embodiment.py
│   ├── memory.py
│   └── development.py  # <-- This component
├── tests/
│   └── test_development.py
```

---

## Design Decisions to Preserve

1. **Genesis hash is the ONLY birth anchor** - no founding memories
2. **Five stages in order** - GENESIS → BABBLING → IMITATION → AUTONOMY → MATURE
3. **MATURE is indefinite** - no "death" stage
4. **Significance threshold 0.7** - only meaningful experiences count for growth
5. **Critical periods are stage-locked** - sensitivity only during specific stage
6. **Milestones are append-only** - history is permanent

---

## Note on Integration

The `process_experience()` return value includes `learning_multiplier`. The calling code (DevelopmentalEntity) should use this to scale how strongly experiences affect the substrate:

```python
# In DevelopmentalEntity.process_experience():
dev_result = self.development.process_experience(experience, significance)
stim_strength = base_strength * dev_result['learning_multiplier']
self.substrate.stimulate_concept(..., strength=stim_strength)
```

This makes critical periods actually matter for learning.
