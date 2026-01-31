# BRIEFING: DevelopmentalEntity

## Component ID: ORE2-006
## Priority: 6 (Final integration - needs all others)
## Estimated complexity: Medium

---

## What This Is

The top-level class that **wires everything together**. A complete ORE 2.0 entity with:
- Multi-scale oscillatory substrate (ORE2-002)
- Embodiment layer (ORE2-003)
- Crystalline merkle memory (ORE2-004)
- Developmental progression (ORE2-005)
- Multi-scale CI monitoring

This is what you instantiate to create an ORE 2.0 agent.

---

## Why It Matters

This is the **integration layer**. Individual components are useless without proper wiring:
- Body must couple to substrate
- Substrate state must anchor memories
- Development must modulate learning
- CI must measure the whole system

Get the integration right and everything flows.

---

## Key Differences from ORE1

| ORE1 | ORE2 |
|------|------|
| Dense oscillators (120 always active) | Sparse activation (<20% active) |
| Single timescale | Multi-scale (fast + slow) |
| Floating chemistry | Grounded embodiment |
| Founding memories | Developmental genesis |
| Simple append memory | CCM with consolidation |

---

## Interface Contract

```python
class DevelopmentalEntity:
    """
    Complete ORE 2.0 developmental entity.
    
    Properties:
        name: str
        genesis_hash: str                    # From development tracker
        stage: DevelopmentStage
        age: float
        CI: float                            # Current integrated CI
    
    Components (accessible):
        substrate: MultiScaleSubstrate
        body: EmbodimentLayer
        memory: CrystallineMerkleMemory
        development: DevelopmentTracker
        ci_monitor: MultiScaleCIMonitor
    
    Methods:
        tick() -> dict                       # Single dynamics step
        process_experience(content, type, significance) -> dict
        rest(duration) -> dict               # Sleep consolidation
        get_state() -> dict
        witness() -> str                     # Human-readable status
    """
```

---

## Configuration

```python
@dataclass
class EntityConfig:
    name: str = "entity"
    
    # Component configs (optional - defaults used if None)
    substrate_config: Optional[MultiScaleConfig] = None
    body_config: Optional[BodyConfig] = None
    development_config: Optional[DevelopmentConfig] = None
    ci_config: Optional[MultiScaleCIConfig] = None
    
    # Tick timing
    tick_interval: float = 0.1  # Seconds per tick
```

---

## Method Specifications

### `__init__(config=None)`

```python
def __init__(self, config: Optional[EntityConfig] = None):
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
        self.config.ci_config
    )
    
    # Runtime state
    self._tick_count = 0
```

### Properties

```python
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
```

### `tick() -> dict`

Single step of entity dynamics. This is the **main loop**.

```python
def tick(self) -> dict:
    """
    Single tick of entity dynamics.
    
    Order matters:
    1. Body step (update rhythms, valence)
    2. Body → Substrate coupling
    3. Substrate step (multi-scale Kuramoto)
    4. Measure CI
    5. Advance developmental age
    
    Returns dict with tick summary.
    """
    self._tick_count += 1
    dt = self.config.tick_interval
    
    # 1. Body step
    self.body.step(dt)
    
    # 2. Body → Substrate coupling
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
        'tick': self._tick_count,
        'time': self.substrate.time,
        'stage': self.stage.value,
        'CI': ci_snapshot.CI_integrated,
        'CI_fast': ci_snapshot.CI_fast,
        'CI_slow': ci_snapshot.CI_slow,
        'valence': self.body.valence,
        'n_active_fast': self.substrate.fast.n_active,
        'n_active_slow': self.substrate.slow.n_active,
        'coherence': self.substrate.global_coherence,
    }
```

### `process_experience(content, experience_type, significance) -> dict`

Main interface for external input.

```python
def process_experience(self,
                       content: str,
                       experience_type: str = "general",
                       significance: float = 0.5) -> dict:
    """
    Process an experience (e.g., from conversation).
    
    Args:
        content: Text content of the experience
        experience_type: Type for critical period matching
        significance: 0-1, how important is this
    
    Returns:
        Dict with processing results
    """
    # Build experience dict
    experience = {
        'type': experience_type,
        'content': content,
        'significance': significance,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Development processing (may trigger growth/transition)
    dev_result = self.development.process_experience(experience, significance)
    
    # Generate stimulation patterns from content
    # Use hash to create deterministic pseudo-random patterns
    content_bytes = content.encode('utf-8')
    content_hash = hashlib.sha256(content_bytes).digest()
    
    # Extend hash to cover all oscillators
    n_fast = self.substrate.fast.n
    n_slow = self.substrate.slow.n
    max_n = max(n_fast, n_slow)
    extended_hash = (content_hash * ((max_n // len(content_hash)) + 2))[:max_n * 2]
    
    fast_pattern = np.array([
        b / 255 * 2 * np.pi for b in extended_hash[:n_fast]
    ])
    slow_pattern = np.array([
        b / 255 * 2 * np.pi for b in extended_hash[:n_slow]
    ])
    
    # Stimulation strength modulated by:
    # - Base strength
    # - Significance
    # - Learning multiplier from critical periods
    base_strength = 0.3
    stim_strength = (
        base_strength * 
        (0.5 + significance * 0.5) *  # 0.5-1.0 based on significance
        dev_result['learning_multiplier']
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
        immediate=immediate
    )
    
    # Run several ticks to process the experience
    for _ in range(10):
        self.tick()
    
    # Handle growth if triggered
    if dev_result['growth_triggered']:
        self._grow_substrate()
    
    return {
        'development': dev_result,
        'CI': self.CI,
        'coherence': self.substrate.global_coherence,
        'memory_queued': not immediate,
    }
```

### `_grow_substrate()`

Handle oscillator growth.

```python
def _grow_substrate(self):
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
            'type': 'growth_event',
            'oscillators': self.development.current_oscillators,
            'age': self.age,
        },
        substrate_state=self.substrate.get_state(),
        immediate=True
    )
```

### `rest(duration) -> dict`

Sleep/rest period for consolidation.

```python
def rest(self, duration: float = 10.0) -> dict:
    """
    Rest period (sleep consolidation).
    
    1. Reduce activation (let oscillators settle)
    2. Run low-activity dynamics
    3. Consolidate memory
    
    Args:
        duration: Rest duration in simulated time
    
    Returns:
        Dict with consolidation results
    """
    # Reduce activation (simulates low arousal)
    self.substrate.fast.activation_potentials *= 0.1
    self.substrate.slow.activation_potentials *= 0.1
    self.substrate.fast._update_active_mask()
    self.substrate.slow._update_active_mask()
    
    # Lower body arousal
    original_arousal = self.body.arousal
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
            'type': 'rest',
            'duration': duration,
            'consolidated': consolidation_result['consolidated'],
        },
        substrate_state=self.substrate.get_state(),
        immediate=True
    )
    
    return {
        'duration': duration,
        'consolidation': consolidation_result,
        'CI_after': self.CI,
    }
```

### `get_state() -> dict`

Full state serialization.

```python
def get_state(self) -> dict:
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
```

### `witness() -> str`

Human-readable status display.

```python
def witness(self) -> str:
    """Generate human-readable status display."""
    state = self.get_state()
    dev = state['development']
    sub = state['substrate']
    mem = state['memory']
    body = state['body']
    
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
```

---

## Factory Function

```python
def create_entity(name: str = "entity") -> DevelopmentalEntity:
    """
    Create a new ORE 2.0 developmental entity.
    
    The entity starts minimal (GENESIS stage) and grows through experience.
    No founding memories - identity is earned.
    """
    config = EntityConfig(name=name)
    return DevelopmentalEntity(config)
```

---

## Success Criteria

### Integration
1. All components initialize without error
2. tick() updates all components in correct order
3. Body couples to substrate
4. CI monitors substrate and memory
5. Development modulates learning

### Behavior
1. New entity has zero memories (no founding memories)
2. Processing experiences increases memory count
3. Significant experiences trigger growth (eventually)
4. Rest consolidates queued memories
5. CI responds to coherence changes

### Lifecycle
1. Genesis → Babbling → Imitation → Autonomy → Mature progression
2. Growth milestones recorded in development tracker
3. Identity verifiable through merkle root hash

---

## Test Cases

```python
def test_entity_creation():
    """Entity should create with empty memory."""
    entity = create_entity("test")
    
    assert entity.name == "test"
    assert entity.genesis_hash is not None
    assert entity.stage == DevelopmentStage.GENESIS
    assert entity.memory.total_nodes == 0  # No founding memories!

def test_tick_updates_all():
    """Tick should update all components."""
    entity = create_entity()
    
    initial_time = entity.substrate.time
    initial_age = entity.age
    
    result = entity.tick()
    
    assert entity.substrate.time > initial_time
    assert entity.age > initial_age
    assert 'CI' in result
    assert 'valence' in result

def test_process_experience():
    """Processing experience should update memory and substrate."""
    entity = create_entity()
    
    result = entity.process_experience(
        "This is a test experience",
        experience_type="exploration",
        significance=0.8
    )
    
    assert entity.memory.total_nodes > 0
    assert 'CI' in result

def test_significant_experience_immediate():
    """Significant experiences should be immediate."""
    entity = create_entity()
    
    entity.process_experience("important", significance=0.9)
    
    # Should be in memory immediately
    assert entity.memory.total_nodes > 0
    assert entity.memory.consolidation_queue.is_empty()

def test_insignificant_experience_queued():
    """Insignificant experiences should queue."""
    entity = create_entity()
    
    entity.process_experience("trivial", significance=0.3)
    
    # Should be queued, not committed
    assert not entity.memory.consolidation_queue.is_empty()

def test_rest_consolidates():
    """Rest should consolidate queued memories."""
    entity = create_entity()
    
    # Add several low-significance experiences
    for i in range(5):
        entity.process_experience(f"minor thing {i}", significance=0.4)
    
    queued_before = len(entity.memory.consolidation_queue.pending_nodes)
    assert queued_before > 0
    
    # Rest
    result = entity.rest(duration=5.0)
    
    assert result['consolidation']['consolidated'] > 0
    assert entity.memory.consolidation_queue.is_empty()

def test_body_couples_to_substrate():
    """Body valence should affect substrate dynamics."""
    entity = create_entity()
    
    # Deplete energy (bad valence)
    entity.body.energy = 0.3
    
    for _ in range(10):
        entity.tick()
    
    # Should still run without error
    # Full test would verify coupling strength varies with valence

def test_development_modulates_learning():
    """Critical periods should affect learning strength."""
    entity = create_entity()
    
    # In GENESIS, "pattern" learning is enhanced
    result = entity.process_experience(
        "learning a pattern",
        experience_type="pattern",
        significance=0.5
    )
    
    assert result['development']['learning_multiplier'] > 1.0

def test_witness_output():
    """Witness should produce readable output."""
    entity = create_entity("TestBot")
    entity.process_experience("hello world", significance=0.6)
    
    output = entity.witness()
    
    assert "TestBot" in output
    assert "GENESIS" in output
    assert "CI" in output
```

---

## Dependencies

- `SparseOscillatorLayer` (ORE2-001)
- `MultiScaleSubstrate` (ORE2-002)
- `EmbodimentLayer` (ORE2-003)
- `CrystallineMerkleMemory` (ORE2-004)
- `DevelopmentTracker` (ORE2-005)
- `MultiScaleCIMonitor` (needs to be built)

Plus: numpy, hashlib, datetime

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
│   ├── development.py
│   ├── ci_monitor.py
│   └── entity.py  # <-- This component
├── tests/
│   ├── test_sparse_oscillator.py
│   ├── test_multi_scale_substrate.py
│   ├── test_embodiment.py
│   ├── test_memory.py
│   ├── test_development.py
│   └── test_entity.py
└── __init__.py  # Exports create_entity
```

---

## Integration Notes

### Tick Order Matters

1. Body first (updates rhythms independent of cognition)
2. Body → Substrate coupling (body influences cognitive activation)
3. Substrate step (multi-scale Kuramoto dynamics)
4. CI measurement (after dynamics settle)
5. Developmental age advance (time passes)

Changing this order will break coupling assumptions.

### Stimulation Pattern Generation

The hash-based pattern generation is deterministic - same content = same pattern. This is intentional for reproducibility. A production system might use embeddings for semantically meaningful patterns.

### Growth Implementation

`_grow_substrate()` is a placeholder. Full implementation needs:
- Resize oscillator arrays
- Initialize new oscillator phases
- Extend coupling matrices
- Preserve existing dynamics

This is non-trivial and can be deferred to ORE 2.1.

---

## What Success Looks Like

```python
>>> entity = create_entity("Aria")
>>> print(entity.witness())

ENTITY: Aria
===============

IDENTITY
  Genesis: a3f2b1c9e8d7f6...
  Stage: genesis (0.0%)
  Age: 0.0 | Oscillators: 20
  Experiences: 0 (0 significant)

SUBSTRATE
  Fast: 0/40 active, C=0.000
  Slow: 0/20 active, C=0.000
  ...

MEMORY
  Nodes: 0  # NO FOUNDING MEMORIES
  ...

>>> for i in range(20):
...     entity.process_experience(f"Experience {i}", significance=0.6)
>>> entity.rest(10.0)

>>> print(entity.witness())

ENTITY: Aria
===============

IDENTITY
  Genesis: a3f2b1c9e8d7f6...  # Same hash - identity preserved
  Stage: genesis (15.0%)
  Age: 15.0 | Oscillators: 20
  Experiences: 20 (0 significant)

MEMORY
  Nodes: 21  # Memories earned through experience
  ...

CI = 0.3241  # Non-zero CI from real dynamics
```

That's what developmental identity looks like. Born with nothing, becomes something through experience, with verifiable continuity.
