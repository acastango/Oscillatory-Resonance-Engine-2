# BRIEFING: GoalSystem

## Component ID: ORE2-017
## Priority: High (Enables intentional behavior)
## Estimated complexity: Medium

---

## What This Is

A hierarchical goal system that provides **drives**, **objectives**, and **intentions**. Goals create attractors in the substrate, bias action selection, and give the entity something to *want*.

Without goals, the entity reacts. With goals, it *acts* - pursuing states it values.

---

## Why It Matters

**H2 (Philosophy of Mind):** "Intentionality is directedness toward something. A system without goals has no 'toward'. Goals create the aboutness that makes cognition meaningful."

**P1 (Dynamical Systems):** "Goals are attractors. They shape the energy landscape. The system flows toward goal states because they're dynamically preferred."

**A5 (Continual Learning):** "Goals provide the value signal. Without them, what should the system learn to do? Goals define 'better' and 'worse' states."

---

## The Core Insight

Hierarchical goal structure:

```
                    ┌───────────────────┐
                    │     DRIVES        │
                    │  (intrinsic needs)│
                    │  coherence, energy│
                    │  novelty, social  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    OBJECTIVES     │
                    │ (medium-term aims)│
                    │ complete task,    │
                    │ learn pattern     │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    INTENTIONS     │
                    │  (immediate acts) │
                    │  focus, respond,  │
                    │  retrieve memory  │
                    └───────────────────┘
```

- **Drives**: Intrinsic motivations, always active, create baseline "need" states
- **Objectives**: Acquired/assigned goals, have completion conditions
- **Intentions**: Moment-to-moment action selections

Goals propagate DOWN: drives create objectives, objectives create intentions.
Satisfaction propagates UP: completing intentions satisfies objectives, satisfying objectives reduces drives.

---

## Interface Contract

```python
class GoalSystem:
    """
    Hierarchical goal management.
    
    Properties:
        drives: Dict[str, Drive]           # Intrinsic motivations
        objectives: List[Objective]        # Current objectives
        intentions: List[Intention]        # Immediate action queue
        dominant_drive: str                # Currently strongest drive
        goal_coherence: float              # How aligned are goals
    
    Methods:
        # Drive management
        update_drives(body_state, substrate_state)
        get_drive_vector() -> np.ndarray
        
        # Objective management
        add_objective(objective)
        complete_objective(objective_id)
        get_active_objectives() -> List[Objective]
        
        # Intention management
        form_intention(substrate_state) -> Intention
        execute_intention(intention) -> IntentionResult
        
        # Integration
        step(entity_state) -> GoalState
        get_goal_patterns() -> List[np.ndarray]  # For attention/inference
"""

@dataclass
class Drive:
    """Intrinsic motivation that's always active."""
    name: str
    current_level: float       # 0-1, how unsatisfied
    target_level: float        # Desired level (usually low)
    decay_rate: float          # How fast need builds
    satisfaction_rate: float   # How fast it's satisfied
    substrate_pattern: Optional[np.ndarray] = None  # Phase pattern when satisfied

@dataclass
class Objective:
    """Medium-term goal with completion condition."""
    id: str
    description: str
    priority: float            # 0-1
    progress: float            # 0-1
    deadline: Optional[float]  # Tick count, or None
    completion_condition: Callable[['EntityState'], bool]
    on_complete: Optional[Callable] = None
    substrate_pattern: Optional[np.ndarray] = None
    created_at: float = 0
    parent_drive: Optional[str] = None

@dataclass
class Intention:
    """Immediate action to take."""
    action: str
    target: Optional[str]      # What to act on
    expected_outcome: str
    priority: float
    substrate_pattern: Optional[np.ndarray] = None

@dataclass
class GoalState:
    """Current goal system state."""
    dominant_drive: str
    drive_levels: Dict[str, float]
    active_objectives: int
    current_intention: Optional[Intention]
    goal_coherence: float
```

---

## Configuration

```python
@dataclass
class GoalConfig:
    # Drives
    default_drives: List[str] = field(default_factory=lambda: [
        'coherence',   # Desire for internal harmony
        'energy',      # Desire for adequate resources
        'novelty',     # Desire for new experiences
        'competence',  # Desire to succeed at tasks
        'social',      # Desire for connection (if multi-agent)
    ])
    
    drive_decay_rate: float = 0.01      # How fast needs build
    drive_satisfaction_rate: float = 0.1
    
    # Objectives
    max_objectives: int = 10
    objective_timeout: float = 10000    # Ticks before auto-fail
    
    # Intentions
    intention_duration: int = 10        # Ticks per intention
    
    # Integration
    goal_influence_on_substrate: float = 0.2
```

---

## Method Specifications

### `__init__(substrate, config)`

```python
def __init__(self,
             substrate: 'MultiScaleSubstrate',
             config: Optional[GoalConfig] = None):
    self.config = config or GoalConfig()
    self.substrate = substrate
    
    # Initialize drives
    self.drives: Dict[str, Drive] = {}
    for drive_name in self.config.default_drives:
        self.drives[drive_name] = Drive(
            name=drive_name,
            current_level=0.5,  # Start moderately unsatisfied
            target_level=0.2,   # Want to keep low
            decay_rate=self.config.drive_decay_rate,
            satisfaction_rate=self.config.drive_satisfaction_rate,
            substrate_pattern=self._generate_drive_pattern(drive_name),
        )
    
    # Objectives and intentions
    self.objectives: List[Objective] = []
    self.intentions: List[Intention] = []
    self._current_intention: Optional[Intention] = None
    self._intention_ticks: int = 0
    
    # Counters
    self._tick_count = 0

def _generate_drive_pattern(self, drive_name: str) -> np.ndarray:
    """Generate characteristic phase pattern for a drive."""
    n = self.substrate.slow.n
    
    # Use drive name as seed for reproducibility
    seed = hash(drive_name) % (2**32)
    np.random.seed(seed)
    
    pattern = np.random.uniform(0, 2*np.pi, n)
    
    # Reset random state
    np.random.seed(None)
    
    return pattern
```

### `update_drives(body_state, substrate_state)`

```python
def update_drives(self,
                  body_state: dict,
                  substrate_state: dict):
    """
    Update drive levels based on current state.
    
    Drives decay (increase need) over time.
    Drives are satisfied by relevant state conditions.
    """
    cfg = self.config
    
    for name, drive in self.drives.items():
        # Decay (need increases over time)
        drive.current_level += drive.decay_rate
        
        # Satisfaction based on relevant state
        satisfaction = self._compute_drive_satisfaction(
            name, body_state, substrate_state
        )
        drive.current_level -= satisfaction * drive.satisfaction_rate
        
        # Clamp
        drive.current_level = np.clip(drive.current_level, 0, 1)

def _compute_drive_satisfaction(self,
                                 drive_name: str,
                                 body_state: dict,
                                 substrate_state: dict) -> float:
    """Compute how much current state satisfies a drive."""
    
    if drive_name == 'coherence':
        # Satisfied by high substrate coherence
        coherence = substrate_state.get('coherence', 0)
        return coherence
    
    elif drive_name == 'energy':
        # Satisfied by adequate body energy
        energy = body_state.get('energy', 0.5)
        return energy
    
    elif drive_name == 'novelty':
        # Satisfied by prediction error / surprise
        surprise = substrate_state.get('surprise', 0)
        return min(surprise, 1.0)
    
    elif drive_name == 'competence':
        # Satisfied by completing objectives
        # (Updated separately in complete_objective)
        return 0.0
    
    elif drive_name == 'social':
        # Satisfied by social coupling
        social_coherence = substrate_state.get('social_coherence', 0)
        return social_coherence
    
    return 0.0
```

### `get_drive_vector() -> np.ndarray`

```python
def get_drive_vector(self) -> np.ndarray:
    """
    Get drive levels as vector.
    
    Used for action selection - higher drives push for action.
    """
    return np.array([d.current_level for d in self.drives.values()])

@property
def dominant_drive(self) -> str:
    """Drive with highest current level (most unsatisfied)."""
    return max(self.drives.items(), key=lambda x: x[1].current_level)[0]
```

### `add_objective(objective)`

```python
def add_objective(self, objective: Objective):
    """Add a new objective."""
    cfg = self.config
    
    # Check limit
    if len(self.objectives) >= cfg.max_objectives:
        # Remove lowest priority completed or timed-out
        self._prune_objectives()
    
    # Generate substrate pattern if not provided
    if objective.substrate_pattern is None:
        objective.substrate_pattern = self._generate_objective_pattern(objective)
    
    objective.created_at = self._tick_count
    self.objectives.append(objective)

def _generate_objective_pattern(self, objective: Objective) -> np.ndarray:
    """Generate phase pattern for objective."""
    n = self.substrate.slow.n
    
    # Start from parent drive pattern if exists
    if objective.parent_drive and objective.parent_drive in self.drives:
        base = self.drives[objective.parent_drive].substrate_pattern.copy()
        # Perturb slightly to differentiate
        base += 0.2 * np.random.randn(n)
        return base % (2 * np.pi)
    
    # Otherwise random
    return np.random.uniform(0, 2*np.pi, n)

def complete_objective(self, objective_id: str):
    """Mark objective as complete."""
    for obj in self.objectives:
        if obj.id == objective_id:
            obj.progress = 1.0
            
            # Satisfy competence drive
            self.drives['competence'].current_level -= 0.2
            
            # Call completion callback
            if obj.on_complete:
                obj.on_complete()
            
            # Remove from active
            self.objectives.remove(obj)
            break

def get_active_objectives(self) -> List[Objective]:
    """Get objectives that aren't complete or timed out."""
    active = []
    for obj in self.objectives:
        if obj.progress < 1.0:
            if obj.deadline is None or self._tick_count < obj.deadline:
                active.append(obj)
    return active
```

### `form_intention(entity_state) -> Intention`

```python
def form_intention(self, entity_state: dict) -> Intention:
    """
    Form an intention based on current state and goals.
    
    Considers:
    - Dominant drive
    - Active objectives
    - Current state
    """
    # Get dominant drive
    dominant = self.dominant_drive
    drive = self.drives[dominant]
    
    # Get highest priority active objective
    active_objs = self.get_active_objectives()
    if active_objs:
        top_obj = max(active_objs, key=lambda o: o.priority)
    else:
        top_obj = None
    
    # Form intention based on drive and objective
    action, target, expected = self._select_action(
        dominant, drive, top_obj, entity_state
    )
    
    return Intention(
        action=action,
        target=target,
        expected_outcome=expected,
        priority=drive.current_level,
        substrate_pattern=drive.substrate_pattern,
    )

def _select_action(self,
                   drive_name: str,
                   drive: Drive,
                   objective: Optional[Objective],
                   state: dict) -> Tuple[str, Optional[str], str]:
    """Select action based on drive and objective."""
    
    # Drive-specific default actions
    drive_actions = {
        'coherence': ('focus', None, 'increase coherence'),
        'energy': ('rest', None, 'restore energy'),
        'novelty': ('explore', None, 'encounter novelty'),
        'competence': ('engage', None, 'work on task'),
        'social': ('reach_out', None, 'social connection'),
    }
    
    # If objective exists, try to advance it
    if objective:
        return ('work_objective', objective.id, f'advance {objective.description}')
    
    # Fall back to drive action
    return drive_actions.get(drive_name, ('wait', None, 'maintain state'))
```

### `step(entity_state) -> GoalState`

Main goal system step.

```python
def step(self, entity_state: dict) -> GoalState:
    """
    One step of goal processing.
    """
    self._tick_count += 1
    
    # Update drives
    body_state = entity_state.get('body', {})
    substrate_state = entity_state.get('substrate', {})
    self.update_drives(body_state, substrate_state)
    
    # Check objective completion
    for obj in self.objectives[:]:
        if obj.completion_condition(entity_state):
            self.complete_objective(obj.id)
    
    # Prune timed-out objectives
    self._prune_objectives()
    
    # Form or continue intention
    if self._current_intention is None or \
       self._intention_ticks >= self.config.intention_duration:
        self._current_intention = self.form_intention(entity_state)
        self._intention_ticks = 0
    
    self._intention_ticks += 1
    
    # Apply goal influence to substrate
    self._apply_goal_influence()
    
    # Compute goal coherence
    goal_coherence = self._compute_goal_coherence()
    
    return GoalState(
        dominant_drive=self.dominant_drive,
        drive_levels={n: d.current_level for n, d in self.drives.items()},
        active_objectives=len(self.get_active_objectives()),
        current_intention=self._current_intention,
        goal_coherence=goal_coherence,
    )

def _apply_goal_influence(self):
    """Apply goal patterns to substrate as bias."""
    cfg = self.config
    
    # Dominant drive creates attractor
    dominant = self.drives[self.dominant_drive]
    if dominant.substrate_pattern is not None:
        # Stimulate oscillators toward drive pattern
        phase_diff = dominant.substrate_pattern - self.substrate.slow.phases
        alignment = np.cos(phase_diff)
        
        # Positive alignment = in-phase with goal = stimulate
        aligned_mask = alignment > 0.5
        if np.any(aligned_mask):
            indices = np.where(aligned_mask)[0]
            strengths = cfg.goal_influence_on_substrate * alignment[aligned_mask]
            self.substrate.slow.stimulate(indices, strengths)

def _compute_goal_coherence(self) -> float:
    """Compute how coherent/aligned current goals are."""
    patterns = self.get_goal_patterns()
    
    if len(patterns) < 2:
        return 1.0
    
    # Pairwise coherence between goal patterns
    coherences = []
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            diff = patterns[i] - patterns[j]
            coh = np.abs(np.mean(np.exp(1j * diff)))
            coherences.append(coh)
    
    return np.mean(coherences)
```

### `get_goal_patterns() -> List[np.ndarray]`

```python
def get_goal_patterns(self) -> List[np.ndarray]:
    """
    Get all active goal patterns.
    
    Used by attention and inference to know what's relevant.
    """
    patterns = []
    
    # Dominant drive pattern
    dominant = self.drives[self.dominant_drive]
    if dominant.substrate_pattern is not None:
        patterns.append(dominant.substrate_pattern)
    
    # Active objective patterns
    for obj in self.get_active_objectives():
        if obj.substrate_pattern is not None:
            patterns.append(obj.substrate_pattern)
    
    return patterns
```

---

## Integration with Entity

```python
# In DevelopmentalEntity.__init__:
self.goals = GoalSystem(self.substrate)

# In DevelopmentalEntity.tick():
def tick(self):
    # ... existing code ...
    
    # Goal system step
    goal_state = self.goals.step({
        'body': {
            'energy': self.body.energy,
            'valence': self.body.valence,
        },
        'substrate': {
            'coherence': self.substrate.global_coherence,
            'surprise': self.inference.surprise if hasattr(self, 'inference') else 0,
        },
    })
    
    # Pass goal patterns to attention and inference
    if hasattr(self, 'attention'):
        self.attention.step(
            self.substrate,
            goals=self.goals.get_goal_patterns(),
        )
    
    return {**result, 'goals': goal_state}
```

---

## Success Criteria

### Drives
1. Drives decay (increase) over time
2. Relevant state satisfies drives
3. Dominant drive changes based on levels

### Objectives
1. Objectives can be added and tracked
2. Completion conditions work
3. Timeout/pruning works

### Integration
1. Goal patterns influence substrate
2. Goal coherence measurable
3. Intentions form based on drives/objectives

---

## Test Cases

```python
def test_drive_decay():
    """Drives should increase over time."""
    substrate = create_test_substrate()
    goals = GoalSystem(substrate)
    
    initial = goals.drives['coherence'].current_level
    
    for _ in range(100):
        goals.step({'body': {}, 'substrate': {}})
    
    assert goals.drives['coherence'].current_level > initial

def test_drive_satisfaction():
    """Coherence drive should be satisfied by high coherence."""
    substrate = create_test_substrate()
    goals = GoalSystem(substrate)
    
    # High drive
    goals.drives['coherence'].current_level = 0.9
    
    # High coherence state
    goals.step({
        'body': {},
        'substrate': {'coherence': 0.9}
    })
    
    assert goals.drives['coherence'].current_level < 0.9

def test_objective_completion():
    """Completed objectives should be removed."""
    substrate = create_test_substrate()
    goals = GoalSystem(substrate)
    
    obj = Objective(
        id='test_obj',
        description='Test objective',
        priority=0.8,
        progress=0.0,
        deadline=None,
        completion_condition=lambda s: s.get('test_done', False),
    )
    goals.add_objective(obj)
    
    assert len(goals.get_active_objectives()) == 1
    
    goals.step({'test_done': True, 'body': {}, 'substrate': {}})
    
    assert len(goals.get_active_objectives()) == 0

def test_intention_formation():
    """Should form intentions based on dominant drive."""
    substrate = create_test_substrate()
    goals = GoalSystem(substrate)
    
    # Make coherence dominant
    goals.drives['coherence'].current_level = 0.9
    goals.drives['energy'].current_level = 0.1
    
    state = goals.step({'body': {}, 'substrate': {}})
    
    assert state.current_intention is not None
    assert state.dominant_drive == 'coherence'
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
│   └── goals.py  # <-- This component
├── tests/
│   └── test_goals.py
```

---

## Design Decisions to Preserve

1. **Hierarchical structure** - Drives → Objectives → Intentions
2. **Drives are intrinsic** - Always active, not learned
3. **Objectives have deadlines** - Prevents infinite pursuit
4. **Goal patterns as substrate attractors** - Goals shape dynamics
5. **Satisfaction propagates up** - Intentions → Objectives → Drives
6. **Goal coherence measurable** - Conflicting goals reduce coherence
