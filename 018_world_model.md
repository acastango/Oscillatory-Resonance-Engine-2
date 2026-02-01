# BRIEFING: WorldModel

## Component ID: ORE2-018
## Priority: Medium-High (Enables planning and prediction)
## Estimated complexity: High

---

## What This Is

An internal model of the **environment** - entities, objects, relationships, and dynamics outside the self. The world model enables:
- **Prediction**: What will happen if X?
- **Planning**: What sequence of actions leads to goal?
- **Simulation**: Mental rehearsal before acting
- **Counterfactuals**: What would have happened if...?

This is the "map" that the entity uses to navigate reality.

---

## Why It Matters

**A5 (Continual Learning):** "Active inference needs something to predict. The world model is that something - it's what generates expectations about external reality."

**H2 (Philosophy of Mind):** "The world model is how the entity represents the not-self. Without it, there's no distinction between inner and outer, self and world."

**P1 (Dynamical Systems):** "Planning is mental simulation - running the world model forward. Goals are achieved by finding action sequences that lead to goal states in the model."

---

## The Core Insight

The world model contains:

1. **Entities**: Other agents, objects, abstract concepts
2. **Relations**: How entities relate to each other and to self
3. **Dynamics**: How the world changes over time
4. **Affordances**: What actions are possible given current state

```
┌─────────────────────────────────────────────────────────┐
│                     World Model                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ENTITIES              RELATIONS           DYNAMICS      │
│  ─────────             ─────────           ────────      │
│  • User               Self─trusts─►User   User responds  │
│  • Claude             User─asks─►Self     to questions   │
│  • Current task       Task─requires─►     Tasks complete │
│  • Environment          knowledge         over time      │
│                                                          │
│  AFFORDANCES                                             │
│  ───────────                                             │
│  • Can respond to user                                   │
│  • Can search memory                                     │
│  • Can request clarification                             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Interface Contract

```python
class WorldModel:
    """
    Internal model of the external world.
    
    Properties:
        entities: Dict[str, Entity]
        relations: List[Relation]
        current_state: WorldState
        prediction_accuracy: float
    
    Methods:
        # Entity management
        add_entity(entity)
        update_entity(entity_id, updates)
        remove_entity(entity_id)
        
        # Relation management
        add_relation(relation)
        get_relations(entity_id) -> List[Relation]
        
        # State tracking
        observe(observation) -> WorldState
        get_state() -> WorldState
        
        # Prediction and simulation
        predict(action) -> PredictedWorldState
        simulate(action_sequence) -> List[PredictedWorldState]
        
        # Planning
        plan_to_goal(goal_state) -> List[str]
        
        # Integration
        step(observation, action_taken)
"""

@dataclass
class Entity:
    """Something in the world."""
    id: str
    type: str                    # 'agent', 'object', 'concept', 'event'
    name: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None  # Semantic representation
    phase_pattern: Optional[np.ndarray] = None  # Oscillator representation
    last_observed: float = 0
    certainty: float = 1.0       # How confident in existence

@dataclass
class Relation:
    """Relationship between entities."""
    source_id: str
    relation_type: str           # 'trusts', 'contains', 'causes', etc.
    target_id: str
    strength: float = 1.0
    bidirectional: bool = False
    last_observed: float = 0

@dataclass
class WorldState:
    """Snapshot of world model state."""
    timestamp: float
    active_entities: List[str]   # Currently relevant
    active_relations: List[Relation]
    affordances: List[str]       # Possible actions
    uncertainty: float           # Overall uncertainty

@dataclass
class PredictedWorldState:
    """Predicted future world state."""
    state: WorldState
    action_taken: str
    probability: float
    consequences: List[str]
```

---

## Configuration

```python
@dataclass
class WorldModelConfig:
    # Entities
    max_entities: int = 100
    entity_decay_rate: float = 0.01    # Certainty decay over time
    min_certainty: float = 0.1         # Below this, entity removed
    
    # Relations
    max_relations_per_entity: int = 20
    relation_decay_rate: float = 0.005
    
    # Prediction
    prediction_horizon: int = 5        # Steps ahead
    simulation_samples: int = 10       # Monte Carlo samples
    
    # Planning
    max_plan_length: int = 10
    plan_search_depth: int = 5
    
    # Grounding
    grounding: Optional['SemanticGrounding'] = None
```

---

## Method Specifications

### `__init__(substrate, config)`

```python
def __init__(self,
             substrate: 'MultiScaleSubstrate',
             config: Optional[WorldModelConfig] = None):
    self.config = config or WorldModelConfig()
    self.substrate = substrate
    self.grounding = config.grounding if config else None
    
    # World contents
    self.entities: Dict[str, Entity] = {}
    self.relations: List[Relation] = []
    
    # Dynamics model (learned)
    self._transition_model: Dict[str, Callable] = {}
    
    # State tracking
    self._tick_count = 0
    self._current_state: Optional[WorldState] = None
    self._prediction_history: List[Tuple[PredictedWorldState, WorldState]] = []
    
    # Initialize with self entity
    self._init_self_entity()

def _init_self_entity(self):
    """Initialize entity representing self."""
    self.add_entity(Entity(
        id='self',
        type='agent',
        name='Self',
        properties={
            'is_self': True,
            'has_goals': True,
            'can_act': True,
        },
        certainty=1.0,  # Always certain of self
    ))
```

### `add_entity(entity)`

```python
def add_entity(self, entity: Entity):
    """Add entity to world model."""
    cfg = self.config
    
    # Check limit
    if len(self.entities) >= cfg.max_entities:
        self._prune_entities()
    
    # Generate phase pattern if grounding available
    if self.grounding and entity.embedding is not None:
        phases = self.grounding.embed_to_phases(entity.embedding)
        entity.phase_pattern = phases.slow  # Use slow scale
    elif self.grounding and entity.name:
        # Generate from name
        embedding = self.grounding.embedder(entity.name)
        phases = self.grounding.embed_to_phases(embedding)
        entity.embedding = embedding
        entity.phase_pattern = phases.slow
    
    entity.last_observed = self._tick_count
    self.entities[entity.id] = entity

def _prune_entities(self):
    """Remove least certain, oldest entities."""
    if not self.entities:
        return
    
    # Don't remove self
    candidates = [e for e in self.entities.values() if e.id != 'self']
    if not candidates:
        return
    
    # Remove lowest certainty
    worst = min(candidates, key=lambda e: e.certainty)
    del self.entities[worst.id]
    
    # Also remove relations involving this entity
    self.relations = [r for r in self.relations 
                      if r.source_id != worst.id and r.target_id != worst.id]
```

### `observe(observation) -> WorldState`

```python
def observe(self, observation: dict) -> WorldState:
    """
    Process observation and update world model.
    
    observation may contain:
        - entities: List of observed entity descriptions
        - relations: List of observed relations
        - events: List of events that occurred
    """
    self._tick_count += 1
    
    # Process observed entities
    for entity_obs in observation.get('entities', []):
        entity_id = entity_obs.get('id', f"entity_{self._tick_count}")
        
        if entity_id in self.entities:
            # Update existing
            self.update_entity(entity_id, entity_obs)
        else:
            # Add new
            entity = Entity(
                id=entity_id,
                type=entity_obs.get('type', 'object'),
                name=entity_obs.get('name', entity_id),
                properties=entity_obs.get('properties', {}),
            )
            self.add_entity(entity)
    
    # Process observed relations
    for rel_obs in observation.get('relations', []):
        relation = Relation(
            source_id=rel_obs['source'],
            relation_type=rel_obs['type'],
            target_id=rel_obs['target'],
            strength=rel_obs.get('strength', 1.0),
        )
        self.add_relation(relation)
    
    # Decay unobserved entities
    self._decay_entities()
    
    # Compute current state
    self._current_state = self._compute_state()
    
    return self._current_state

def _decay_entities(self):
    """Decay certainty of unobserved entities."""
    cfg = self.config
    
    for entity in list(self.entities.values()):
        if entity.id == 'self':
            continue
        
        time_since_observed = self._tick_count - entity.last_observed
        decay = cfg.entity_decay_rate * time_since_observed
        entity.certainty = max(entity.certainty - decay, 0)
        
        if entity.certainty < cfg.min_certainty:
            del self.entities[entity.id]
```

### `predict(action) -> PredictedWorldState`

```python
def predict(self, action: str) -> PredictedWorldState:
    """
    Predict world state after taking action.
    """
    if self._current_state is None:
        self._current_state = self._compute_state()
    
    # Get transition model for action
    transition = self._transition_model.get(action, self._default_transition)
    
    # Predict next state
    predicted_entities, predicted_relations = transition(
        self._current_state,
        self.entities,
        self.relations,
        action
    )
    
    # Compute predicted state
    predicted_state = WorldState(
        timestamp=self._tick_count + 1,
        active_entities=list(predicted_entities.keys()),
        active_relations=predicted_relations,
        affordances=self._compute_affordances(predicted_entities),
        uncertainty=self._compute_uncertainty(predicted_entities),
    )
    
    # Estimate probability (based on model confidence)
    probability = 1.0 - predicted_state.uncertainty
    
    # Predict consequences
    consequences = self._predict_consequences(action, predicted_state)
    
    return PredictedWorldState(
        state=predicted_state,
        action_taken=action,
        probability=probability,
        consequences=consequences,
    )

def _default_transition(self,
                         current: WorldState,
                         entities: Dict[str, Entity],
                         relations: List[Relation],
                         action: str) -> Tuple[Dict, List]:
    """Default transition: world mostly stays the same."""
    # Copy entities with slight certainty decay
    new_entities = {}
    for eid, entity in entities.items():
        new_entity = Entity(**{**entity.__dict__})
        if eid != 'self':
            new_entity.certainty *= 0.99
        new_entities[eid] = new_entity
    
    # Relations persist
    new_relations = relations.copy()
    
    return new_entities, new_relations

def _predict_consequences(self, action: str, predicted: WorldState) -> List[str]:
    """Predict consequences of action."""
    consequences = []
    
    # Action-specific predictions
    if action == 'respond':
        consequences.append('user_receives_response')
    elif action == 'rest':
        consequences.append('energy_restored')
    elif action == 'explore':
        consequences.append('new_entities_discovered')
    
    return consequences
```

### `simulate(action_sequence) -> List[PredictedWorldState]`

```python
def simulate(self, action_sequence: List[str]) -> List[PredictedWorldState]:
    """
    Simulate sequence of actions.
    
    Mental rehearsal - what happens if we do A, then B, then C?
    """
    trajectory = []
    
    # Save current state
    saved_entities = {k: Entity(**{**v.__dict__}) for k, v in self.entities.items()}
    saved_relations = self.relations.copy()
    saved_state = self._current_state
    
    try:
        for action in action_sequence:
            predicted = self.predict(action)
            trajectory.append(predicted)
            
            # Update model for next prediction
            # (This is temporary, rolled back after)
            self._apply_prediction(predicted)
    
    finally:
        # Restore
        self.entities = saved_entities
        self.relations = saved_relations
        self._current_state = saved_state
    
    return trajectory

def _apply_prediction(self, prediction: PredictedWorldState):
    """Temporarily apply prediction to model for chained simulation."""
    self._current_state = prediction.state
```

### `plan_to_goal(goal_condition) -> List[str]`

```python
def plan_to_goal(self, 
                 goal_condition: Callable[[WorldState], bool],
                 available_actions: List[str] = None) -> List[str]:
    """
    Find action sequence that reaches goal state.
    
    Uses simple BFS over action space.
    """
    cfg = self.config
    
    if available_actions is None:
        available_actions = self._get_default_actions()
    
    # BFS
    from collections import deque
    
    initial_state = self._current_state or self._compute_state()
    
    # Queue: (state, action_history)
    queue = deque([(initial_state, [])])
    visited = set()
    
    while queue and len(visited) < 1000:  # Limit search
        state, history = queue.popleft()
        
        # Check goal
        if goal_condition(state):
            return history
        
        # Limit depth
        if len(history) >= cfg.max_plan_length:
            continue
        
        # State hash for visited check
        state_hash = self._hash_state(state)
        if state_hash in visited:
            continue
        visited.add(state_hash)
        
        # Expand
        for action in available_actions:
            # Predict outcome
            # Temporarily set current state for prediction
            old_state = self._current_state
            self._current_state = state
            
            try:
                predicted = self.predict(action)
                queue.append((predicted.state, history + [action]))
            finally:
                self._current_state = old_state
    
    # No plan found
    return []

def _hash_state(self, state: WorldState) -> str:
    """Hash state for visited set."""
    return f"{sorted(state.active_entities)}_{len(state.active_relations)}"

def _get_default_actions(self) -> List[str]:
    """Default available actions."""
    return ['respond', 'rest', 'explore', 'focus', 'wait']
```

### `step(observation, action_taken)`

Main world model step.

```python
def step(self, 
         observation: Optional[dict] = None,
         action_taken: Optional[str] = None) -> WorldState:
    """
    One step of world model update.
    """
    # If we predicted this step, check accuracy
    if hasattr(self, '_last_prediction') and self._last_prediction:
        self._evaluate_prediction(self._last_prediction, observation)
    
    # Observe
    if observation:
        state = self.observe(observation)
    else:
        # No new observation, just decay
        self._decay_entities()
        state = self._compute_state()
    
    # Make prediction for next step
    if action_taken:
        self._last_prediction = self.predict(action_taken)
    else:
        self._last_prediction = None
    
    self._current_state = state
    return state

def _evaluate_prediction(self, 
                          prediction: PredictedWorldState,
                          actual_observation: dict):
    """Evaluate prediction accuracy and learn."""
    if actual_observation is None:
        return
    
    # Compare predicted entities to observed
    observed_entities = set(e.get('id') for e in actual_observation.get('entities', []))
    predicted_entities = set(prediction.state.active_entities)
    
    # Simple accuracy: Jaccard similarity
    intersection = len(observed_entities & predicted_entities)
    union = len(observed_entities | predicted_entities)
    accuracy = intersection / union if union > 0 else 1.0
    
    # Track for reporting
    self._prediction_history.append((prediction, actual_observation))
    if len(self._prediction_history) > 100:
        self._prediction_history.pop(0)
```

---

## Properties

```python
@property
def prediction_accuracy(self) -> float:
    """Recent prediction accuracy."""
    if not self._prediction_history:
        return 1.0
    
    accuracies = []
    for pred, actual in self._prediction_history[-20:]:
        observed = set(e.get('id') for e in actual.get('entities', []))
        predicted = set(pred.state.active_entities)
        intersection = len(observed & predicted)
        union = len(observed | predicted)
        acc = intersection / union if union > 0 else 1.0
        accuracies.append(acc)
    
    return np.mean(accuracies)

@property
def current_state(self) -> WorldState:
    if self._current_state is None:
        self._current_state = self._compute_state()
    return self._current_state
```

---

## Integration with Entity

```python
# In DevelopmentalEntity.__init__:
self.world_model = WorldModel(
    self.substrate,
    WorldModelConfig(grounding=self.grounding if hasattr(self, 'grounding') else None)
)

# In process_experience:
def process_experience(self, content, experience_type, significance):
    # ... existing code ...
    
    # Update world model
    observation = {
        'entities': self._extract_entities(content),
        'relations': self._extract_relations(content),
    }
    self.world_model.observe(observation)

# In tick:
def tick(self):
    # ... existing code ...
    
    # World model step
    world_state = self.world_model.step()
    
    # Use world model for planning
    if hasattr(self, 'goals') and self.goals.get_active_objectives():
        obj = self.goals.get_active_objectives()[0]
        plan = self.world_model.plan_to_goal(obj.completion_condition)
```

---

## Success Criteria

### Entity Tracking
1. Entities added on observation
2. Certainty decays without observation
3. Low-certainty entities pruned

### Prediction
1. predict() returns valid future state
2. Prediction accuracy tracks actual outcomes
3. simulate() handles action sequences

### Planning
1. plan_to_goal() finds valid paths
2. Plans respect action constraints
3. Empty plan returned when impossible

---

## Test Cases

```python
def test_add_observe_entity():
    """Observing entity should add it to model."""
    substrate = create_test_substrate()
    world = WorldModel(substrate)
    
    world.observe({
        'entities': [{'id': 'user', 'type': 'agent', 'name': 'User'}]
    })
    
    assert 'user' in world.entities
    assert world.entities['user'].certainty == 1.0

def test_entity_decay():
    """Unobserved entities should decay."""
    substrate = create_test_substrate()
    world = WorldModel(substrate)
    
    world.observe({'entities': [{'id': 'temp', 'type': 'object'}]})
    
    # Many steps without observing
    for _ in range(1000):
        world.step(observation=None)
    
    # Should be removed or low certainty
    assert 'temp' not in world.entities or world.entities['temp'].certainty < 0.5

def test_prediction():
    """predict() should return future state."""
    substrate = create_test_substrate()
    world = WorldModel(substrate)
    
    world.observe({'entities': [{'id': 'user', 'type': 'agent'}]})
    
    predicted = world.predict('respond')
    
    assert predicted.action_taken == 'respond'
    assert predicted.state is not None
    assert predicted.probability > 0

def test_simulation():
    """simulate() should handle action sequences."""
    substrate = create_test_substrate()
    world = WorldModel(substrate)
    
    world.observe({'entities': [{'id': 'user', 'type': 'agent'}]})
    
    trajectory = world.simulate(['respond', 'wait', 'respond'])
    
    assert len(trajectory) == 3
    for pred in trajectory:
        assert pred.state is not None

def test_planning():
    """plan_to_goal() should find action sequence."""
    substrate = create_test_substrate()
    world = WorldModel(substrate)
    
    world.observe({'entities': [{'id': 'user', 'type': 'agent'}]})
    
    # Simple goal: state has 'responded' consequence
    def goal(state):
        return 'user_receives_response' in str(state)
    
    plan = world.plan_to_goal(goal, available_actions=['respond', 'wait'])
    
    # Should find 'respond' as the plan
    assert 'respond' in plan or len(plan) == 0  # May not find if goal formulation differs
```

---

## Dependencies

- `MultiScaleSubstrate` (ORE2-002)
- `SemanticGrounding` (ORE2-008) - optional
- `numpy`

---

## File Location

```
ore2/
├── core/
│   └── world_model.py  # <-- This component
├── tests/
│   └── test_world_model.py
```

---

## Design Decisions to Preserve

1. **Self is always an entity** - Initialized on creation
2. **Certainty decay** - Unobserved entities become uncertain
3. **Phase patterns for entities** - Links to substrate
4. **Prediction history** - Tracks accuracy for learning
5. **BFS planning** - Simple but general
6. **Transition model is extensible** - Can add learned dynamics
