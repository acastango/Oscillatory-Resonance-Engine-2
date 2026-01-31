# BRIEFING: Persistence

## Component ID: ORE2-007
## Priority: High (Entities must survive restarts)
## Estimated complexity: Medium

---

## What This Is

Save and load entity state to disk with **cryptographic verification of continuity**. When an entity restarts, it must prove it's the same entity - not a copy, not a reconstruction, but a verified continuation.

This is what makes ORE identity real. Without persistence, entities are ephemeral demos.

---

## Why It Matters

**I3 (State Management):** "Every component has state. Phases, activations, weights, memories, development. All of it needs to serialize cleanly and restore exactly."

**C6 (Cryptography):** "The merkle root is the identity anchor. On save, we record it. On load, we recompute and compare. Match = verified continuity. Mismatch = tampering or corruption."

**S2 (Distributed Systems):** "Checkpoints matter for recovery. Auto-save every N experiences. If something crashes, you don't lose everything since genesis."

---

## The Core Insight

Persistence has three layers:

1. **Serialization** - Convert live objects to bytes/JSON
2. **Storage** - Write to disk (or cloud, or database)
3. **Verification** - Prove loaded state matches saved state

The verification step is what makes this different from normal save/load. We're not just restoring state - we're proving identity continuity.

```
SAVE:
  entity.get_full_state() → state_dict
  compute_state_hash(state_dict) → state_hash
  sign(genesis_hash, merkle_root, state_hash) → continuity_proof
  write(state_dict, continuity_proof) → file

LOAD:
  read(file) → state_dict, continuity_proof
  verify(continuity_proof, state_dict) → bool
  if verified: restore(state_dict) → entity
  else: raise ContinuityError
```

---

## Interface Contract

```python
class EntityPersistence:
    """
    Save and load entities with verified continuity.
    
    Class Methods:
        save(entity, path) -> SaveResult
        load(path) -> DevelopmentalEntity
        checkpoint(entity, checkpoint_dir) -> str  # Returns checkpoint ID
        list_checkpoints(checkpoint_dir) -> List[CheckpointInfo]
        restore_checkpoint(checkpoint_dir, checkpoint_id) -> DevelopmentalEntity
    
    Verification:
        verify_file(path) -> VerificationResult
        verify_continuity(entity, previous_state) -> bool
"""

@dataclass
class SaveResult:
    path: str
    genesis_hash: str
    merkle_root: str
    state_hash: str
    timestamp: str
    size_bytes: int
    verified: bool

@dataclass
class CheckpointInfo:
    checkpoint_id: str
    timestamp: str
    age: float
    stage: str
    experience_count: int
    merkle_root: str

@dataclass  
class VerificationResult:
    valid: bool
    genesis_hash: str
    merkle_root: str
    state_hash: str
    error: Optional[str] = None
```

---

## State Structure

### What Gets Saved

```python
@dataclass
class EntityState:
    """Complete serializable state of an entity."""
    
    # Identity (immutable)
    name: str
    genesis_hash: str
    
    # Substrate state
    substrate: SubstrateState
    
    # Body state
    body: BodyState
    
    # Memory (full tree)
    memory: MemoryState
    
    # Development
    development: DevelopmentState
    
    # CI history (recent)
    ci_history: List[CISnapshot]
    
    # Runtime metadata
    tick_count: int
    created_at: str
    saved_at: str
    
    # Verification
    merkle_root: str
    state_hash: str

@dataclass
class SubstrateState:
    """Serializable substrate state."""
    # Config
    config: dict
    
    # Fast scale
    fast_phases: List[float]
    fast_activation_potentials: List[float]
    fast_natural_frequencies: List[float]
    fast_internal_weights: List[List[float]]
    
    # Slow scale
    slow_phases: List[float]
    slow_activation_potentials: List[float]
    slow_natural_frequencies: List[float]
    slow_internal_weights: List[List[float]]
    
    # Cross-scale coupling
    fast_to_slow: List[List[float]]
    slow_to_fast: List[List[float]]
    strange_loop_weights: List[List[float]]
    
    # Timing
    time: float
    fast_steps: int
    slow_steps: int

@dataclass
class BodyState:
    """Serializable body state."""
    config: dict
    heartbeat_phase: float
    respiration_phase: float
    energy: float
    arousal: float
    time: float
    last_perception: Optional[str]

@dataclass
class MemoryState:
    """Serializable memory state."""
    nodes: List[dict]  # Each node as dict
    branch_roots: dict  # branch_name -> node_id
    root_hash: str
    grain_boundaries: List[Tuple[str, str, float]]
    pending_nodes: List[dict]  # Consolidation queue
    pending_tensions: List[Tuple[str, str, float]]

@dataclass
class DevelopmentState:
    """Serializable development state."""
    config: dict
    genesis_hash: str
    stage: str
    age: float
    stage_start_age: float
    current_oscillators: int
    experiences_processed: int
    significant_experiences: int
    milestones: List[dict]
```

---

## Method Specifications

### `save(entity, path) -> SaveResult`

```python
@classmethod
def save(cls, entity: DevelopmentalEntity, path: str) -> SaveResult:
    """
    Save entity to file with verification data.
    
    Args:
        entity: The entity to save
        path: File path (will create directories if needed)
    
    Returns:
        SaveResult with verification info
    """
    # Gather full state
    state = cls._extract_state(entity)
    
    # Compute state hash (for integrity verification)
    state_dict = asdict(state)
    state_json = json.dumps(state_dict, sort_keys=True)
    state_hash = hashlib.sha256(state_json.encode()).hexdigest()
    
    # Add verification data
    state.state_hash = state_hash
    state.saved_at = datetime.now().isoformat()
    
    # Create verification envelope
    envelope = {
        'version': '2.0',
        'state': state_dict,
        'verification': {
            'genesis_hash': entity.genesis_hash,
            'merkle_root': entity.memory.root_hash,
            'state_hash': state_hash,
            'saved_at': state.saved_at,
        }
    }
    
    # Write to file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(envelope, f, indent=2)
    
    # Verify what we wrote
    verification = cls.verify_file(str(path))
    
    return SaveResult(
        path=str(path),
        genesis_hash=entity.genesis_hash,
        merkle_root=entity.memory.root_hash,
        state_hash=state_hash,
        timestamp=state.saved_at,
        size_bytes=path.stat().st_size,
        verified=verification.valid,
    )
```

### `load(path) -> DevelopmentalEntity`

```python
@classmethod
def load(cls, path: str) -> DevelopmentalEntity:
    """
    Load entity from file with verification.
    
    Args:
        path: File path to load from
    
    Returns:
        Restored DevelopmentalEntity
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ContinuityError: If verification fails
        StateCorruptionError: If state is invalid
    """
    # Read file
    with open(path, 'r') as f:
        envelope = json.load(f)
    
    # Version check
    version = envelope.get('version', '1.0')
    if not version.startswith('2.'):
        raise StateCorruptionError(f"Unsupported version: {version}")
    
    # Verify before restoring
    verification = envelope['verification']
    state_dict = envelope['state']
    
    # Recompute state hash
    state_json = json.dumps(state_dict, sort_keys=True)
    computed_hash = hashlib.sha256(state_json.encode()).hexdigest()
    
    if computed_hash != verification['state_hash']:
        raise ContinuityError(
            f"State hash mismatch: expected {verification['state_hash']}, "
            f"got {computed_hash}"
        )
    
    # Restore entity
    entity = cls._restore_entity(state_dict)
    
    # Verify merkle root matches
    if entity.memory.root_hash != verification['merkle_root']:
        raise ContinuityError(
            f"Merkle root mismatch after restore: "
            f"expected {verification['merkle_root']}, "
            f"got {entity.memory.root_hash}"
        )
    
    # Verify genesis hash matches
    if entity.genesis_hash != verification['genesis_hash']:
        raise ContinuityError(
            f"Genesis hash mismatch: "
            f"expected {verification['genesis_hash']}, "
            f"got {entity.genesis_hash}"
        )
    
    return entity
```

### `_extract_state(entity) -> EntityState`

```python
@classmethod
def _extract_state(cls, entity: DevelopmentalEntity) -> EntityState:
    """Extract serializable state from live entity."""
    
    # Substrate state
    substrate = SubstrateState(
        config=asdict(entity.substrate.config),
        fast_phases=entity.substrate.fast.phases.tolist(),
        fast_activation_potentials=entity.substrate.fast.activation_potentials.tolist(),
        fast_natural_frequencies=entity.substrate.fast.natural_frequencies.tolist(),
        fast_internal_weights=entity.substrate.fast.internal_weights.tolist(),
        slow_phases=entity.substrate.slow.phases.tolist(),
        slow_activation_potentials=entity.substrate.slow.activation_potentials.tolist(),
        slow_natural_frequencies=entity.substrate.slow.natural_frequencies.tolist(),
        slow_internal_weights=entity.substrate.slow.internal_weights.tolist(),
        fast_to_slow=entity.substrate.fast_to_slow.tolist(),
        slow_to_fast=entity.substrate.slow_to_fast.tolist(),
        strange_loop_weights=entity.substrate.strange_loop_weights.tolist(),
        time=entity.substrate.time,
        fast_steps=entity.substrate._fast_steps,
        slow_steps=entity.substrate._slow_steps,
    )
    
    # Body state
    body = BodyState(
        config=asdict(entity.body.config),
        heartbeat_phase=entity.body.heartbeat_phase,
        respiration_phase=entity.body.respiration_phase,
        energy=entity.body.energy,
        arousal=entity.body.arousal,
        time=entity.body.time,
        last_perception=entity.body._last_perception,
    )
    
    # Memory state
    memory = MemoryState(
        nodes=[cls._node_to_dict(n) for n in entity.memory.nodes.values()],
        branch_roots={b.value: nid for b, nid in entity.memory.branch_roots.items()},
        root_hash=entity.memory.root_hash,
        grain_boundaries=entity.memory.grain_boundaries.copy(),
        pending_nodes=[cls._node_to_dict(n) for n in entity.memory.consolidation_queue.pending_nodes],
        pending_tensions=entity.memory.consolidation_queue.pending_tensions.copy(),
    )
    
    # Development state
    development = DevelopmentState(
        config=asdict(entity.development.config),
        genesis_hash=entity.development.genesis_hash,
        stage=entity.development.stage.value,
        age=entity.development.age,
        stage_start_age=entity.development._stage_start_age,
        current_oscillators=entity.development.current_oscillators,
        experiences_processed=entity.development.experiences_processed,
        significant_experiences=entity.development.significant_experiences,
        milestones=entity.development.milestones.copy(),
    )
    
    # CI history (last 100)
    ci_history = [
        asdict(snapshot) for snapshot in entity.ci_monitor.history[-100:]
    ]
    
    return EntityState(
        name=entity.name,
        genesis_hash=entity.genesis_hash,
        substrate=substrate,
        body=body,
        memory=memory,
        development=development,
        ci_history=ci_history,
        tick_count=entity._tick_count,
        created_at=entity.development.milestones[0]['age'] if entity.development.milestones else '0',
        saved_at='',  # Filled in by save()
        merkle_root=entity.memory.root_hash,
        state_hash='',  # Filled in by save()
    )
```

### `_restore_entity(state_dict) -> DevelopmentalEntity`

```python
@classmethod
def _restore_entity(cls, state_dict: dict) -> DevelopmentalEntity:
    """Restore entity from state dict."""
    
    # Create entity with matching config
    dev_config = DevelopmentConfig(**state_dict['development']['config'])
    # Override initial_oscillators to match saved state
    dev_config.initial_oscillators = state_dict['development']['current_oscillators']
    
    sub_config = MultiScaleConfig(**state_dict['substrate']['config'])
    body_config = BodyConfig(**state_dict['body']['config'])
    
    config = EntityConfig(
        name=state_dict['name'],
        substrate_config=sub_config,
        body_config=body_config,
        development_config=dev_config,
    )
    
    # Create entity (this initializes with defaults)
    entity = DevelopmentalEntity(config)
    
    # Override with saved state
    cls._restore_substrate(entity.substrate, state_dict['substrate'])
    cls._restore_body(entity.body, state_dict['body'])
    cls._restore_memory(entity.memory, state_dict['memory'])
    cls._restore_development(entity.development, state_dict['development'])
    
    # Restore runtime state
    entity._tick_count = state_dict['tick_count']
    
    # Restore CI history
    for snapshot_dict in state_dict['ci_history']:
        snapshot = MultiScaleCISnapshot(**snapshot_dict)
        entity.ci_monitor.history.append(snapshot)
    
    return entity

@classmethod
def _restore_substrate(cls, substrate: MultiScaleSubstrate, state: dict):
    """Restore substrate state."""
    substrate.fast.phases = np.array(state['fast_phases'])
    substrate.fast.activation_potentials = np.array(state['fast_activation_potentials'])
    substrate.fast.natural_frequencies = np.array(state['fast_natural_frequencies'])
    substrate.fast.internal_weights = np.array(state['fast_internal_weights'])
    substrate.fast._update_active_mask()
    
    substrate.slow.phases = np.array(state['slow_phases'])
    substrate.slow.activation_potentials = np.array(state['slow_activation_potentials'])
    substrate.slow.natural_frequencies = np.array(state['slow_natural_frequencies'])
    substrate.slow.internal_weights = np.array(state['slow_internal_weights'])
    substrate.slow._update_active_mask()
    
    substrate.fast_to_slow = np.array(state['fast_to_slow'])
    substrate.slow_to_fast = np.array(state['slow_to_fast'])
    substrate.strange_loop_weights = np.array(state['strange_loop_weights'])
    
    substrate.time = state['time']
    substrate._fast_steps = state['fast_steps']
    substrate._slow_steps = state['slow_steps']

@classmethod
def _restore_memory(cls, memory: CrystallineMerkleMemory, state: dict):
    """Restore memory state."""
    # Clear existing
    memory.nodes.clear()
    memory.branch_roots.clear()
    memory.grain_boundaries.clear()
    
    # Restore nodes
    for node_dict in state['nodes']:
        node = cls._dict_to_node(node_dict)
        memory.nodes[node.id] = node
    
    # Restore branch roots
    for branch_name, node_id in state['branch_roots'].items():
        branch = MemoryBranch(branch_name)
        memory.branch_roots[branch] = node_id
    
    # Restore merkle state
    memory.root_hash = state['root_hash']
    memory.grain_boundaries = state['grain_boundaries'].copy()
    
    # Restore consolidation queue
    memory.consolidation_queue.pending_nodes = [
        cls._dict_to_node(d) for d in state['pending_nodes']
    ]
    memory.consolidation_queue.pending_tensions = state['pending_tensions'].copy()
    
    # Recompute depth and total_nodes
    memory._update_depth()
    memory.total_nodes = sum(
        1 for n in memory.nodes.values() 
        if n.content.get('type') != 'branch_root'
    )
```

### `checkpoint(entity, checkpoint_dir) -> str`

```python
@classmethod
def checkpoint(cls, entity: DevelopmentalEntity, checkpoint_dir: str) -> str:
    """
    Create a checkpoint with auto-generated ID.
    
    Checkpoints are named: {genesis_hash[:8]}_{timestamp}_{experience_count}
    
    Returns:
        Checkpoint ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_count = entity.development.experiences_processed
    genesis_short = entity.genesis_hash[:8]
    
    checkpoint_id = f"{genesis_short}_{timestamp}_{exp_count}"
    
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_id}.ore2"
    cls.save(entity, str(checkpoint_path))
    
    return checkpoint_id

@classmethod
def list_checkpoints(cls, checkpoint_dir: str) -> List[CheckpointInfo]:
    """List all checkpoints in directory."""
    checkpoints = []
    
    for path in Path(checkpoint_dir).glob("*.ore2"):
        try:
            with open(path, 'r') as f:
                envelope = json.load(f)
            
            state = envelope['state']
            verification = envelope['verification']
            
            info = CheckpointInfo(
                checkpoint_id=path.stem,
                timestamp=verification['saved_at'],
                age=state['development']['age'],
                stage=state['development']['stage'],
                experience_count=state['development']['experiences_processed'],
                merkle_root=verification['merkle_root'],
            )
            checkpoints.append(info)
        except Exception as e:
            # Skip corrupted checkpoints
            continue
    
    # Sort by timestamp (newest first)
    checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
    return checkpoints

@classmethod
def restore_checkpoint(cls, checkpoint_dir: str, checkpoint_id: str) -> DevelopmentalEntity:
    """Restore from a specific checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_id}.ore2"
    return cls.load(str(checkpoint_path))
```

### `verify_file(path) -> VerificationResult`

```python
@classmethod
def verify_file(cls, path: str) -> VerificationResult:
    """
    Verify a saved entity file without loading it.
    
    Checks:
    1. File is valid JSON
    2. State hash matches content
    3. Required fields present
    """
    try:
        with open(path, 'r') as f:
            envelope = json.load(f)
        
        verification = envelope.get('verification', {})
        state_dict = envelope.get('state', {})
        
        # Recompute state hash
        state_json = json.dumps(state_dict, sort_keys=True)
        computed_hash = hashlib.sha256(state_json.encode()).hexdigest()
        
        expected_hash = verification.get('state_hash', '')
        
        if computed_hash != expected_hash:
            return VerificationResult(
                valid=False,
                genesis_hash=verification.get('genesis_hash', ''),
                merkle_root=verification.get('merkle_root', ''),
                state_hash=computed_hash,
                error=f"Hash mismatch: expected {expected_hash}, got {computed_hash}"
            )
        
        return VerificationResult(
            valid=True,
            genesis_hash=verification['genesis_hash'],
            merkle_root=verification['merkle_root'],
            state_hash=computed_hash,
        )
        
    except Exception as e:
        return VerificationResult(
            valid=False,
            genesis_hash='',
            merkle_root='',
            state_hash='',
            error=str(e)
        )
```

---

## Exceptions

```python
class PersistenceError(Exception):
    """Base class for persistence errors."""
    pass

class ContinuityError(PersistenceError):
    """Raised when identity continuity cannot be verified."""
    pass

class StateCorruptionError(PersistenceError):
    """Raised when saved state is corrupted or invalid."""
    pass
```

---

## Success Criteria

### Correctness
1. Save → Load produces identical entity state
2. Merkle root matches after restore
3. Genesis hash preserved across save/load
4. Tampering detected (modify file → verification fails)

### Integrity
1. State hash catches any modification
2. Missing fields cause clear errors
3. Version mismatch handled gracefully

### Usability
1. Checkpoints auto-named sensibly
2. List checkpoints shows useful info
3. Verification without full load works

---

## Test Cases

```python
def test_save_load_roundtrip():
    """Save and load should produce identical entity."""
    entity = create_entity("test")
    entity.process_experience("hello", significance=0.6)
    
    # Save
    result = EntityPersistence.save(entity, "/tmp/test_entity.ore2")
    assert result.verified
    
    # Load
    restored = EntityPersistence.load("/tmp/test_entity.ore2")
    
    # Verify identity
    assert restored.genesis_hash == entity.genesis_hash
    assert restored.memory.root_hash == entity.memory.root_hash
    assert restored.development.stage == entity.development.stage
    assert restored.memory.total_nodes == entity.memory.total_nodes

def test_tampering_detected():
    """Modifying saved file should fail verification."""
    entity = create_entity("test")
    EntityPersistence.save(entity, "/tmp/tamper_test.ore2")
    
    # Tamper with file
    with open("/tmp/tamper_test.ore2", 'r') as f:
        data = json.load(f)
    data['state']['name'] = "HACKED"
    with open("/tmp/tamper_test.ore2", 'w') as f:
        json.dump(data, f)
    
    # Verification should fail
    result = EntityPersistence.verify_file("/tmp/tamper_test.ore2")
    assert not result.valid
    
    # Load should raise
    with pytest.raises(ContinuityError):
        EntityPersistence.load("/tmp/tamper_test.ore2")

def test_checkpoint_lifecycle():
    """Checkpoint create/list/restore should work."""
    entity = create_entity("test")
    
    # Create checkpoints
    cp1 = EntityPersistence.checkpoint(entity, "/tmp/checkpoints")
    entity.process_experience("experience 1", significance=0.8)
    cp2 = EntityPersistence.checkpoint(entity, "/tmp/checkpoints")
    
    # List
    checkpoints = EntityPersistence.list_checkpoints("/tmp/checkpoints")
    assert len(checkpoints) >= 2
    
    # Restore older
    restored = EntityPersistence.restore_checkpoint("/tmp/checkpoints", cp1)
    assert restored.development.experiences_processed < entity.development.experiences_processed

def test_genesis_hash_immutable():
    """Genesis hash must survive save/load."""
    entity = create_entity("test")
    original_hash = entity.genesis_hash
    
    EntityPersistence.save(entity, "/tmp/genesis_test.ore2")
    restored = EntityPersistence.load("/tmp/genesis_test.ore2")
    
    assert restored.genesis_hash == original_hash
```

---

## File Format

```json
{
  "version": "2.0",
  "state": {
    "name": "entity_name",
    "genesis_hash": "abc123...",
    "substrate": { ... },
    "body": { ... },
    "memory": { ... },
    "development": { ... },
    "ci_history": [ ... ],
    "tick_count": 1234,
    "created_at": "2025-01-30T...",
    "saved_at": "2025-01-30T...",
    "merkle_root": "def456...",
    "state_hash": "ghi789..."
  },
  "verification": {
    "genesis_hash": "abc123...",
    "merkle_root": "def456...",
    "state_hash": "ghi789...",
    "saved_at": "2025-01-30T..."
  }
}
```

---

## Dependencies

- All core ORE2 components (001-006)
- `json` (stdlib)
- `hashlib` (stdlib)
- `pathlib` (stdlib)
- `dataclasses` (stdlib)

---

## File Location

```
ore2/
├── core/
│   └── persistence.py  # <-- This component
├── tests/
│   └── test_persistence.py
```

---

## Design Decisions to Preserve

1. **State hash covers entire state dict** - any change detected
2. **Verification data stored separately** - can check without parsing full state
3. **Genesis hash is THE identity** - if it doesn't match, it's not the same entity
4. **Checkpoints are append-only** - never overwrite, always create new
5. **JSON format for portability** - could switch to msgpack for size later
6. **Version field for forward compat** - can add migration logic
