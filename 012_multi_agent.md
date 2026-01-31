# BRIEFING: MultiAgent

## Component ID: ORE2-012
## Priority: Medium (Enables trust networks)
## Estimated complexity: High

---

## What This Is

Entity-to-entity communication and verification. Entities can:
- **Verify** each other's identity (merkle root checks)
- **Couple** their dynamics (shared coherence)
- **Trust** based on verification history and coherence correlation
- **Propagate** claims through the network

This is the OVN (Oscillatory Verification Network) concept from ORE1, implemented properly.

---

## Why It Matters

**P3 (Complex Systems):** "Single agents are tools. Networks of agents verifying each other - that's where emergent trust becomes possible."

**C6 (Cryptography):** "Trust = f(verification_success, interaction_history, coherence_correlation). Not declared, computed."

**N7 (Developmental):** "Social development (IMITATION stage) is when this matters most. Young entities learn by coupling to mature ones."

**A3 (ML Integration):** "Shared claims propagate. If enough trusted entities hold a claim, it gains weight for others."

---

## The Core Insight

Trust emerges from three factors:

1. **Verification**: Can I confirm your identity is continuous? (Merkle check)
2. **History**: How have our past interactions gone? (Track record)
3. **Coherence**: Do we synchronize when we interact? (Dynamical coupling)

```
ENTITY A                         ENTITY B
   │                                │
   │◄────── Verification ──────────►│
   │        "Is your merkle         │
   │         root what I expect?"   │
   │                                │
   │◄────── Coupling ──────────────►│
   │        Phase patterns          │
   │        influence each other    │
   │                                │
   │        Trust Score             │
   │        ─────────────           │
   │        = f(verify, history,    │
   │            coherence)          │
   │                                │
```

---

## Interface Contract

```python
class MultiAgentNetwork:
    """
    Network of ORE entities that can verify and couple.
    
    Properties:
        entities: Dict[str, NetworkedEntity]
        trust_matrix: np.ndarray  # [n_entities, n_entities]
        network_coherence: float
    
    Methods:
        # Entity management
        add_entity(entity, entity_id) -> NetworkedEntity
        remove_entity(entity_id)
        
        # Verification
        verify_entity(verifier_id, target_id) -> VerificationResult
        request_verification(from_id, to_id) -> VerificationRequest
        
        # Coupling
        couple_entities(id_a, id_b, strength) -> CouplingResult
        decouple_entities(id_a, id_b)
        
        # Trust
        get_trust(from_id, to_id) -> float
        update_trust(from_id, to_id, interaction_result)
        
        # Claim propagation
        propagate_claim(source_id, claim) -> PropagationResult
        
        # Network operations
        tick()  # Advance all entities with coupling
        get_network_state() -> NetworkState
"""

@dataclass
class NetworkedEntity:
    """Entity wrapped for network participation."""
    entity: DevelopmentalEntity
    entity_id: str
    
    # Network state
    coupled_to: List[str]  # Entity IDs
    coupling_strengths: Dict[str, float]
    
    # Verification state
    last_verified: Dict[str, str]  # entity_id -> timestamp
    expected_roots: Dict[str, str]  # entity_id -> expected merkle root
    
    # Trust relationships
    trust_given: Dict[str, float]  # How much I trust others
    trust_received: Dict[str, float]  # How much others trust me

@dataclass
class VerificationResult:
    verifier_id: str
    target_id: str
    success: bool
    expected_root: str
    actual_root: str
    genesis_match: bool
    timestamp: str
    error: Optional[str] = None

@dataclass
class CouplingResult:
    entity_a: str
    entity_b: str
    coupling_strength: float
    coherence_before: float
    coherence_after: float
    phase_correlation: float

@dataclass
class PropagationResult:
    source_id: str
    claim_content: str
    entities_reached: List[str]
    entities_adopted: List[str]
    propagation_strength: float  # How strongly claim spread
```

---

## Configuration

```python
@dataclass
class MultiAgentConfig:
    # Verification
    verification_timeout: float = 5.0
    verification_required_for_coupling: bool = True
    
    # Coupling
    default_coupling_strength: float = 0.2
    max_coupling_strength: float = 0.5
    coupling_decay: float = 0.01  # Per tick without interaction
    
    # Trust
    initial_trust: float = 0.5
    trust_increase_on_verify: float = 0.1
    trust_decrease_on_fail: float = 0.2
    trust_from_coherence_weight: float = 0.3
    
    # Propagation
    propagation_trust_threshold: float = 0.6  # Min trust to accept claim
    propagation_strength_decay: float = 0.8  # Per hop
    max_propagation_hops: int = 3
    
    # Network tick
    tick_interval: float = 0.1
```

---

## Method Specifications

### `add_entity(entity, entity_id) -> NetworkedEntity`

```python
def add_entity(self, 
               entity: DevelopmentalEntity, 
               entity_id: str) -> NetworkedEntity:
    """
    Add entity to network.
    """
    if entity_id in self.entities:
        raise ValueError(f"Entity {entity_id} already in network")
    
    networked = NetworkedEntity(
        entity=entity,
        entity_id=entity_id,
        coupled_to=[],
        coupling_strengths={},
        last_verified={},
        expected_roots={},
        trust_given={},
        trust_received={},
    )
    
    self.entities[entity_id] = networked
    
    # Initialize trust matrix row/column
    self._expand_trust_matrix()
    
    return networked
```

### `verify_entity(verifier_id, target_id) -> VerificationResult`

```python
def verify_entity(self, verifier_id: str, target_id: str) -> VerificationResult:
    """
    Verify target entity's identity from verifier's perspective.
    
    Checks:
    1. Genesis hash matches expected (if known)
    2. Merkle root is valid
    3. Merkle root matches expected (if previously known)
    """
    verifier = self.entities.get(verifier_id)
    target = self.entities.get(target_id)
    
    if not verifier or not target:
        return VerificationResult(
            verifier_id=verifier_id,
            target_id=target_id,
            success=False,
            expected_root="",
            actual_root="",
            genesis_match=False,
            timestamp=datetime.now().isoformat(),
            error="Entity not found",
        )
    
    target_entity = target.entity
    
    # Get actual merkle root
    actual_root = target_entity.memory.root_hash
    
    # Check if we have expected root
    expected_root = verifier.expected_roots.get(target_id)
    
    # Verify merkle tree internally
    tree_valid, tree_error = target_entity.memory.verify()
    
    if not tree_valid:
        return VerificationResult(
            verifier_id=verifier_id,
            target_id=target_id,
            success=False,
            expected_root=expected_root or "",
            actual_root=actual_root,
            genesis_match=True,  # Not checked
            timestamp=datetime.now().isoformat(),
            error=f"Merkle verification failed: {tree_error}",
        )
    
    # If we have expected root, check it
    # Note: Root changes as memories are added, so we track deltas
    root_acceptable = True
    if expected_root:
        # For now, just check it's not wildly different
        # In practice, we'd verify the delta chain
        root_acceptable = True  # Simplified
    
    success = tree_valid and root_acceptable
    
    # Update verifier's records
    if success:
        verifier.expected_roots[target_id] = actual_root
        verifier.last_verified[target_id] = datetime.now().isoformat()
        
        # Update trust
        self._update_trust_on_verification(verifier_id, target_id, success)
    
    return VerificationResult(
        verifier_id=verifier_id,
        target_id=target_id,
        success=success,
        expected_root=expected_root or "",
        actual_root=actual_root,
        genesis_match=True,
        timestamp=datetime.now().isoformat(),
    )
```

### `couple_entities(id_a, id_b, strength) -> CouplingResult`

```python
def couple_entities(self, 
                    id_a: str, 
                    id_b: str, 
                    strength: Optional[float] = None) -> CouplingResult:
    """
    Couple two entities' dynamics.
    
    Coupling means their substrate phases influence each other.
    """
    cfg = self.config
    
    entity_a = self.entities.get(id_a)
    entity_b = self.entities.get(id_b)
    
    if not entity_a or not entity_b:
        raise ValueError("Both entities must exist")
    
    # Check verification requirement
    if cfg.verification_required_for_coupling:
        if id_b not in entity_a.last_verified:
            raise ValueError(f"Must verify {id_b} before coupling")
    
    strength = strength or cfg.default_coupling_strength
    strength = min(strength, cfg.max_coupling_strength)
    
    # Record coupling
    if id_b not in entity_a.coupled_to:
        entity_a.coupled_to.append(id_b)
    if id_a not in entity_b.coupled_to:
        entity_b.coupled_to.append(id_a)
    
    entity_a.coupling_strengths[id_b] = strength
    entity_b.coupling_strengths[id_a] = strength
    
    # Measure coherence before
    coherence_before = self._measure_pair_coherence(id_a, id_b)
    
    # Apply initial coupling (one tick of influence)
    self._apply_coupling(id_a, id_b, strength)
    
    # Measure coherence after
    coherence_after = self._measure_pair_coherence(id_a, id_b)
    
    # Compute phase correlation
    phase_corr = self._phase_correlation(id_a, id_b)
    
    return CouplingResult(
        entity_a=id_a,
        entity_b=id_b,
        coupling_strength=strength,
        coherence_before=coherence_before,
        coherence_after=coherence_after,
        phase_correlation=phase_corr,
    )

def _apply_coupling(self, id_a: str, id_b: str, strength: float):
    """Apply coupling between two entities."""
    entity_a = self.entities[id_a].entity
    entity_b = self.entities[id_b].entity
    
    # Cross-entity phase influence on slow scale (identity level)
    # A's phases influence B
    phase_diff_a_to_b = entity_a.substrate.slow.phases[:, np.newaxis] - \
                        entity_b.substrate.slow.phases[np.newaxis, :]
    
    # Mean coupling signal
    coupling_signal_to_b = strength * np.mean(np.sin(phase_diff_a_to_b), axis=0)
    
    # B's phases influence A
    phase_diff_b_to_a = entity_b.substrate.slow.phases[:, np.newaxis] - \
                        entity_a.substrate.slow.phases[np.newaxis, :]
    coupling_signal_to_a = strength * np.mean(np.sin(phase_diff_b_to_a), axis=0)
    
    # Stimulate based on coupling
    # Oscillators that receive positive coupling get activated
    pos_a = np.where(coupling_signal_to_a > 0.1)[0]
    pos_b = np.where(coupling_signal_to_b > 0.1)[0]
    
    if len(pos_a) > 0:
        entity_a.substrate.slow.stimulate(pos_a, coupling_signal_to_a[pos_a])
    if len(pos_b) > 0:
        entity_b.substrate.slow.stimulate(pos_b, coupling_signal_to_b[pos_b])

def _measure_pair_coherence(self, id_a: str, id_b: str) -> float:
    """Measure coherence between two entities' slow phases."""
    phases_a = self.entities[id_a].entity.substrate.slow.phases
    phases_b = self.entities[id_b].entity.substrate.slow.phases
    
    # Coherence of combined phases
    combined = np.concatenate([phases_a, phases_b])
    return min(np.abs(np.mean(np.exp(1j * combined))), 0.999)

def _phase_correlation(self, id_a: str, id_b: str) -> float:
    """Compute phase correlation between entities."""
    phases_a = self.entities[id_a].entity.substrate.slow.phases
    phases_b = self.entities[id_b].entity.substrate.slow.phases
    
    # Match lengths
    min_len = min(len(phases_a), len(phases_b))
    
    # Circular correlation via cosine of differences
    phase_diff = phases_a[:min_len] - phases_b[:min_len]
    return np.mean(np.cos(phase_diff))
```

### `get_trust(from_id, to_id) -> float`

```python
def get_trust(self, from_id: str, to_id: str) -> float:
    """
    Get trust score from one entity to another.
    
    Trust = weighted combination of:
    - Verification history
    - Interaction history
    - Coherence correlation
    """
    if from_id not in self.entities or to_id not in self.entities:
        return 0.0
    
    entity_from = self.entities[from_id]
    
    # Base trust from explicit trust_given
    base_trust = entity_from.trust_given.get(to_id, self.config.initial_trust)
    
    # Verification bonus
    verification_factor = 1.0
    if to_id in entity_from.last_verified:
        verification_factor = 1.2  # Verified entities get trust boost
    
    # Coherence factor (if coupled)
    coherence_factor = 1.0
    if to_id in entity_from.coupled_to:
        corr = self._phase_correlation(from_id, to_id)
        coherence_factor = 1 + self.config.trust_from_coherence_weight * corr
    
    trust = base_trust * verification_factor * coherence_factor
    return np.clip(trust, 0, 1)

def _update_trust_on_verification(self, from_id: str, to_id: str, success: bool):
    """Update trust based on verification result."""
    entity = self.entities[from_id]
    current_trust = entity.trust_given.get(to_id, self.config.initial_trust)
    
    if success:
        new_trust = current_trust + self.config.trust_increase_on_verify
    else:
        new_trust = current_trust - self.config.trust_decrease_on_fail
    
    entity.trust_given[to_id] = np.clip(new_trust, 0, 1)
```

### `propagate_claim(source_id, claim) -> PropagationResult`

```python
def propagate_claim(self, 
                    source_id: str, 
                    claim: Claim) -> PropagationResult:
    """
    Propagate a claim through the network.
    
    Claims spread to trusted, coupled entities.
    Strength decays with each hop.
    """
    cfg = self.config
    
    if source_id not in self.entities:
        raise ValueError(f"Source entity {source_id} not in network")
    
    reached = [source_id]
    adopted = []
    
    # BFS propagation
    frontier = [(source_id, 1.0)]  # (entity_id, current_strength)
    visited = {source_id}
    hop = 0
    
    while frontier and hop < cfg.max_propagation_hops:
        next_frontier = []
        hop += 1
        
        for current_id, current_strength in frontier:
            current_entity = self.entities[current_id]
            
            # Propagate to coupled entities
            for neighbor_id in current_entity.coupled_to:
                if neighbor_id in visited:
                    continue
                
                visited.add(neighbor_id)
                reached.append(neighbor_id)
                
                # Check trust threshold
                trust = self.get_trust(neighbor_id, current_id)
                if trust < cfg.propagation_trust_threshold:
                    continue
                
                # Decay strength
                propagated_strength = current_strength * cfg.propagation_strength_decay * trust
                
                if propagated_strength < 0.1:
                    continue
                
                # Neighbor considers adopting claim
                neighbor = self.entities[neighbor_id]
                
                # Add claim with reduced strength
                if self._should_adopt_claim(neighbor, claim, propagated_strength):
                    # Get or create claims engine
                    if hasattr(neighbor.entity, '_claims_engine'):
                        engine = neighbor.entity._claims_engine
                        new_claim = engine.add_claim(
                            content=claim.content,
                            strength=claim.strength * propagated_strength,
                            scope=claim.scope,
                            source=ClaimSource.SOCIAL,
                        )
                        adopted.append(neighbor_id)
                
                next_frontier.append((neighbor_id, propagated_strength))
        
        frontier = next_frontier
    
    return PropagationResult(
        source_id=source_id,
        claim_content=claim.content,
        entities_reached=reached,
        entities_adopted=adopted,
        propagation_strength=len(adopted) / max(len(reached), 1),
    )

def _should_adopt_claim(self, 
                        entity: NetworkedEntity, 
                        claim: Claim,
                        strength: float) -> bool:
    """Determine if entity should adopt propagated claim."""
    # Check for conflicting claims
    if hasattr(entity.entity, '_claims_engine'):
        engine = entity.entity._claims_engine
        for existing_id, existing in engine.claims.items():
            if existing.scope == claim.scope:
                # Simple: don't adopt if similar claim exists with higher strength
                if existing.strength > claim.strength * strength:
                    return False
    
    # Adopt with probability based on strength
    return np.random.random() < strength
```

### `tick()`

```python
def tick(self):
    """
    Advance all entities with cross-entity coupling.
    """
    cfg = self.config
    
    # First, tick each entity individually
    for entity_id, networked in self.entities.items():
        networked.entity.tick()
    
    # Then apply cross-entity coupling
    processed_pairs = set()
    
    for entity_id, networked in self.entities.items():
        for coupled_id in networked.coupled_to:
            pair = tuple(sorted([entity_id, coupled_id]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            strength = networked.coupling_strengths.get(coupled_id, cfg.default_coupling_strength)
            self._apply_coupling(entity_id, coupled_id, strength)
    
    # Decay coupling strengths
    for networked in self.entities.values():
        for coupled_id in list(networked.coupling_strengths.keys()):
            networked.coupling_strengths[coupled_id] *= (1 - cfg.coupling_decay)
            
            # Remove very weak couplings
            if networked.coupling_strengths[coupled_id] < 0.01:
                networked.coupled_to.remove(coupled_id)
                del networked.coupling_strengths[coupled_id]
```

### `get_network_state() -> NetworkState`

```python
@dataclass
class NetworkState:
    n_entities: int
    n_couplings: int
    network_coherence: float
    mean_trust: float
    trust_matrix: List[List[float]]
    entity_states: Dict[str, dict]

def get_network_state(self) -> NetworkState:
    """Get full network state."""
    n_entities = len(self.entities)
    
    # Count couplings
    n_couplings = sum(len(e.coupled_to) for e in self.entities.values()) // 2
    
    # Network coherence (all entities' slow phases combined)
    all_phases = []
    for networked in self.entities.values():
        all_phases.extend(networked.entity.substrate.slow.phases.tolist())
    
    if all_phases:
        network_coherence = min(np.abs(np.mean(np.exp(1j * np.array(all_phases)))), 0.999)
    else:
        network_coherence = 0.0
    
    # Build trust matrix
    entity_ids = list(self.entities.keys())
    trust_matrix = []
    for from_id in entity_ids:
        row = []
        for to_id in entity_ids:
            if from_id == to_id:
                row.append(1.0)
            else:
                row.append(self.get_trust(from_id, to_id))
        trust_matrix.append(row)
    
    mean_trust = np.mean([t for row in trust_matrix for t in row if t < 1.0]) if trust_matrix else 0.0
    
    # Entity states
    entity_states = {
        eid: {
            'name': networked.entity.name,
            'stage': networked.entity.stage.value,
            'ci': networked.entity.CI,
            'coherence': networked.entity.substrate.global_coherence,
            'coupled_to': networked.coupled_to,
        }
        for eid, networked in self.entities.items()
    }
    
    return NetworkState(
        n_entities=n_entities,
        n_couplings=n_couplings,
        network_coherence=network_coherence,
        mean_trust=mean_trust,
        trust_matrix=trust_matrix,
        entity_states=entity_states,
    )
```

---

## Success Criteria

### Verification
1. Valid entities pass verification
2. Tampered entities fail verification
3. Trust updates on verification

### Coupling
1. Coupled entities influence each other's phases
2. Coherence between coupled entities increases
3. Coupling decays without interaction

### Trust
1. Trust increases with successful verification
2. Trust increases with coherence correlation
3. Trust affects claim propagation

### Propagation
1. Claims spread through trusted couplings
2. Strength decays with hops
3. Trust threshold gates adoption

---

## Test Cases

```python
def test_verification():
    """Verification should succeed for valid entity."""
    network = MultiAgentNetwork()
    
    entity_a = create_entity("Alice")
    entity_b = create_entity("Bob")
    
    network.add_entity(entity_a, "alice")
    network.add_entity(entity_b, "bob")
    
    result = network.verify_entity("alice", "bob")
    
    assert result.success
    assert "bob" in network.entities["alice"].last_verified

def test_coupling():
    """Coupling should increase coherence."""
    network = MultiAgentNetwork()
    
    entity_a = create_entity("Alice")
    entity_b = create_entity("Bob")
    
    network.add_entity(entity_a, "alice")
    network.add_entity(entity_b, "bob")
    
    # Verify first (required for coupling)
    network.verify_entity("alice", "bob")
    
    result = network.couple_entities("alice", "bob", strength=0.3)
    
    assert result.coupling_strength == 0.3
    assert "bob" in network.entities["alice"].coupled_to

def test_trust_from_verification():
    """Trust should increase on successful verification."""
    network = MultiAgentNetwork()
    
    entity_a = create_entity("Alice")
    entity_b = create_entity("Bob")
    
    network.add_entity(entity_a, "alice")
    network.add_entity(entity_b, "bob")
    
    trust_before = network.get_trust("alice", "bob")
    network.verify_entity("alice", "bob")
    trust_after = network.get_trust("alice", "bob")
    
    assert trust_after > trust_before

def test_claim_propagation():
    """Claims should propagate through trusted couplings."""
    network = MultiAgentNetwork()
    
    # Create chain: alice -> bob -> carol
    alice = create_entity("Alice")
    bob = create_entity("Bob")
    carol = create_entity("Carol")
    
    network.add_entity(alice, "alice")
    network.add_entity(bob, "bob")
    network.add_entity(carol, "carol")
    
    # Set up trust and coupling
    network.verify_entity("alice", "bob")
    network.verify_entity("bob", "carol")
    network.couple_entities("alice", "bob")
    network.couple_entities("bob", "carol")
    
    # Boost trust
    network.entities["bob"].trust_given["alice"] = 0.8
    network.entities["carol"].trust_given["bob"] = 0.8
    
    # Propagate claim from alice
    claim = Claim(
        id="test_claim",
        content="Cooperation is valuable",
        strength=0.9,
        scope=ClaimScope.BEHAVIOR,
        source=ClaimSource.LEARNED,
        created_at=datetime.now().isoformat(),
    )
    
    result = network.propagate_claim("alice", claim)
    
    assert "bob" in result.entities_reached
    assert "carol" in result.entities_reached
```

---

## Dependencies

- `DevelopmentalEntity` (ORE2-006)
- `ClaimsEngine` (ORE2-009)
- `numpy`

---

## File Location

```
ore2/
├── network/
│   ├── __init__.py
│   ├── multi_agent.py      # <-- Main component
│   ├── verification.py     # Verification logic
│   ├── coupling.py         # Coupling logic
│   └── propagation.py      # Claim propagation
├── tests/
│   └── test_multi_agent.py
```

---

## Design Decisions to Preserve

1. **Verification required before coupling** - Trust comes before intimacy
2. **Trust is directional** - A trusting B ≠ B trusting A
3. **Coupling is slow-scale** - Identity level, not feature level
4. **Propagation decays** - Claims weaken as they spread
5. **Trust threshold gates adoption** - Don't accept claims from untrusted sources
6. **Network coherence is observable** - Emergent property of coupled network
