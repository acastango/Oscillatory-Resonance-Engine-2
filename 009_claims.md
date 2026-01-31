# BRIEFING: ClaimsEngine

## Component ID: ORE2-009
## Priority: High (Makes architecture injectable)
## Estimated complexity: Medium-High

---

## What This Is

A system for **injectable knowledge** - declarative statements that become **operational** in the substrate. Claims are how an entity knows things, adopts roles, and maintains beliefs.

The key insight from COGNIZEN: **architecture is injectable data**. A claim like "I am a helpful assistant" isn't just stored - it modifies coupling weights, creates attractors, and biases dynamics.

---

## Why It Matters

**N4 (Computational Neuro):** "Claims are the bridge between declarative and operational knowledge. Saying 'I am cautious' should actually make the system more cautious - not just remember that it said so."

**A3 (ML Integration):** "This is how we inject roles into ORE. The COGNIZEN 5-role pattern (Analyst, Creative, Skeptic, Integrator, Meta) becomes claims that shape dynamics."

**P1 (Dynamical Systems):** "Claims modify the energy landscape. A strong claim creates an attractor basin. The substrate *wants* to be consistent with its claims."

**C6 (Cryptography):** "Claims should be merkle-anchored. Auditable belief history."

---

## The Core Insight

Claims have three aspects:

1. **Content** - What the claim asserts (text/embedding)
2. **Strength** - How confidently it's held (0 to 1)
3. **Scope** - What it affects (substrate regions, behaviors)

When a claim is **activated**, it:
- Generates a phase pattern (via SemanticGrounding)
- Modifies coupling weights toward that pattern
- Creates an attractor that pulls dynamics toward consistency

```
CLAIM: "I am a careful analyst"
        │
        ▼
    [Embedding] ──► [Phase Pattern]
        │                  │
        ▼                  ▼
    [Stored in         [Modifies substrate
     SELF branch]       coupling weights]
        │                  │
        ▼                  ▼
    [Merkle anchor]    [Attractor basin
                        toward "careful"]
```

---

## Interface Contract

```python
class ClaimsEngine:
    """
    Manages claims that shape entity dynamics.
    
    Properties:
        claims: Dict[str, Claim]         # All claims by ID
        active_claims: List[str]         # Currently active claim IDs
        claim_coherence: float           # How consistent claims are
    
    Methods:
        # Claim management
        add_claim(content, strength, scope, source) -> Claim
        remove_claim(claim_id)
        update_strength(claim_id, new_strength)
        
        # Activation
        activate_claim(claim_id)
        deactivate_claim(claim_id)
        activate_role(role_name)  # Activates claim set
        
        # Substrate integration
        apply_to_substrate(substrate)  # Modify couplings
        measure_consistency(substrate) -> float  # How aligned is substrate
        
        # Memory integration
        anchor_to_memory(memory)  # Store in merkle tree
        
        # Queries
        get_claims_by_scope(scope) -> List[Claim]
        get_conflicting_claims() -> List[Tuple[Claim, Claim]]
"""

@dataclass
class Claim:
    """A single claim that can shape dynamics."""
    id: str
    content: str                    # Natural language content
    strength: float                 # 0-1, confidence/importance
    scope: ClaimScope               # What it affects
    source: ClaimSource             # Where it came from
    
    # Grounding
    embedding: Optional[np.ndarray] = None
    phase_pattern: Optional[PhasePair] = None
    
    # State
    active: bool = False
    created_at: str = ""
    activated_at: Optional[str] = None
    
    # Memory anchoring
    memory_node_id: Optional[str] = None
    coherence_at_creation: float = 0.0

class ClaimScope(Enum):
    """What a claim affects."""
    IDENTITY = "identity"       # Core self-conception
    BEHAVIOR = "behavior"       # Action tendencies  
    KNOWLEDGE = "knowledge"     # Factual beliefs
    RELATION = "relation"       # Beliefs about others
    GOAL = "goal"               # Objectives/drives
    CONSTRAINT = "constraint"   # Limitations/rules

class ClaimSource(Enum):
    """Where a claim originated."""
    INNATE = "innate"           # Built-in (rare in developmental model)
    LEARNED = "learned"         # From experience
    INSTRUCTED = "instructed"   # Explicitly told
    INFERRED = "inferred"       # Derived from other claims
    SOCIAL = "social"           # From other entities
```

---

## Configuration

```python
@dataclass
class ClaimsEngineConfig:
    # Grounding
    grounding: Optional[SemanticGrounding] = None
    
    # Strength dynamics
    strength_decay: float = 0.001      # Per tick when inactive
    strength_boost: float = 0.01       # Per tick when consistent
    min_strength: float = 0.1          # Below this, claim is removed
    
    # Coupling modification
    coupling_scale: float = 0.3        # How much claims affect weights
    
    # Consistency
    consistency_threshold: float = 0.6  # Below = conflict
    
    # Limits
    max_active_claims: int = 10        # Cognitive load limit
    max_total_claims: int = 100
```

---

## Method Specifications

### `__init__(config, grounding)`

```python
def __init__(self, 
             config: Optional[ClaimsEngineConfig] = None,
             grounding: Optional[SemanticGrounding] = None):
    self.config = config or ClaimsEngineConfig()
    self.grounding = grounding or self.config.grounding
    
    if self.grounding is None:
        raise ValueError("ClaimsEngine requires SemanticGrounding")
    
    self.claims: Dict[str, Claim] = {}
    self.active_claims: List[str] = []
    
    # Role templates (COGNIZEN-style)
    self._role_templates = self._init_role_templates()

def _init_role_templates(self) -> Dict[str, List[str]]:
    """Initialize standard role claim sets."""
    return {
        'analyst': [
            "I examine problems systematically and thoroughly",
            "I seek evidence before drawing conclusions",
            "I notice patterns and inconsistencies",
        ],
        'creative': [
            "I generate novel ideas and perspectives",
            "I make unexpected connections between concepts",
            "I value originality and experimentation",
        ],
        'skeptic': [
            "I question assumptions and challenge claims",
            "I look for flaws and edge cases",
            "I resist premature conclusions",
        ],
        'integrator': [
            "I synthesize diverse viewpoints into coherent wholes",
            "I find common ground between conflicting ideas",
            "I build bridges between perspectives",
        ],
        'meta': [
            "I observe and reflect on my own thinking processes",
            "I monitor the quality and coherence of my reasoning",
            "I adjust my approach based on self-observation",
        ],
    }
```

### `add_claim(content, strength, scope, source) -> Claim`

```python
def add_claim(self,
              content: str,
              strength: float = 0.5,
              scope: ClaimScope = ClaimScope.KNOWLEDGE,
              source: ClaimSource = ClaimSource.LEARNED) -> Claim:
    """
    Add a new claim.
    
    Args:
        content: Natural language claim content
        strength: Initial strength (0-1)
        scope: What the claim affects
        source: Where it came from
    
    Returns:
        The created Claim
    """
    # Check limits
    if len(self.claims) >= self.config.max_total_claims:
        # Remove weakest inactive claim
        self._prune_weakest()
    
    # Generate ID
    claim_id = f"claim_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{len(self.claims)}"
    
    # Ground the claim
    embedding = self.grounding.embedder(content) if self.grounding.embedder else None
    phase_pattern = self.grounding.embed_to_phases(embedding) if embedding is not None else None
    
    claim = Claim(
        id=claim_id,
        content=content,
        strength=np.clip(strength, 0, 1),
        scope=scope,
        source=source,
        embedding=embedding,
        phase_pattern=phase_pattern,
        created_at=datetime.now().isoformat(),
    )
    
    self.claims[claim_id] = claim
    return claim

def _prune_weakest(self):
    """Remove the weakest inactive claim."""
    inactive = [c for c in self.claims.values() if not c.active]
    if inactive:
        weakest = min(inactive, key=lambda c: c.strength)
        del self.claims[weakest.id]
```

### `activate_claim(claim_id)` / `deactivate_claim(claim_id)`

```python
def activate_claim(self, claim_id: str):
    """
    Activate a claim, making it influence dynamics.
    """
    if claim_id not in self.claims:
        raise KeyError(f"Unknown claim: {claim_id}")
    
    claim = self.claims[claim_id]
    
    if claim.active:
        return  # Already active
    
    # Check capacity
    if len(self.active_claims) >= self.config.max_active_claims:
        # Deactivate weakest active claim
        weakest_id = min(self.active_claims, 
                         key=lambda cid: self.claims[cid].strength)
        self.deactivate_claim(weakest_id)
    
    claim.active = True
    claim.activated_at = datetime.now().isoformat()
    self.active_claims.append(claim_id)

def deactivate_claim(self, claim_id: str):
    """Deactivate a claim."""
    if claim_id not in self.claims:
        return
    
    claim = self.claims[claim_id]
    claim.active = False
    claim.activated_at = None
    
    if claim_id in self.active_claims:
        self.active_claims.remove(claim_id)
```

### `activate_role(role_name)`

```python
def activate_role(self, role_name: str):
    """
    Activate a predefined role (set of claims).
    
    COGNIZEN roles: analyst, creative, skeptic, integrator, meta
    """
    if role_name not in self._role_templates:
        raise ValueError(f"Unknown role: {role_name}. "
                        f"Available: {list(self._role_templates.keys())}")
    
    # Add and activate role claims
    for content in self._role_templates[role_name]:
        # Check if similar claim exists
        existing = self._find_similar_claim(content)
        
        if existing:
            # Boost existing claim
            existing.strength = min(1.0, existing.strength + 0.2)
            self.activate_claim(existing.id)
        else:
            # Create new claim
            claim = self.add_claim(
                content=content,
                strength=0.7,
                scope=ClaimScope.BEHAVIOR,
                source=ClaimSource.INSTRUCTED,
            )
            self.activate_claim(claim.id)

def _find_similar_claim(self, content: str, threshold: float = 0.85) -> Optional[Claim]:
    """Find existing claim similar to content."""
    if self.grounding.embedder is None:
        return None
    
    new_embedding = self.grounding.embedder(content)
    
    for claim in self.claims.values():
        if claim.embedding is not None:
            similarity = np.dot(new_embedding, claim.embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(claim.embedding)
            )
            if similarity > threshold:
                return claim
    
    return None
```

### `apply_to_substrate(substrate)`

**The key method**: Make claims operational by modifying substrate.

```python
def apply_to_substrate(self, substrate: 'MultiScaleSubstrate'):
    """
    Apply active claims to substrate dynamics.
    
    This modifies coupling weights to create attractors
    toward claim-consistent states.
    """
    if not self.active_claims:
        return
    
    cfg = self.config
    
    for claim_id in self.active_claims:
        claim = self.claims[claim_id]
        
        if claim.phase_pattern is None:
            continue
        
        # Claim creates an attractor toward its phase pattern
        # We modify internal weights to favor this pattern
        
        # For fast scale
        self._apply_claim_to_layer(
            substrate.fast,
            claim.phase_pattern.fast,
            claim.strength * cfg.coupling_scale
        )
        
        # For slow scale (stronger effect - identity level)
        self._apply_claim_to_layer(
            substrate.slow,
            claim.phase_pattern.slow,
            claim.strength * cfg.coupling_scale * 1.5
        )

def _apply_claim_to_layer(self, 
                          layer: 'SparseOscillatorLayer',
                          target_phases: np.ndarray,
                          strength: float):
    """
    Modify layer coupling to favor target phase pattern.
    
    The idea: oscillators that should be in-phase get positive coupling,
    oscillators that should be anti-phase get negative coupling.
    """
    n = layer.n
    
    # Compute pairwise target phase differences
    target_diff = target_phases[:, np.newaxis] - target_phases[np.newaxis, :]
    
    # Coupling modification: positive for in-phase targets, negative for anti-phase
    # This creates an attractor toward the target pattern
    coupling_mod = strength * np.cos(target_diff)
    
    # Apply modification (additive, preserving existing structure)
    # Only modify, don't replace
    layer.internal_weights += coupling_mod / n
    
    # Also stimulate oscillators aligned with target
    # This activates the claim's pattern
    current_alignment = np.cos(layer.phases - target_phases)
    aligned_mask = current_alignment > 0.5
    
    if np.any(aligned_mask):
        indices = np.where(aligned_mask)[0]
        strengths = strength * current_alignment[aligned_mask]
        layer.stimulate(indices, strengths)
```

### `measure_consistency(substrate) -> float`

```python
def measure_consistency(self, substrate: 'MultiScaleSubstrate') -> float:
    """
    Measure how consistent substrate state is with active claims.
    
    Returns 0-1 where 1 = perfectly consistent.
    """
    if not self.active_claims:
        return 1.0  # No claims = vacuously consistent
    
    consistencies = []
    
    for claim_id in self.active_claims:
        claim = self.claims[claim_id]
        
        if claim.phase_pattern is None:
            continue
        
        # Measure phase alignment with claim pattern
        # Fast scale
        fast_alignment = np.mean(np.cos(
            substrate.fast.phases - claim.phase_pattern.fast
        ))
        fast_alignment = (fast_alignment + 1) / 2  # [0, 1]
        
        # Slow scale (weighted more)
        slow_alignment = np.mean(np.cos(
            substrate.slow.phases - claim.phase_pattern.slow
        ))
        slow_alignment = (slow_alignment + 1) / 2
        
        # Weight by claim strength
        consistency = (0.4 * fast_alignment + 0.6 * slow_alignment) * claim.strength
        consistencies.append(consistency)
    
    return np.mean(consistencies) if consistencies else 1.0
```

### `anchor_to_memory(memory)`

```python
def anchor_to_memory(self, 
                     memory: 'CrystallineMerkleMemory',
                     substrate_state: Optional[dict] = None):
    """
    Anchor all unanchored claims to memory.
    """
    for claim in self.claims.values():
        if claim.memory_node_id is not None:
            continue  # Already anchored
        
        # Store in SELF branch
        node = memory.add(
            MemoryBranch.SELF,
            {
                'type': 'claim',
                'claim_id': claim.id,
                'content': claim.content,
                'strength': claim.strength,
                'scope': claim.scope.value,
                'source': claim.source.value,
                'created_at': claim.created_at,
            },
            substrate_state=substrate_state,
            immediate=True,  # Claims are important
        )
        
        claim.memory_node_id = node.id
        claim.coherence_at_creation = node.coherence_at_creation
```

### `get_conflicting_claims() -> List[Tuple[Claim, Claim]]`

```python
def get_conflicting_claims(self) -> List[Tuple[Claim, Claim]]:
    """
    Find pairs of claims that are semantically inconsistent.
    
    Uses embedding similarity - very similar claims with
    opposite implications are conflicts.
    """
    conflicts = []
    claim_list = list(self.claims.values())
    
    for i in range(len(claim_list)):
        for j in range(i + 1, len(claim_list)):
            c1, c2 = claim_list[i], claim_list[j]
            
            if c1.embedding is None or c2.embedding is None:
                continue
            
            # Check if same scope (conflicts most relevant within scope)
            if c1.scope != c2.scope:
                continue
            
            # Compute phase similarity
            if c1.phase_pattern and c2.phase_pattern:
                similarity = self.grounding.phase_similarity(
                    c1.phase_pattern, c2.phase_pattern
                )
                
                # Medium similarity = potential conflict
                # (Very high = same claim, very low = unrelated)
                if 0.3 < similarity < 0.7:
                    # Check for negation patterns in content
                    if self._appears_contradictory(c1.content, c2.content):
                        conflicts.append((c1, c2))
    
    return conflicts

def _appears_contradictory(self, content1: str, content2: str) -> bool:
    """Simple heuristic for contradiction detection."""
    negations = ['not', "don't", "never", "avoid", "refuse", "cannot"]
    
    c1_lower = content1.lower()
    c2_lower = content2.lower()
    
    # One has negation, other doesn't, on similar topic
    c1_negated = any(neg in c1_lower for neg in negations)
    c2_negated = any(neg in c2_lower for neg in negations)
    
    return c1_negated != c2_negated
```

---

## Properties

```python
@property
def claim_coherence(self) -> float:
    """
    Internal coherence of active claims.
    High = claims are mutually consistent.
    Low = claims conflict with each other.
    """
    if len(self.active_claims) < 2:
        return 1.0
    
    # Pairwise phase coherence between active claims
    coherences = []
    
    for i, cid1 in enumerate(self.active_claims):
        for cid2 in self.active_claims[i+1:]:
            c1, c2 = self.claims[cid1], self.claims[cid2]
            
            if c1.phase_pattern and c2.phase_pattern:
                coh = self.grounding.phase_similarity(
                    c1.phase_pattern, c2.phase_pattern
                )
                coherences.append(coh)
    
    return np.mean(coherences) if coherences else 1.0
```

---

## Success Criteria

### Operational Effect
1. Active claims modify substrate coupling weights
2. Substrate consistency measurable and changes with claims
3. Activating "analyst" role biases toward analytical dynamics

### Coherence
1. Similar claims have similar phase patterns
2. Conflicting claims detected
3. claim_coherence reflects internal consistency

### Memory Integration
1. Claims anchored to SELF branch
2. Claim history auditable via merkle tree

---

## Test Cases

```python
def test_add_claim():
    """Claims should be created with proper grounding."""
    grounding = SemanticGrounding(embedder=mock_embedder)
    engine = ClaimsEngine(grounding=grounding)
    
    claim = engine.add_claim(
        "I value honesty",
        strength=0.8,
        scope=ClaimScope.IDENTITY,
    )
    
    assert claim.id in engine.claims
    assert claim.embedding is not None
    assert claim.phase_pattern is not None

def test_activate_deactivate():
    """Activation state should track correctly."""
    engine = create_test_engine()
    claim = engine.add_claim("test claim")
    
    assert not claim.active
    assert claim.id not in engine.active_claims
    
    engine.activate_claim(claim.id)
    
    assert claim.active
    assert claim.id in engine.active_claims
    
    engine.deactivate_claim(claim.id)
    
    assert not claim.active
    assert claim.id not in engine.active_claims

def test_max_active_claims():
    """Should enforce max active limit."""
    config = ClaimsEngineConfig(max_active_claims=3)
    engine = ClaimsEngine(config=config, grounding=create_test_grounding())
    
    claims = [engine.add_claim(f"claim {i}") for i in range(5)]
    for c in claims:
        engine.activate_claim(c.id)
    
    assert len(engine.active_claims) == 3

def test_apply_to_substrate():
    """Claims should modify substrate dynamics."""
    engine = create_test_engine()
    substrate = create_test_substrate()
    
    claim = engine.add_claim("I am focused", strength=0.9)
    engine.activate_claim(claim.id)
    
    # Measure substrate before
    weights_before = substrate.slow.internal_weights.copy()
    
    engine.apply_to_substrate(substrate)
    
    # Weights should have changed
    weights_after = substrate.slow.internal_weights
    assert not np.allclose(weights_before, weights_after)

def test_measure_consistency():
    """Consistency should reflect alignment."""
    engine = create_test_engine()
    substrate = create_test_substrate()
    
    claim = engine.add_claim("test")
    engine.activate_claim(claim.id)
    
    # Force substrate to match claim pattern
    substrate.slow.phases = claim.phase_pattern.slow.copy()
    
    consistency = engine.measure_consistency(substrate)
    assert consistency > 0.8

def test_activate_role():
    """Role activation should add multiple claims."""
    engine = create_test_engine()
    
    initial_count = len(engine.claims)
    engine.activate_role('analyst')
    
    assert len(engine.claims) > initial_count
    assert len(engine.active_claims) > 0

def test_anchor_to_memory():
    """Claims should be stored in memory."""
    engine = create_test_engine()
    memory = CrystallineMerkleMemory()
    
    claim = engine.add_claim("I remember this")
    engine.anchor_to_memory(memory)
    
    assert claim.memory_node_id is not None
    assert memory.total_nodes > 0
```

---

## Dependencies

- `SemanticGrounding` (ORE2-008)
- `MultiScaleSubstrate` (ORE2-002)
- `CrystallineMerkleMemory` (ORE2-004)
- `numpy`

---

## File Location

```
ore2/
├── core/
│   └── claims.py  # <-- This component
├── tests/
│   └── test_claims.py
```

---

## Design Decisions to Preserve

1. **Claims modify couplings, not just store data** - operational, not declarative
2. **Slow scale weighted 1.5x** - identity/behavior claims need stability
3. **Max 10 active claims** - cognitive load limit
4. **COGNIZEN role templates** - analyst, creative, skeptic, integrator, meta
5. **Merkle anchoring for auditability** - belief history is provable
6. **Strength decay when inactive** - beliefs fade without reinforcement

---

## COGNIZEN Integration Pattern

```python
# Full COGNIZEN 5-role activation
entity = create_entity("Thinker")
claims = ClaimsEngine(grounding=entity_grounding)

# Activate analytical mode
claims.activate_role('analyst')
claims.activate_role('skeptic')

# Apply to entity
claims.apply_to_substrate(entity.substrate)

# Entity now has bias toward:
# - Systematic examination
# - Evidence-seeking
# - Questioning assumptions
# - Looking for flaws
```

This is how LLM outputs can be "verified" by ORE - the claims create attractors, and generation is steered toward claim-consistent states.
