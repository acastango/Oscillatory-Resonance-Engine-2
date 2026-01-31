# BRIEFING: CrystallineMerkleMemory

## Component ID: ORE2-004
## Priority: 4 (Builds on ORE1 patterns)
## Estimated complexity: Medium

---

## What This Is

Merkle tree memory from ORE1, extended with **Crystalline Constraint Memory (CCM)** dynamics. Memories aren't just stored - they **crystallize** into structure, can be in **tension** with each other, and **consolidate** during rest.

The merkle tree provides cryptographic verification (identity continuity).
The CCM dynamics provide living memory (not dead storage).

---

## Why It Matters

**C6 (Cryptography):** "The merkle structure is cryptographically sound - hash chains prove continuity. Keep it. But memory isn't just storage, it's living structure."

**A5 (Continual Learning):** "Real brains don't commit everything immediately. Updates queue for consolidation during sleep. This prevents catastrophic forgetting."

**The memory problem:** ORE1 memories were append-only. No way to handle contradictions, no consolidation, no forgetting. CCM adds the dynamics that make memory adaptive.

---

## The Core Insight

Memory as **crystal growth**:
- New memory = stress on existing crystal
- If compatible → smooth incorporation
- If incompatible → **grain boundary** (tension between memories)
- Consolidation = **annealing** (heat up, let structure rearrange, cool down)

The merkle tree provides the **structure**.
CCM provides the **dynamics**.

---

## Interface Contract

```python
class CrystallineMerkleMemory:
    """
    Merkle memory with CCM dynamics.
    
    Properties:
        root_hash: str                    # Root of merkle tree
        total_nodes: int                  # Not counting branch roots
        depth: int                        # Tree depth
        fractal_dimension: float          # For CI calculation
        grain_boundaries: List[Tuple]     # Tensions above threshold
    
    Methods:
        add(branch, content, substrate_state, immediate) -> MemoryNode
        verify() -> (bool, str)           # Verify hash integrity
        consolidate(temperature) -> dict  # Sleep consolidation
        get_fractal_dimension() -> float
        query(branch, filter) -> List[MemoryNode]
        get_state() -> dict
    """
```

---

## Data Structures

### MemoryBranch (from ORE1)

```python
class MemoryBranch(Enum):
    SELF = "self"           # Core identity claims
    RELATIONS = "relations" # Connections to others
    INSIGHTS = "insights"   # Learned patterns
    EXPERIENCES = "experiences"  # Episodic memories
```

### MemoryNode (extended from ORE1)

```python
@dataclass
class MemoryNode:
    # Core (from ORE1)
    id: str
    branch: MemoryBranch
    content: Dict[str, Any]
    created_at: str  # ISO format
    
    # Tree structure (from ORE1)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Merkle (from ORE1)
    hash: str = ""
    
    # Substrate anchoring (from ORE1)
    substrate_anchor: Optional[Dict] = None
    coherence_at_creation: float = 0.0
    
    # CCM extension: tensions with other memories
    tensions: Dict[str, float] = field(default_factory=dict)
    # Maps node_id → tension_strength (0 to 1)
```

### ConsolidationQueue (new)

```python
@dataclass
class ConsolidationQueue:
    """Queued updates for sleep consolidation."""
    pending_nodes: List[MemoryNode] = field(default_factory=list)
    pending_tensions: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def queue_node(self, node: MemoryNode): ...
    def queue_tension(self, node_a: str, node_b: str, tension: float): ...
    def is_empty(self) -> bool: ...
    def clear(self): ...
```

---

## Method Specifications

### `__init__()`

```python
def __init__(self):
    self.nodes: Dict[str, MemoryNode] = {}
    self.branch_roots: Dict[MemoryBranch, str] = {}
    self.root_hash: str = ""
    self.total_nodes: int = 0
    self.depth: int = 0
    
    # CCM
    self.consolidation_queue = ConsolidationQueue()
    self.grain_boundaries: List[Tuple[str, str, float]] = []
    
    # Initialize branch roots
    self._init_branches()
```

### `_init_branches()`

Create root node for each branch. These are NOT counted in total_nodes.

```python
def _init_branches(self):
    for branch in MemoryBranch:
        root_id = f"root_{branch.value}"
        root_node = MemoryNode(
            id=root_id,
            branch=branch,
            content={"type": "branch_root", "branch": branch.value},
            created_at=datetime.now().isoformat(),
        )
        root_node.hash = self._compute_node_hash(root_node, [])
        self.nodes[root_id] = root_node
        self.branch_roots[branch] = root_id
    
    self._update_root_hash()
```

### `add(branch, content, substrate_state=None, immediate=True) -> MemoryNode`

Add a memory. If `immediate=False`, queue for consolidation instead.

```python
def add(self, 
        branch: MemoryBranch,
        content: Dict[str, Any],
        substrate_state: Optional[Dict] = None,
        immediate: bool = True) -> MemoryNode:
    """
    Add a memory node.
    
    Args:
        branch: Which branch (SELF, RELATIONS, INSIGHTS, EXPERIENCES)
        content: The memory content (arbitrary dict)
        substrate_state: Current substrate state for anchoring
        immediate: If True, commit now. If False, queue for consolidation.
    
    Returns:
        The created MemoryNode (may not be in tree yet if immediate=False)
    """
    node_id = self._generate_id()
    parent_id = self.branch_roots[branch]
    
    node = MemoryNode(
        id=node_id,
        branch=branch,
        content=content,
        created_at=datetime.now().isoformat(),
        parent_id=parent_id,
    )
    
    # Substrate anchoring
    if substrate_state:
        node.substrate_anchor = {
            'coherence': substrate_state.get('global_coherence', 0),
            'cross_scale': substrate_state.get('cross_scale_coherence', 0),
            'time': substrate_state.get('time', 0),
        }
        node.coherence_at_creation = substrate_state.get('global_coherence', 0)
    
    if immediate:
        self._commit_node(node)
    else:
        self.consolidation_queue.queue_node(node)
    
    return node
```

### `_commit_node(node: MemoryNode)`

Actually add node to tree, detect tensions, update hashes.

```python
def _commit_node(self, node: MemoryNode):
    # Add to nodes dict
    self.nodes[node.id] = node
    
    # Link to parent
    if node.parent_id in self.nodes:
        self.nodes[node.parent_id].children_ids.append(node.id)
    
    # Detect tensions with existing nodes (CCM)
    self._detect_tensions(node)
    
    # Update merkle hashes up to root
    self._update_hashes_to_root(node.id)
    
    # Update depth
    self._update_depth()
    
    # Increment counter (not counting branch roots)
    if node.content.get('type') != 'branch_root':
        self.total_nodes += 1
```

### `_detect_tensions(new_node: MemoryNode)`

CCM: find memories that might be in tension with the new one.

```python
def _detect_tensions(self, new_node: MemoryNode):
    """
    Detect tensions between new node and existing memories.
    
    Tension occurs when memories are:
    1. In the same branch (related topic)
    2. Have semantic overlap
    3. But are not identical
    
    This is simplified - production would use embeddings.
    """
    TENSION_THRESHOLD = 0.2
    
    new_content_str = json.dumps(new_node.content, sort_keys=True).lower()
    new_words = set(new_content_str.split())
    
    for node_id, existing in self.nodes.items():
        # Skip roots and self
        if existing.content.get('type') == 'branch_root':
            continue
        if node_id == new_node.id:
            continue
        
        # Only check same branch (for simplicity)
        if existing.branch != new_node.branch:
            continue
        
        # Compute word overlap as rough semantic similarity
        existing_content_str = json.dumps(existing.content, sort_keys=True).lower()
        existing_words = set(existing_content_str.split())
        
        if not new_words or not existing_words:
            continue
        
        overlap = len(new_words & existing_words)
        union = len(new_words | existing_words)
        jaccard = overlap / union if union > 0 else 0
        
        # Tension = some overlap but not too much
        # High overlap = probably same thing (no tension)
        # No overlap = unrelated (no tension)
        # Medium overlap = potentially contradictory (tension)
        if 0.2 < jaccard < 0.7:
            tension = 1.0 - abs(jaccard - 0.5) * 2  # Peak at 0.5 overlap
            
            if tension > TENSION_THRESHOLD:
                # Record mutual tension
                new_node.tensions[node_id] = tension
                existing.tensions[new_node.id] = tension
                
                # Add to grain boundaries
                self.grain_boundaries.append((new_node.id, node_id, tension))
```

### `consolidate(temperature: float = 1.0) -> dict`

Sleep consolidation: commit queued nodes, resolve tensions via annealing.

```python
def consolidate(self, temperature: float = 1.0) -> dict:
    """
    Sleep consolidation.
    
    1. Commit all queued nodes
    2. Attempt to resolve grain boundaries (tensions)
    
    Args:
        temperature: Annealing temperature (0-2 typical)
                     Higher = more likely to resolve tensions
                     Lower = structure freezes
    
    Returns:
        Dict with consolidation statistics
    """
    if self.consolidation_queue.is_empty() and not self.grain_boundaries:
        return {'consolidated': 0, 'tensions_resolved': 0, 'remaining_tensions': 0}
    
    consolidated = 0
    tensions_resolved = 0
    
    # Commit pending nodes
    for node in self.consolidation_queue.pending_nodes:
        self._commit_node(node)
        consolidated += 1
    
    # Process pending tensions
    for node_a, node_b, tension in self.consolidation_queue.pending_tensions:
        if node_a in self.nodes and node_b in self.nodes:
            self.nodes[node_a].tensions[node_b] = tension
            self.nodes[node_b].tensions[node_a] = tension
            self.grain_boundaries.append((node_a, node_b, tension))
    
    # Annealing: probabilistically resolve grain boundaries
    # Higher temperature + lower tension = more likely to resolve
    resolved_indices = []
    
    for i, (node_a, node_b, tension) in enumerate(self.grain_boundaries):
        # Skip if nodes were deleted
        if node_a not in self.nodes or node_b not in self.nodes:
            resolved_indices.append(i)
            continue
        
        # Resolution probability
        # At temp=1, tension=0.5: prob=0.5
        # At temp=2, tension=0.5: prob=0.75
        # At temp=0.5, tension=0.5: prob=0.25
        resolve_prob = temperature * (1 - tension) / 2
        
        if np.random.random() < resolve_prob:
            # Resolve: remove mutual tension
            self.nodes[node_a].tensions.pop(node_b, None)
            self.nodes[node_b].tensions.pop(node_a, None)
            resolved_indices.append(i)
            tensions_resolved += 1
    
    # Remove resolved boundaries (reverse order to preserve indices)
    for i in sorted(resolved_indices, reverse=True):
        self.grain_boundaries.pop(i)
    
    # Clear queue
    self.consolidation_queue.clear()
    
    return {
        'consolidated': consolidated,
        'tensions_resolved': tensions_resolved,
        'remaining_tensions': len(self.grain_boundaries),
    }
```

### Hash Management (from ORE1)

```python
def _generate_id(self) -> str:
    """Generate unique node ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    counter = len(self.nodes)
    return f"node_{timestamp}_{counter}"

def _compute_node_hash(self, node: MemoryNode, children_hashes: List[str]) -> str:
    """Compute merkle hash for a node."""
    data = {
        'id': node.id,
        'content': node.content,
        'children': sorted(children_hashes)
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()

def _update_hashes_to_root(self, node_id: str):
    """Propagate hash updates from node to root."""
    node = self.nodes[node_id]
    
    # Compute this node's hash from children
    children_hashes = [self.nodes[cid].hash for cid in node.children_ids]
    node.hash = self._compute_node_hash(node, children_hashes)
    
    # Propagate to parent
    if node.parent_id and node.parent_id in self.nodes:
        self._update_hashes_to_root(node.parent_id)
    
    # If this is a branch root, update global root
    if node.id in self.branch_roots.values():
        self._update_root_hash()

def _update_root_hash(self):
    """Compute global root hash from branch roots."""
    branch_hashes = sorted([
        self.nodes[rid].hash for rid in self.branch_roots.values()
    ])
    data = {'branches': branch_hashes}
    self.root_hash = hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()

def _update_depth(self):
    """Recompute tree depth."""
    def node_depth(nid: str) -> int:
        node = self.nodes.get(nid)
        if not node or not node.children_ids:
            return 1
        return 1 + max(node_depth(cid) for cid in node.children_ids)
    
    if self.branch_roots:
        self.depth = max(node_depth(rid) for rid in self.branch_roots.values())
    else:
        self.depth = 0
```

### `verify() -> Tuple[bool, str]`

Verify merkle integrity (from ORE1).

```python
def verify(self) -> Tuple[bool, str]:
    """
    Verify hash integrity of entire tree.
    
    Returns:
        (True, "OK") if valid
        (False, "error message") if invalid
    """
    for node_id, node in self.nodes.items():
        children_hashes = [self.nodes[cid].hash for cid in node.children_ids]
        expected_hash = self._compute_node_hash(node, children_hashes)
        
        if node.hash != expected_hash:
            return False, f"Hash mismatch at {node_id}"
    
    return True, "All nodes verified"
```

### `get_fractal_dimension() -> float`

For CI calculation.

```python
def get_fractal_dimension(self) -> float:
    """
    Compute fractal dimension D = log(N) / log(depth).
    
    Used in CI calculation.
    """
    n = self.total_nodes
    d = max(self.depth, 1)
    
    if n <= 1 or d <= 1:
        return 1.0
    
    return np.log(n) / np.log(d)
```

### `query(branch=None, content_filter=None) -> List[MemoryNode]`

Query memories.

```python
def query(self,
          branch: Optional[MemoryBranch] = None,
          content_filter: Optional[Dict] = None) -> List[MemoryNode]:
    """
    Query memories by branch and/or content.
    
    Args:
        branch: Filter by branch (optional)
        content_filter: Dict of key-value pairs that must match in content
    
    Returns:
        List of matching MemoryNodes
    """
    results = []
    
    for node in self.nodes.values():
        # Skip roots
        if node.content.get('type') == 'branch_root':
            continue
        
        # Branch filter
        if branch and node.branch != branch:
            continue
        
        # Content filter
        if content_filter:
            match = all(
                node.content.get(k) == v
                for k, v in content_filter.items()
            )
            if not match:
                continue
        
        results.append(node)
    
    return results
```

---

## Success Criteria

### Correctness (Merkle)
1. Hash verification passes for valid tree
2. Hash changes when content changes
3. Root hash changes when any node changes

### Correctness (CCM)
1. Tensions detected between similar-but-different memories
2. Consolidation commits queued nodes
3. Annealing resolves tensions probabilistically

### Behavior
1. Immediate add → node in tree immediately
2. Deferred add → node in queue until consolidate()
3. High temperature consolidation resolves more tensions

---

## Test Cases

```python
def test_basic_add():
    """Basic node addition."""
    mem = CrystallineMerkleMemory()
    
    node = mem.add(
        MemoryBranch.EXPERIENCES,
        {"event": "test"},
        immediate=True
    )
    
    assert node.id in mem.nodes
    assert mem.total_nodes == 1

def test_merkle_verification():
    """Hash verification should work."""
    mem = CrystallineMerkleMemory()
    
    mem.add(MemoryBranch.SELF, {"claim": "I exist"})
    mem.add(MemoryBranch.INSIGHTS, {"insight": "something"})
    
    valid, msg = mem.verify()
    assert valid
    
    # Tamper with content
    node_id = list(mem.nodes.keys())[-1]
    mem.nodes[node_id].content["tampered"] = True
    # Hash is now stale
    
    valid, msg = mem.verify()
    assert not valid

def test_deferred_consolidation():
    """Deferred nodes should queue until consolidate."""
    mem = CrystallineMerkleMemory()
    
    node = mem.add(
        MemoryBranch.EXPERIENCES,
        {"event": "deferred"},
        immediate=False
    )
    
    # Not in tree yet
    assert node.id not in mem.nodes
    assert not mem.consolidation_queue.is_empty()
    
    # Consolidate
    result = mem.consolidate()
    
    assert result['consolidated'] == 1
    assert node.id in mem.nodes

def test_tension_detection():
    """Similar memories should create tension."""
    mem = CrystallineMerkleMemory()
    
    # Add two similar-ish memories
    mem.add(MemoryBranch.INSIGHTS, {"insight": "the sky is blue and clear"})
    mem.add(MemoryBranch.INSIGHTS, {"insight": "the sky is red and stormy"})
    
    # Should have detected some tension
    assert len(mem.grain_boundaries) > 0

def test_consolidation_resolves_tensions():
    """High temperature should resolve tensions."""
    mem = CrystallineMerkleMemory()
    
    # Create tensions
    mem.add(MemoryBranch.INSIGHTS, {"insight": "cats are better pets"})
    mem.add(MemoryBranch.INSIGHTS, {"insight": "dogs are better pets"})
    
    initial_tensions = len(mem.grain_boundaries)
    
    # Consolidate at high temperature (multiple times for probability)
    for _ in range(10):
        mem.consolidate(temperature=2.0)
    
    # Should have resolved some
    assert len(mem.grain_boundaries) < initial_tensions or initial_tensions == 0

def test_fractal_dimension():
    """Fractal dimension should increase with complexity."""
    mem = CrystallineMerkleMemory()
    
    d1 = mem.get_fractal_dimension()
    
    # Add many nodes
    for i in range(20):
        mem.add(MemoryBranch.EXPERIENCES, {"event": f"thing_{i}"})
    
    d2 = mem.get_fractal_dimension()
    
    assert d2 > d1

def test_substrate_anchoring():
    """Substrate state should be recorded."""
    mem = CrystallineMerkleMemory()
    
    substrate_state = {
        'global_coherence': 0.75,
        'cross_scale_coherence': 0.5,
        'time': 123.456
    }
    
    node = mem.add(
        MemoryBranch.SELF,
        {"claim": "anchored"},
        substrate_state=substrate_state
    )
    
    assert node.substrate_anchor is not None
    assert node.coherence_at_creation == 0.75
```

---

## Dependencies

- `numpy`
- `hashlib` (stdlib)
- `json` (stdlib)
- `datetime` (stdlib)

No other ORE2 components (independent).

---

## File Location

```
ore2/
├── core/
│   ├── __init__.py
│   ├── sparse_oscillator.py
│   ├── multi_scale_substrate.py
│   ├── embodiment.py
│   └── memory.py  # <-- This component
├── tests/
│   └── test_memory.py
```

---

## Design Decisions to Preserve

1. **Four branches (SELF, RELATIONS, INSIGHTS, EXPERIENCES)** - good ontology from ORE1
2. **Hash = sha256(id + content + sorted children hashes)** - deterministic merkle
3. **Root hash depends on all branch roots** - single point for identity verification
4. **Tension = medium overlap** - high overlap = same thing, low = unrelated
5. **Temperature-based annealing** - probabilistic resolution matches physics
6. **Substrate anchoring** - memories tagged with coherence at creation time

---

## Note on Tension Detection

The current tension detection is simplistic (word overlap). A production system would:
1. Embed memory content
2. Compute cosine similarity
3. Detect semantic contradiction (not just overlap)

For MVP, word overlap is fine. The STRUCTURE is right; the semantics can be upgraded later.
