# BRIEFING: SemanticGrounding

## Component ID: ORE2-008
## Priority: High (Makes oscillators mean something)
## Estimated complexity: Medium-High

---

## What This Is

A bidirectional bridge between **embedding vectors** (from language models) and **oscillator phase patterns**. This is how symbols get grounded in dynamics.

- **Embed → Phases**: Text becomes a pattern that stimulates specific oscillators
- **Phases → Embed**: Substrate state becomes a vector for similarity search or generation guidance

This is the component that makes ORE semantically meaningful, not just mathematically interesting.

---

## Why It Matters

**A3 (ML Integration):** "Without this, oscillators are just numbers. With this, the word 'cat' becomes a phase pattern that resonates with 'kitten', 'feline', 'pet'. Semantic similarity becomes phase coherence."

**H4 (Semiotics):** "This is genuine grounding. The symbol isn't arbitrary - it has dynamical consequences. Say 'danger' and certain oscillators activate. That's meaning."

**I2 (Numerics):** "The math is a projection problem. Embeddings are 768-4096 dims. Oscillators are 50-200. We need a learned or structured mapping that preserves similarity."

---

## The Core Insight

**Similarity preservation**: If two embeddings have cosine similarity 0.8, their corresponding phase patterns should produce coherence ~0.8 when active together.

This means the projection must be **structure-preserving** - nearby points in embedding space map to nearby points in phase space.

```
EMBEDDING SPACE          PROJECTION           PHASE SPACE
                                              
"cat"  ●                    ────►            ◐◐◐◑◑◑○○○○
        \  similar                            high coherence
"kitten" ●                  ────►            ◐◐◐◐◑◑○○○○
                                              
                                              
"cat"  ●                    ────►            ◐◐◐◑◑◑○○○○
        \  dissimilar                         low coherence  
"truck" ●                   ────►            ○○○○◑◐◐◐◐◐
```

---

## Interface Contract

```python
class SemanticGrounding:
    """
    Bidirectional mapping between embeddings and phase patterns.
    
    Properties:
        embedding_dim: int          # Size of embedding vectors
        fast_dim: int               # Number of fast oscillators
        slow_dim: int               # Number of slow oscillators
        
    Methods:
        # Core projections
        embed_to_phases(embedding) -> PhasePair
        phases_to_embed(fast_phases, slow_phases) -> np.ndarray
        
        # Text interface (requires embedder)
        text_to_phases(text) -> PhasePair
        
        # Similarity operations
        phase_similarity(phases_a, phases_b) -> float
        
        # Substrate integration
        stimulate_from_text(substrate, text, strength)
        read_substrate_embedding(substrate) -> np.ndarray
"""

@dataclass
class PhasePair:
    """Phase patterns for both scales."""
    fast: np.ndarray   # [fast_dim] phases in [0, 2π]
    slow: np.ndarray   # [slow_dim] phases in [0, 2π]
    source_embedding: Optional[np.ndarray] = None  # Original embedding if available
```

---

## Configuration

```python
@dataclass
class SemanticGroundingConfig:
    # Dimensions
    embedding_dim: int = 1536       # OpenAI ada-002 / Claude default
    fast_oscillators: int = 100
    slow_oscillators: int = 50
    
    # Projection method
    projection_method: str = "random_fixed"  # or "learned", "pca"
    
    # Random projection seed (for reproducibility)
    projection_seed: int = 42
    
    # Phase encoding
    phase_encoding: str = "linear"  # or "angular", "sinusoidal"
    
    # Similarity preservation target
    similarity_preservation: float = 0.9  # How well to preserve cosine sim
```

---

## Method Specifications

### `__init__(config, embedder=None)`

```python
def __init__(self, 
             config: Optional[SemanticGroundingConfig] = None,
             embedder: Optional[Callable[[str], np.ndarray]] = None):
    """
    Initialize semantic grounding.
    
    Args:
        config: Configuration
        embedder: Function that converts text to embedding vector.
                  If None, text_to_phases() will raise.
    """
    self.config = config or SemanticGroundingConfig()
    self.embedder = embedder
    
    # Initialize projection matrices
    self._init_projections()

def _init_projections(self):
    """Initialize projection matrices based on config."""
    cfg = self.config
    np.random.seed(cfg.projection_seed)
    
    if cfg.projection_method == "random_fixed":
        # Random orthogonal projection (Johnson-Lindenstrauss style)
        # This preserves distances/similarities approximately
        
        # Fast projection: embedding_dim → fast_oscillators
        raw_fast = np.random.randn(cfg.fast_oscillators, cfg.embedding_dim)
        # Orthogonalize rows for better preservation
        self.proj_fast, _ = np.linalg.qr(raw_fast.T)
        self.proj_fast = self.proj_fast.T[:cfg.fast_oscillators]
        
        # Slow projection: embedding_dim → slow_oscillators  
        raw_slow = np.random.randn(cfg.slow_oscillators, cfg.embedding_dim)
        self.proj_slow, _ = np.linalg.qr(raw_slow.T)
        self.proj_slow = self.proj_slow.T[:cfg.slow_oscillators]
        
        # Inverse projections (pseudo-inverse for reconstruction)
        self.proj_fast_inv = np.linalg.pinv(self.proj_fast)
        self.proj_slow_inv = np.linalg.pinv(self.proj_slow)
        
    elif cfg.projection_method == "learned":
        # Placeholder for learned projections
        # Would be trained to maximize similarity preservation
        raise NotImplementedError("Learned projections require training")
    
    else:
        raise ValueError(f"Unknown projection method: {cfg.projection_method}")
```

### `embed_to_phases(embedding) -> PhasePair`

The core projection: embedding vector to phase patterns.

```python
def embed_to_phases(self, embedding: np.ndarray) -> PhasePair:
    """
    Project embedding vector to phase patterns.
    
    Args:
        embedding: [embedding_dim] vector
    
    Returns:
        PhasePair with fast and slow phase patterns
    
    The projection:
    1. Linear projection to lower dimension
    2. Normalize to [-1, 1]
    3. Scale to [0, 2π] for phases
    """
    cfg = self.config
    
    # Ensure correct shape
    embedding = np.asarray(embedding).flatten()
    if len(embedding) != cfg.embedding_dim:
        raise ValueError(f"Expected embedding dim {cfg.embedding_dim}, got {len(embedding)}")
    
    # Normalize input embedding
    embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    # Project to oscillator dimensions
    fast_raw = self.proj_fast @ embedding_norm  # [fast_dim]
    slow_raw = self.proj_slow @ embedding_norm  # [slow_dim]
    
    # Convert to phases
    fast_phases = self._raw_to_phases(fast_raw)
    slow_phases = self._raw_to_phases(slow_raw)
    
    return PhasePair(
        fast=fast_phases,
        slow=slow_phases,
        source_embedding=embedding,
    )

def _raw_to_phases(self, raw: np.ndarray) -> np.ndarray:
    """Convert raw projection to phase angles."""
    cfg = self.config
    
    if cfg.phase_encoding == "linear":
        # Simple linear scaling
        # Normalize to [-1, 1] then scale to [0, 2π]
        normalized = np.tanh(raw)  # Soft normalization to [-1, 1]
        phases = (normalized + 1) * np.pi  # Scale to [0, 2π]
        
    elif cfg.phase_encoding == "angular":
        # Use atan2 for angular encoding
        # Pair up dimensions and compute angles
        n = len(raw)
        phases = np.zeros(n)
        for i in range(0, n-1, 2):
            phases[i] = np.arctan2(raw[i], raw[i+1]) + np.pi  # [0, 2π]
            phases[i+1] = np.arctan2(raw[i+1], raw[i]) + np.pi
        if n % 2 == 1:
            phases[-1] = (np.tanh(raw[-1]) + 1) * np.pi
            
    elif cfg.phase_encoding == "sinusoidal":
        # Sinusoidal encoding (like positional encoding)
        phases = np.mod(raw * np.pi, 2 * np.pi)
        
    else:
        raise ValueError(f"Unknown phase encoding: {cfg.phase_encoding}")
    
    return phases
```

### `phases_to_embed(fast_phases, slow_phases) -> np.ndarray`

Inverse projection: read substrate state as embedding.

```python
def phases_to_embed(self, 
                    fast_phases: np.ndarray, 
                    slow_phases: np.ndarray,
                    fast_weights: Optional[np.ndarray] = None,
                    slow_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project phase patterns back to embedding space.
    
    Args:
        fast_phases: [fast_dim] phase angles
        slow_phases: [slow_dim] phase angles
        fast_weights: Optional weights (e.g., activation potentials)
        slow_weights: Optional weights
    
    Returns:
        [embedding_dim] reconstructed embedding
    
    This is approximate - information is lost in projection.
    Useful for:
    - Similarity search against stored embeddings
    - Understanding what substrate "means" semantically
    """
    # Convert phases back to raw values
    fast_raw = self._phases_to_raw(fast_phases)
    slow_raw = self._phases_to_raw(slow_phases)
    
    # Apply weights if provided (weight by activation)
    if fast_weights is not None:
        fast_raw = fast_raw * fast_weights
    if slow_weights is not None:
        slow_raw = slow_raw * slow_weights
    
    # Inverse project
    fast_contrib = self.proj_fast_inv @ fast_raw
    slow_contrib = self.proj_slow_inv @ slow_raw
    
    # Combine (weighted average)
    # Slow scale gets slightly more weight (identity/gist level)
    embedding = 0.4 * fast_contrib + 0.6 * slow_contrib
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    
    return embedding

def _phases_to_raw(self, phases: np.ndarray) -> np.ndarray:
    """Convert phase angles back to raw values."""
    cfg = self.config
    
    if cfg.phase_encoding == "linear":
        # Inverse of linear scaling
        normalized = (phases / np.pi) - 1  # [0, 2π] → [-1, 1]
        raw = np.arctanh(np.clip(normalized, -0.999, 0.999))
        
    elif cfg.phase_encoding == "angular":
        # Use sin/cos representation
        raw = np.sin(phases)  # Simplified inverse
        
    elif cfg.phase_encoding == "sinusoidal":
        raw = phases / np.pi  # Simplified inverse
        
    else:
        raise ValueError(f"Unknown phase encoding: {cfg.phase_encoding}")
    
    return raw
```

### `text_to_phases(text) -> PhasePair`

Convenience method for text input.

```python
def text_to_phases(self, text: str) -> PhasePair:
    """
    Convert text to phase patterns via embedding.
    
    Requires embedder to be set.
    """
    if self.embedder is None:
        raise RuntimeError("No embedder configured. Pass embedder to __init__.")
    
    embedding = self.embedder(text)
    return self.embed_to_phases(embedding)
```

### `phase_similarity(phases_a, phases_b) -> float`

Compute similarity between phase patterns.

```python
def phase_similarity(self, phases_a: PhasePair, phases_b: PhasePair) -> float:
    """
    Compute similarity between two phase patterns.
    
    This should approximate cosine similarity of original embeddings.
    Uses coherence-like measure.
    """
    # Fast scale similarity
    fast_diff = phases_a.fast - phases_b.fast
    fast_sim = np.mean(np.cos(fast_diff))  # [-1, 1]
    fast_sim = (fast_sim + 1) / 2  # [0, 1]
    
    # Slow scale similarity
    slow_diff = phases_a.slow - phases_b.slow
    slow_sim = np.mean(np.cos(slow_diff))
    slow_sim = (slow_sim + 1) / 2
    
    # Combined (slow weighted higher for semantic similarity)
    return 0.4 * fast_sim + 0.6 * slow_sim

@staticmethod
def coherence_between(phases_a: np.ndarray, phases_b: np.ndarray) -> float:
    """
    Compute coherence if both patterns were active together.
    
    This is what would happen in the substrate.
    """
    combined = np.concatenate([phases_a, phases_b])
    return min(np.abs(np.mean(np.exp(1j * combined))), 0.999)
```

### `stimulate_from_text(substrate, text, strength)`

Integration with substrate.

```python
def stimulate_from_text(self,
                        substrate: 'MultiScaleSubstrate',
                        text: str,
                        strength: float = 0.5) -> PhasePair:
    """
    Stimulate substrate with text content.
    
    Args:
        substrate: The MultiScaleSubstrate to stimulate
        text: Text content to ground
        strength: Stimulation strength
    
    Returns:
        The phase patterns used for stimulation
    """
    phases = self.text_to_phases(text)
    substrate.stimulate_concept(phases.fast, phases.slow, strength)
    return phases

def read_substrate_embedding(self, 
                             substrate: 'MultiScaleSubstrate') -> np.ndarray:
    """
    Read current substrate state as embedding vector.
    
    Uses active oscillator phases weighted by activation.
    """
    return self.phases_to_embed(
        substrate.fast.phases,
        substrate.slow.phases,
        fast_weights=substrate.fast.activation_potentials,
        slow_weights=substrate.slow.activation_potentials,
    )
```

---

## Similarity Preservation Validation

```python
def validate_similarity_preservation(self, 
                                     test_embeddings: List[np.ndarray],
                                     tolerance: float = 0.1) -> dict:
    """
    Test how well projection preserves pairwise similarities.
    
    Args:
        test_embeddings: List of embedding vectors to test
        tolerance: Acceptable deviation from original similarity
    
    Returns:
        Dict with preservation statistics
    """
    n = len(test_embeddings)
    original_sims = []
    projected_sims = []
    
    # Compute all pairwise similarities
    for i in range(n):
        for j in range(i+1, n):
            # Original cosine similarity
            orig_sim = np.dot(test_embeddings[i], test_embeddings[j]) / (
                np.linalg.norm(test_embeddings[i]) * np.linalg.norm(test_embeddings[j])
            )
            original_sims.append(orig_sim)
            
            # Projected phase similarity
            phases_i = self.embed_to_phases(test_embeddings[i])
            phases_j = self.embed_to_phases(test_embeddings[j])
            proj_sim = self.phase_similarity(phases_i, phases_j)
            projected_sims.append(proj_sim)
    
    original_sims = np.array(original_sims)
    projected_sims = np.array(projected_sims)
    
    # Compute preservation metrics
    correlation = np.corrcoef(original_sims, projected_sims)[0, 1]
    mean_error = np.mean(np.abs(original_sims - projected_sims))
    max_error = np.max(np.abs(original_sims - projected_sims))
    
    return {
        'correlation': correlation,
        'mean_error': mean_error,
        'max_error': max_error,
        'within_tolerance': mean_error <= tolerance,
        'n_pairs': len(original_sims),
    }
```

---

## Success Criteria

### Similarity Preservation
1. Cosine similarity correlation > 0.8 between original and projected
2. Mean similarity error < 0.15
3. Semantically similar texts produce high phase coherence

### Correctness
1. embed_to_phases → phases_to_embed approximately recovers original
2. Phase patterns are valid (all in [0, 2π])
3. Projection is deterministic (same input → same output)

### Integration
1. stimulate_from_text activates relevant oscillators
2. read_substrate_embedding produces meaningful vector
3. Works with actual embedder (OpenAI, local model, etc.)

---

## Test Cases

```python
def test_embed_to_phases_deterministic():
    """Same embedding should produce same phases."""
    grounding = SemanticGrounding()
    embedding = np.random.randn(1536)
    
    phases1 = grounding.embed_to_phases(embedding)
    phases2 = grounding.embed_to_phases(embedding)
    
    np.testing.assert_array_almost_equal(phases1.fast, phases2.fast)
    np.testing.assert_array_almost_equal(phases1.slow, phases2.slow)

def test_phases_in_valid_range():
    """Phases should be in [0, 2π]."""
    grounding = SemanticGrounding()
    
    for _ in range(100):
        embedding = np.random.randn(1536)
        phases = grounding.embed_to_phases(embedding)
        
        assert np.all(phases.fast >= 0) and np.all(phases.fast <= 2*np.pi)
        assert np.all(phases.slow >= 0) and np.all(phases.slow <= 2*np.pi)

def test_similar_embeddings_similar_phases():
    """Similar embeddings should produce similar phase patterns."""
    grounding = SemanticGrounding()
    
    # Create similar embeddings
    base = np.random.randn(1536)
    similar = base + 0.1 * np.random.randn(1536)  # Small perturbation
    different = np.random.randn(1536)  # Unrelated
    
    phases_base = grounding.embed_to_phases(base)
    phases_similar = grounding.embed_to_phases(similar)
    phases_different = grounding.embed_to_phases(different)
    
    sim_to_similar = grounding.phase_similarity(phases_base, phases_similar)
    sim_to_different = grounding.phase_similarity(phases_base, phases_different)
    
    assert sim_to_similar > sim_to_different

def test_similarity_preservation():
    """Projection should preserve pairwise similarities."""
    grounding = SemanticGrounding()
    
    # Generate test embeddings with known similarity structure
    embeddings = [np.random.randn(1536) for _ in range(20)]
    
    result = grounding.validate_similarity_preservation(embeddings)
    
    assert result['correlation'] > 0.7  # Reasonable preservation
    assert result['mean_error'] < 0.2

def test_roundtrip_reconstruction():
    """embed_to_phases → phases_to_embed should approximate original."""
    grounding = SemanticGrounding()
    
    original = np.random.randn(1536)
    original = original / np.linalg.norm(original)
    
    phases = grounding.embed_to_phases(original)
    reconstructed = grounding.phases_to_embed(phases.fast, phases.slow)
    
    # Won't be perfect due to dimensionality reduction
    similarity = np.dot(original, reconstructed)
    assert similarity > 0.3  # Some information preserved

def test_substrate_integration():
    """Should integrate with substrate."""
    grounding = SemanticGrounding(SemanticGroundingConfig(
        fast_oscillators=40,
        slow_oscillators=20,
    ))
    substrate = MultiScaleSubstrate(MultiScaleConfig(
        fast_oscillators=40,
        slow_oscillators=20,
    ))
    
    # Create embedder mock
    def mock_embedder(text):
        return np.random.randn(1536)
    
    grounding.embedder = mock_embedder
    
    # Should not crash
    phases = grounding.stimulate_from_text(substrate, "hello world", 0.5)
    
    assert phases.fast.shape[0] == 40
    assert phases.slow.shape[0] == 20
    
    # Substrate should have some activation
    assert substrate.fast.n_active > 0 or np.max(substrate.fast.activation_potentials) > 0
```

---

## Dependencies

- `numpy`
- `MultiScaleSubstrate` (ORE2-002)
- Optional: embedding provider (OpenAI, local model, etc.)

---

## File Location

```
ore2/
├── core/
│   └── semantic_grounding.py  # <-- This component
├── tests/
│   └── test_semantic_grounding.py
```

---

## Design Decisions to Preserve

1. **Random orthogonal projection** - Johnson-Lindenstrauss guarantees approximate preservation
2. **Separate fast/slow projections** - Different semantic granularity
3. **Slow scale weighted higher** - Gist/identity level more important for meaning
4. **Phase encoding via tanh** - Smooth, bounded, differentiable
5. **Seed for reproducibility** - Same projection across runs
6. **Embedder as dependency injection** - Flexible for different providers

---

## Embedder Examples

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

def openai_embedder(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

grounding = SemanticGrounding(embedder=openai_embedder)

# Local sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def local_embedder(text: str) -> np.ndarray:
    return model.encode(text)

grounding = SemanticGrounding(
    SemanticGroundingConfig(embedding_dim=384),  # MiniLM is 384-dim
    embedder=local_embedder
)
```

---

## Future: Learned Projections

The random projection is a good starting point. For better preservation, train a projection:

```python
# Pseudocode for learned projection
class LearnedProjection(nn.Module):
    def __init__(self, embed_dim, phase_dim):
        self.proj = nn.Linear(embed_dim, phase_dim)
        self.phase_out = nn.Tanh()  # Bounded output
    
    def forward(self, embedding):
        raw = self.proj(embedding)
        return (self.phase_out(raw) + 1) * np.pi

# Training objective: preserve pairwise cosine similarities
# loss = MSE(cosine_sim(e1, e2), phase_coherence(p1, p2))
```

This is for ORE 2.1+, not MVP.
