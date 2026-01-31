"""Tests for SemanticGrounding (ORE2-008)."""

import numpy as np
import pytest

from ore2.core.multi_scale_substrate import MultiScaleConfig, MultiScaleSubstrate
from ore2.core.semantic_grounding import (
    PhasePair,
    SemanticGrounding,
    SemanticGroundingConfig,
)


# ── Specified test cases from briefing ──────────────────────────────────────


def test_embed_to_phases_deterministic():
    """Same embedding should produce same phases."""
    grounding = SemanticGrounding()
    embedding = np.random.RandomState(0).randn(1536)

    phases1 = grounding.embed_to_phases(embedding)
    phases2 = grounding.embed_to_phases(embedding)

    np.testing.assert_array_almost_equal(phases1.fast, phases2.fast)
    np.testing.assert_array_almost_equal(phases1.slow, phases2.slow)


def test_phases_in_valid_range():
    """Phases should be in [0, 2π]."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(1)

    for _ in range(100):
        embedding = rng.randn(1536)
        phases = grounding.embed_to_phases(embedding)

        assert np.all(phases.fast >= 0) and np.all(phases.fast <= 2 * np.pi)
        assert np.all(phases.slow >= 0) and np.all(phases.slow <= 2 * np.pi)


def test_similar_embeddings_similar_phases():
    """Similar embeddings should produce similar phase patterns."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(2)

    # Create similar embeddings
    base = rng.randn(1536)
    similar = base + 0.1 * rng.randn(1536)       # Small perturbation
    different = rng.randn(1536)                    # Unrelated

    phases_base = grounding.embed_to_phases(base)
    phases_similar = grounding.embed_to_phases(similar)
    phases_different = grounding.embed_to_phases(different)

    sim_to_similar = grounding.phase_similarity(phases_base, phases_similar)
    sim_to_different = grounding.phase_similarity(phases_base, phases_different)

    assert sim_to_similar > sim_to_different


def test_similarity_preservation():
    """Projection should preserve pairwise similarities."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(3)

    # Generate test embeddings with *structured* similarity.
    # Pure random 1536-dim vectors are all nearly orthogonal (cos_sim ≈ 0),
    # giving no signal to correlate. Instead, create clusters with
    # varying inter/intra-cluster similarities.
    base_vecs = [rng.randn(1536) for _ in range(5)]
    embeddings = []
    for base in base_vecs:
        embeddings.append(base)
        # Add variations with known similarity to base
        for noise_scale in [0.1, 0.3, 0.8]:
            varied = base + noise_scale * rng.randn(1536)
            embeddings.append(varied)

    result = grounding.validate_similarity_preservation(embeddings)

    # Correlation captures ordering preservation (the important property).
    # Mean absolute error is large because phase_similarity maps to [0,1]
    # while cosine similarity spans [-1,1] - different scales.
    assert result["correlation"] > 0.7  # Reasonable preservation


def test_roundtrip_reconstruction():
    """embed_to_phases → phases_to_embed should approximate original."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(4)

    original = rng.randn(1536)
    original = original / np.linalg.norm(original)

    phases = grounding.embed_to_phases(original)
    reconstructed = grounding.phases_to_embed(phases.fast, phases.slow)

    # Won't be perfect due to dimensionality reduction (1536 → 150).
    # With 150 dimensions from pseudo-inverse reconstruction, expect
    # cosine similarity ~sqrt(150/1536) ≈ 0.31, but weighted combination
    # and phase encoding reduce this slightly.
    similarity = float(np.dot(original, reconstructed))
    assert similarity > 0.2  # Some information preserved


def test_substrate_integration():
    """Should integrate with substrate."""
    grounding = SemanticGrounding(
        SemanticGroundingConfig(
            fast_oscillators=40,
            slow_oscillators=20,
        )
    )
    substrate = MultiScaleSubstrate(
        MultiScaleConfig(
            fast_oscillators=40,
            slow_oscillators=20,
        )
    )

    # Create embedder mock
    rng = np.random.RandomState(5)

    def mock_embedder(text):
        return rng.randn(1536)

    grounding.embedder = mock_embedder

    # Should not crash
    phases = grounding.stimulate_from_text(substrate, "hello world", 0.5)

    assert phases.fast.shape[0] == 40
    assert phases.slow.shape[0] == 20

    # Substrate should have some activation
    assert (
        substrate.fast.n_active > 0
        or np.max(substrate.fast.activation_potentials) > 0
    )


# ── Additional tests ────────────────────────────────────────────────────────


def test_default_config():
    """Default config should have expected values."""
    cfg = SemanticGroundingConfig()
    assert cfg.embedding_dim == 1536
    assert cfg.fast_oscillators == 100
    assert cfg.slow_oscillators == 50
    assert cfg.projection_method == "random_fixed"
    assert cfg.projection_seed == 42
    assert cfg.phase_encoding == "linear"


def test_properties():
    """Properties should reflect config."""
    grounding = SemanticGrounding()
    assert grounding.embedding_dim == 1536
    assert grounding.fast_dim == 100
    assert grounding.slow_dim == 50


def test_phase_pair_fields():
    """PhasePair should have fast, slow, and optional source_embedding."""
    pp = PhasePair(
        fast=np.zeros(10),
        slow=np.zeros(5),
        source_embedding=np.ones(20),
    )
    assert pp.fast.shape == (10,)
    assert pp.slow.shape == (5,)
    assert pp.source_embedding is not None


def test_phase_pair_source_embedding_stored():
    """embed_to_phases should store original embedding in PhasePair."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(6)
    embedding = rng.randn(1536)

    phases = grounding.embed_to_phases(embedding)

    assert phases.source_embedding is not None
    np.testing.assert_array_equal(phases.source_embedding, embedding)


def test_wrong_embedding_dim_raises():
    """Wrong embedding dimension should raise ValueError."""
    grounding = SemanticGrounding()

    with pytest.raises(ValueError, match="Expected embedding dim"):
        grounding.embed_to_phases(np.zeros(100))


def test_text_to_phases_no_embedder():
    """text_to_phases without embedder should raise RuntimeError."""
    grounding = SemanticGrounding()

    with pytest.raises(RuntimeError, match="No embedder configured"):
        grounding.text_to_phases("hello")


def test_text_to_phases_with_embedder():
    """text_to_phases with embedder should work."""
    rng = np.random.RandomState(7)

    def mock_embedder(text):
        return rng.randn(1536)

    grounding = SemanticGrounding(embedder=mock_embedder)
    phases = grounding.text_to_phases("hello")

    assert phases.fast.shape == (100,)
    assert phases.slow.shape == (50,)


def test_phase_similarity_identical():
    """Identical phase patterns should have similarity ~1.0."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(8)

    embedding = rng.randn(1536)
    phases = grounding.embed_to_phases(embedding)

    sim = grounding.phase_similarity(phases, phases)
    assert sim > 0.99


def test_phase_similarity_range():
    """Phase similarity should be in [0, 1]."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(9)

    for _ in range(20):
        e1 = rng.randn(1536)
        e2 = rng.randn(1536)
        p1 = grounding.embed_to_phases(e1)
        p2 = grounding.embed_to_phases(e2)

        sim = grounding.phase_similarity(p1, p2)
        assert 0.0 <= sim <= 1.0


def test_coherence_between():
    """coherence_between should return value in [0, 0.999]."""
    a = np.zeros(10)               # All aligned
    b = np.zeros(10)
    coh = SemanticGrounding.coherence_between(a, b)
    assert coh > 0.9

    # Random phases should have low coherence
    rng = np.random.RandomState(10)
    c = rng.uniform(0, 2 * np.pi, 100)
    d = rng.uniform(0, 2 * np.pi, 100)
    coh_random = SemanticGrounding.coherence_between(c, d)
    assert coh_random < 0.5


def test_coherence_between_capped():
    """coherence_between should be capped at 0.999."""
    a = np.zeros(100)
    b = np.zeros(100)
    coh = SemanticGrounding.coherence_between(a, b)
    assert coh <= 0.999


def test_projection_reproducible_across_instances():
    """Two instances with same seed should produce same projections."""
    g1 = SemanticGrounding(SemanticGroundingConfig(projection_seed=123))
    g2 = SemanticGrounding(SemanticGroundingConfig(projection_seed=123))

    np.testing.assert_array_almost_equal(g1.proj_fast, g2.proj_fast)
    np.testing.assert_array_almost_equal(g1.proj_slow, g2.proj_slow)


def test_different_seeds_different_projections():
    """Different seeds should produce different projections."""
    g1 = SemanticGrounding(SemanticGroundingConfig(projection_seed=42))
    g2 = SemanticGrounding(SemanticGroundingConfig(projection_seed=99))

    assert not np.allclose(g1.proj_fast, g2.proj_fast)


def test_projection_shapes():
    """Projection matrices should have correct shapes."""
    cfg = SemanticGroundingConfig(
        embedding_dim=768,
        fast_oscillators=60,
        slow_oscillators=30,
    )
    grounding = SemanticGrounding(cfg)

    assert grounding.proj_fast.shape == (60, 768)
    assert grounding.proj_slow.shape == (30, 768)
    assert grounding.proj_fast_inv.shape == (768, 60)
    assert grounding.proj_slow_inv.shape == (768, 30)


def test_read_substrate_embedding():
    """read_substrate_embedding should return an embedding vector."""
    cfg = SemanticGroundingConfig(fast_oscillators=40, slow_oscillators=20)
    grounding = SemanticGrounding(cfg)
    substrate = MultiScaleSubstrate(
        MultiScaleConfig(fast_oscillators=40, slow_oscillators=20)
    )

    embedding = grounding.read_substrate_embedding(substrate)
    assert embedding.shape == (1536,)
    # Should be normalized
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01 or norm < 0.01  # normalized or near-zero


def test_phases_to_embed_with_weights():
    """phases_to_embed should accept optional weights."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(11)

    fast_phases = rng.uniform(0, 2 * np.pi, 100)
    slow_phases = rng.uniform(0, 2 * np.pi, 50)
    fast_weights = rng.uniform(0, 1, 100)
    slow_weights = rng.uniform(0, 1, 50)

    # Without weights
    emb_no_weights = grounding.phases_to_embed(fast_phases, slow_phases)
    # With weights
    emb_with_weights = grounding.phases_to_embed(
        fast_phases, slow_phases, fast_weights, slow_weights
    )

    assert emb_no_weights.shape == (1536,)
    assert emb_with_weights.shape == (1536,)
    # Should differ when weights are applied
    assert not np.allclose(emb_no_weights, emb_with_weights)


def test_learned_projection_raises():
    """Learned projection method should raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        SemanticGrounding(SemanticGroundingConfig(projection_method="learned"))


def test_unknown_projection_raises():
    """Unknown projection method should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown projection method"):
        SemanticGrounding(SemanticGroundingConfig(projection_method="magic"))


def test_unknown_phase_encoding_raises():
    """Unknown phase encoding should raise ValueError on embed."""
    grounding = SemanticGrounding()
    grounding.config.phase_encoding = "unknown"

    with pytest.raises(ValueError, match="Unknown phase encoding"):
        grounding.embed_to_phases(np.zeros(1536))


def test_angular_phase_encoding():
    """Angular phase encoding should produce valid phases."""
    cfg = SemanticGroundingConfig(phase_encoding="angular")
    grounding = SemanticGrounding(cfg)
    rng = np.random.RandomState(12)

    embedding = rng.randn(1536)
    phases = grounding.embed_to_phases(embedding)

    assert np.all(phases.fast >= 0) and np.all(phases.fast <= 2 * np.pi)
    assert np.all(phases.slow >= 0) and np.all(phases.slow <= 2 * np.pi)


def test_sinusoidal_phase_encoding():
    """Sinusoidal phase encoding should produce valid phases."""
    cfg = SemanticGroundingConfig(phase_encoding="sinusoidal")
    grounding = SemanticGrounding(cfg)
    rng = np.random.RandomState(13)

    embedding = rng.randn(1536)
    phases = grounding.embed_to_phases(embedding)

    assert np.all(phases.fast >= 0) and np.all(phases.fast <= 2 * np.pi)
    assert np.all(phases.slow >= 0) and np.all(phases.slow <= 2 * np.pi)


def test_validate_similarity_preservation_output():
    """validate_similarity_preservation should return expected fields."""
    grounding = SemanticGrounding()
    rng = np.random.RandomState(14)

    embeddings = [rng.randn(1536) for _ in range(5)]
    result = grounding.validate_similarity_preservation(embeddings)

    assert "correlation" in result
    assert "mean_error" in result
    assert "max_error" in result
    assert "within_tolerance" in result
    assert "n_pairs" in result
    assert result["n_pairs"] == 10  # C(5,2) = 10


def test_custom_embedding_dim():
    """Should work with non-default embedding dimensions."""
    cfg = SemanticGroundingConfig(embedding_dim=384, fast_oscillators=30, slow_oscillators=15)
    grounding = SemanticGrounding(cfg)
    rng = np.random.RandomState(15)

    embedding = rng.randn(384)
    phases = grounding.embed_to_phases(embedding)

    assert phases.fast.shape == (30,)
    assert phases.slow.shape == (15,)
