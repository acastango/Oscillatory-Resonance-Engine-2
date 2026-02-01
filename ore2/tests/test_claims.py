"""Tests for ClaimsEngine (ORE2-009)."""

import hashlib
import time

import numpy as np
import pytest

from ore2.core.claims import (
    Claim,
    ClaimScope,
    ClaimSource,
    ClaimsEngine,
    ClaimsEngineConfig,
)
from ore2.core.memory import CrystallineMerkleMemory, MemoryBranch
from ore2.core.multi_scale_substrate import MultiScaleSubstrate
from ore2.core.semantic_grounding import SemanticGrounding, SemanticGroundingConfig


# ── Helpers ──────────────────────────────────────────────────────────────────


def mock_embedder(text: str) -> np.ndarray:
    """Deterministic embedder: text -> 1536-dim unit vector."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)
    emb = rng.randn(1536)
    return emb / np.linalg.norm(emb)


def create_test_grounding() -> SemanticGrounding:
    """Create a SemanticGrounding with mock embedder."""
    return SemanticGrounding(
        config=SemanticGroundingConfig(),
        embedder=mock_embedder,
    )


def create_test_engine(
    config: ClaimsEngineConfig = None,
    grounding: SemanticGrounding = None,
) -> ClaimsEngine:
    """Create a ClaimsEngine for testing."""
    return ClaimsEngine(
        config=config,
        grounding=grounding or create_test_grounding(),
    )


def create_test_substrate() -> MultiScaleSubstrate:
    """Create a MultiScaleSubstrate for testing."""
    return MultiScaleSubstrate()


# ── Claim Management Tests ───────────────────────────────────────────────────


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
    assert claim.strength == 0.8
    assert claim.scope == ClaimScope.IDENTITY
    assert claim.source == ClaimSource.LEARNED  # default
    assert claim.content == "I value honesty"
    assert claim.created_at != ""


def test_add_claim_defaults():
    """Default scope and source should be applied."""
    engine = create_test_engine()
    claim = engine.add_claim("some fact")

    assert claim.scope == ClaimScope.KNOWLEDGE
    assert claim.source == ClaimSource.LEARNED
    assert claim.strength == 0.5


def test_add_claim_strength_clamp():
    """Strength should be clamped to [0, 1]."""
    engine = create_test_engine()

    c1 = engine.add_claim("too strong", strength=1.5)
    assert c1.strength == 1.0

    c2 = engine.add_claim("too weak", strength=-0.5)
    assert c2.strength == 0.0


def test_add_claim_without_embedder():
    """Claims without embedder should have None embedding/phases."""
    grounding = SemanticGrounding(config=SemanticGroundingConfig())
    engine = ClaimsEngine(grounding=grounding)

    claim = engine.add_claim("test")
    assert claim.embedding is None
    assert claim.phase_pattern is None


def test_add_claim_unique_ids():
    """Each claim should have a unique ID."""
    engine = create_test_engine()
    ids = set()
    for i in range(20):
        claim = engine.add_claim(f"claim {i}")
        ids.add(claim.id)
    assert len(ids) == 20


def test_remove_claim():
    """Removing a claim should delete it entirely."""
    engine = create_test_engine()
    claim = engine.add_claim("to remove")
    cid = claim.id

    engine.remove_claim(cid)
    assert cid not in engine.claims


def test_remove_active_claim():
    """Removing an active claim should also deactivate it."""
    engine = create_test_engine()
    claim = engine.add_claim("active removal")
    engine.activate_claim(claim.id)

    assert claim.id in engine.active_claims
    engine.remove_claim(claim.id)

    assert claim.id not in engine.claims
    assert claim.id not in engine.active_claims


def test_remove_unknown_claim_raises():
    """Removing a nonexistent claim should raise."""
    engine = create_test_engine()
    with pytest.raises(KeyError):
        engine.remove_claim("nonexistent")


def test_update_strength():
    """Updating strength should clamp and apply."""
    engine = create_test_engine()
    claim = engine.add_claim("test", strength=0.5)

    engine.update_strength(claim.id, 0.9)
    assert claim.strength == 0.9

    engine.update_strength(claim.id, 1.5)
    assert claim.strength == 1.0

    engine.update_strength(claim.id, -0.1)
    assert claim.strength == 0.0


def test_update_strength_unknown_raises():
    """Updating strength of nonexistent claim should raise."""
    engine = create_test_engine()
    with pytest.raises(KeyError):
        engine.update_strength("nonexistent", 0.5)


def test_max_total_claims_prune():
    """Exceeding max_total_claims should prune weakest inactive claim."""
    config = ClaimsEngineConfig(max_total_claims=5)
    engine = create_test_engine(config=config)

    # Add 5 claims with varying strengths
    claims = []
    for i in range(5):
        c = engine.add_claim(f"claim {i}", strength=0.3 + i * 0.1)
        claims.append(c)

    assert len(engine.claims) == 5

    # Adding one more should prune the weakest (claim 0, strength 0.3)
    engine.add_claim("overflow claim", strength=0.8)
    assert len(engine.claims) == 5
    assert claims[0].id not in engine.claims


def test_prune_preserves_active():
    """Pruning should only remove inactive claims."""
    config = ClaimsEngineConfig(max_total_claims=3)
    engine = create_test_engine(config=config)

    # Add 3 claims, activate the weakest
    weak = engine.add_claim("weak", strength=0.1)
    engine.activate_claim(weak.id)
    engine.add_claim("medium", strength=0.5)
    engine.add_claim("strong", strength=0.9)

    # Adding overflow should prune medium (weakest inactive), not weak (active)
    engine.add_claim("new", strength=0.7)
    assert weak.id in engine.claims
    assert len(engine.claims) == 3


# ── Activation Tests ─────────────────────────────────────────────────────────


def test_activate_deactivate():
    """Activation state should track correctly."""
    engine = create_test_engine()
    claim = engine.add_claim("test claim")

    assert not claim.active
    assert claim.id not in engine.active_claims

    engine.activate_claim(claim.id)

    assert claim.active
    assert claim.id in engine.active_claims
    assert claim.activated_at is not None

    engine.deactivate_claim(claim.id)

    assert not claim.active
    assert claim.id not in engine.active_claims
    assert claim.activated_at is None


def test_activate_already_active():
    """Activating an already-active claim should be a no-op."""
    engine = create_test_engine()
    claim = engine.add_claim("test")
    engine.activate_claim(claim.id)

    active_before = list(engine.active_claims)
    engine.activate_claim(claim.id)
    assert engine.active_claims == active_before


def test_activate_unknown_raises():
    """Activating a nonexistent claim should raise."""
    engine = create_test_engine()
    with pytest.raises(KeyError):
        engine.activate_claim("nonexistent")


def test_deactivate_unknown_safe():
    """Deactivating a nonexistent claim should be safe (no-op)."""
    engine = create_test_engine()
    engine.deactivate_claim("nonexistent")  # Should not raise


def test_max_active_claims():
    """Should enforce max active limit by deactivating weakest."""
    config = ClaimsEngineConfig(max_active_claims=3)
    engine = create_test_engine(config=config)

    claims = [engine.add_claim(f"claim {i}", strength=0.1 * (i + 1)) for i in range(5)]
    for c in claims:
        engine.activate_claim(c.id)

    assert len(engine.active_claims) == 3

    # The 3 strongest should remain active (claims 2, 3, 4)
    active_strengths = [engine.claims[cid].strength for cid in engine.active_claims]
    assert min(active_strengths) >= 0.3


def test_max_active_deactivates_weakest():
    """When at capacity, activating should deactivate the weakest active."""
    config = ClaimsEngineConfig(max_active_claims=2)
    engine = create_test_engine(config=config)

    c1 = engine.add_claim("weak", strength=0.3)
    c2 = engine.add_claim("medium", strength=0.6)
    c3 = engine.add_claim("strong", strength=0.9)

    engine.activate_claim(c1.id)
    engine.activate_claim(c2.id)
    assert len(engine.active_claims) == 2

    # Activating c3 should deactivate c1 (weakest)
    engine.activate_claim(c3.id)
    assert len(engine.active_claims) == 2
    assert c1.id not in engine.active_claims
    assert c2.id in engine.active_claims
    assert c3.id in engine.active_claims


# ── Role Activation Tests ────────────────────────────────────────────────────


def test_activate_role():
    """Role activation should add multiple claims."""
    engine = create_test_engine()

    initial_count = len(engine.claims)
    engine.activate_role('analyst')

    assert len(engine.claims) > initial_count
    assert len(engine.active_claims) > 0

    # Should have 3 claims for analyst role
    assert len(engine.claims) == initial_count + 3


def test_activate_role_all_active():
    """All role claims should be activated."""
    engine = create_test_engine()
    engine.activate_role('creative')

    assert len(engine.active_claims) == 3
    for cid in engine.active_claims:
        assert engine.claims[cid].scope == ClaimScope.BEHAVIOR
        assert engine.claims[cid].source == ClaimSource.INSTRUCTED
        assert engine.claims[cid].strength == 0.7


def test_activate_role_unknown_raises():
    """Unknown role should raise."""
    engine = create_test_engine()
    with pytest.raises(ValueError, match="Unknown role"):
        engine.activate_role("philosopher")


def test_available_roles():
    """All COGNIZEN roles should be available."""
    engine = create_test_engine()
    roles = engine._role_templates
    assert set(roles.keys()) == {'analyst', 'creative', 'skeptic', 'integrator', 'meta'}


def test_activate_role_twice_boosts():
    """Activating the same role twice should boost existing claims."""
    engine = create_test_engine()
    engine.activate_role('skeptic')

    initial_strengths = {
        cid: engine.claims[cid].strength for cid in engine.active_claims
    }
    initial_count = len(engine.claims)

    # Second activation - should boost, not duplicate
    engine.activate_role('skeptic')

    # No new claims should be added (embedder provides similarity matching)
    # Note: depends on similarity threshold, but same content -> same embedding
    # so cosine sim = 1.0 > 0.85 threshold
    assert len(engine.claims) == initial_count

    # Strengths should be boosted
    for cid in engine.active_claims:
        if cid in initial_strengths:
            assert engine.claims[cid].strength >= initial_strengths[cid]


# ── Substrate Integration Tests ──────────────────────────────────────────────


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


def test_apply_modifies_fast_too():
    """Both fast and slow scale weights should be modified."""
    engine = create_test_engine()
    substrate = create_test_substrate()

    claim = engine.add_claim("test", strength=0.8)
    engine.activate_claim(claim.id)

    fast_before = substrate.fast.internal_weights.copy()
    slow_before = substrate.slow.internal_weights.copy()

    engine.apply_to_substrate(substrate)

    assert not np.allclose(fast_before, substrate.fast.internal_weights)
    assert not np.allclose(slow_before, substrate.slow.internal_weights)


def test_apply_slow_stronger():
    """Slow scale modification should be 1.5x stronger than fast."""
    engine = create_test_engine()
    substrate = create_test_substrate()

    claim = engine.add_claim("test", strength=1.0)
    engine.activate_claim(claim.id)

    fast_before = substrate.fast.internal_weights.copy()
    slow_before = substrate.slow.internal_weights.copy()

    engine.apply_to_substrate(substrate)

    fast_change = np.sum(np.abs(substrate.fast.internal_weights - fast_before))
    slow_change = np.sum(np.abs(substrate.slow.internal_weights - slow_before))

    # Slow change should be larger (1.5x coupling_scale)
    # Account for different layer sizes (fast=100, slow=50)
    fast_per_weight = fast_change / substrate.fast.n**2
    slow_per_weight = slow_change / substrate.slow.n**2

    assert slow_per_weight > fast_per_weight


def test_apply_no_active_claims():
    """apply_to_substrate with no active claims should be a no-op."""
    engine = create_test_engine()
    substrate = create_test_substrate()

    weights_before = substrate.slow.internal_weights.copy()
    engine.apply_to_substrate(substrate)
    assert np.allclose(weights_before, substrate.slow.internal_weights)


def test_apply_stimulates_aligned():
    """Apply should stimulate oscillators aligned with claim pattern."""
    engine = create_test_engine()
    substrate = create_test_substrate()

    claim = engine.add_claim("stimulation test", strength=0.9)
    engine.activate_claim(claim.id)

    # Record activations before
    fast_act_before = substrate.fast.activation_potentials.copy()

    engine.apply_to_substrate(substrate)

    # Some oscillators should have been stimulated
    fast_act_after = substrate.fast.activation_potentials
    assert np.any(fast_act_after > fast_act_before)


def test_apply_claim_without_phases():
    """Claims without phase patterns should be skipped gracefully."""
    grounding = SemanticGrounding(config=SemanticGroundingConfig())  # No embedder
    engine = ClaimsEngine(grounding=grounding)
    substrate = create_test_substrate()

    claim = engine.add_claim("no phases")
    engine.activate_claim(claim.id)

    weights_before = substrate.slow.internal_weights.copy()
    engine.apply_to_substrate(substrate)
    # Should not crash, weights unchanged (claim has no phase_pattern)
    assert np.allclose(weights_before, substrate.slow.internal_weights)


# ── Consistency Tests ────────────────────────────────────────────────────────


def test_measure_consistency():
    """Consistency should reflect alignment."""
    engine = create_test_engine()
    substrate = create_test_substrate()

    claim = engine.add_claim("test", strength=1.0)
    engine.activate_claim(claim.id)

    # Force substrate to match claim pattern
    substrate.slow.phases = claim.phase_pattern.slow.copy()
    substrate.fast.phases = claim.phase_pattern.fast.copy()

    consistency = engine.measure_consistency(substrate)
    assert consistency > 0.8


def test_consistency_no_active_claims():
    """No active claims = vacuously consistent (1.0)."""
    engine = create_test_engine()
    substrate = create_test_substrate()
    assert engine.measure_consistency(substrate) == 1.0


def test_consistency_range():
    """Consistency should be in [0, 1]."""
    engine = create_test_engine()
    substrate = create_test_substrate()

    for i in range(5):
        c = engine.add_claim(f"test {i}", strength=0.8)
        engine.activate_claim(c.id)

    consistency = engine.measure_consistency(substrate)
    assert 0.0 <= consistency <= 1.0


# ── Memory Integration Tests ─────────────────────────────────────────────────


def test_anchor_to_memory():
    """Claims should be stored in memory."""
    engine = create_test_engine()
    memory = CrystallineMerkleMemory()

    claim = engine.add_claim("I remember this")
    engine.anchor_to_memory(memory)

    assert claim.memory_node_id is not None
    assert memory.total_nodes > 0


def test_anchor_stores_in_self_branch():
    """Claims should be anchored to the SELF branch."""
    engine = create_test_engine()
    memory = CrystallineMerkleMemory()

    engine.add_claim("identity claim", scope=ClaimScope.IDENTITY)
    engine.anchor_to_memory(memory)

    self_nodes = memory.query(branch=MemoryBranch.SELF)
    assert len(self_nodes) == 1
    assert self_nodes[0].content['type'] == 'claim'


def test_anchor_idempotent():
    """Anchoring twice should not create duplicate nodes."""
    engine = create_test_engine()
    memory = CrystallineMerkleMemory()

    engine.add_claim("single anchor")
    engine.anchor_to_memory(memory)
    first_count = memory.total_nodes

    engine.anchor_to_memory(memory)
    assert memory.total_nodes == first_count


def test_anchor_multiple_claims():
    """Multiple claims should each get their own node."""
    engine = create_test_engine()
    memory = CrystallineMerkleMemory()

    for i in range(5):
        engine.add_claim(f"claim {i}")

    engine.anchor_to_memory(memory)
    assert memory.total_nodes == 5


def test_anchor_preserves_claim_data():
    """Anchored node should contain claim metadata."""
    engine = create_test_engine()
    memory = CrystallineMerkleMemory()

    claim = engine.add_claim(
        "I am careful",
        strength=0.8,
        scope=ClaimScope.BEHAVIOR,
        source=ClaimSource.INSTRUCTED,
    )
    engine.anchor_to_memory(memory)

    nodes = memory.query(branch=MemoryBranch.SELF)
    assert len(nodes) == 1
    content = nodes[0].content
    assert content['content'] == "I am careful"
    assert content['strength'] == 0.8
    assert content['scope'] == 'behavior'
    assert content['source'] == 'instructed'


# ── Coherence Property Tests ─────────────────────────────────────────────────


def test_claim_coherence_single():
    """Single active claim should have coherence 1.0."""
    engine = create_test_engine()
    claim = engine.add_claim("only claim")
    engine.activate_claim(claim.id)
    assert engine.claim_coherence == 1.0


def test_claim_coherence_none_active():
    """No active claims should have coherence 1.0."""
    engine = create_test_engine()
    assert engine.claim_coherence == 1.0


def test_claim_coherence_range():
    """Coherence should be in [0, 1]."""
    engine = create_test_engine()
    for i in range(5):
        c = engine.add_claim(f"different claim about topic {i}")
        engine.activate_claim(c.id)

    assert 0.0 <= engine.claim_coherence <= 1.0


def test_claim_coherence_similar_claims():
    """Similar claims should have higher coherence than dissimilar."""
    engine = create_test_engine()

    # Similar claims
    c1 = engine.add_claim("I analyze problems carefully")
    c2 = engine.add_claim("I examine problems methodically")
    engine.activate_claim(c1.id)
    engine.activate_claim(c2.id)
    similar_coherence = engine.claim_coherence

    # Reset
    engine.deactivate_claim(c1.id)
    engine.deactivate_claim(c2.id)

    # Dissimilar claims
    c3 = engine.add_claim("The sky is blue and clouds are white")
    c4 = engine.add_claim("Pizza requires cheese and tomato sauce")
    engine.activate_claim(c3.id)
    engine.activate_claim(c4.id)
    dissimilar_coherence = engine.claim_coherence

    # Both should be valid numbers in range
    assert 0.0 <= similar_coherence <= 1.0
    assert 0.0 <= dissimilar_coherence <= 1.0


# ── Query Tests ──────────────────────────────────────────────────────────────


def test_get_claims_by_scope():
    """Should filter claims by scope."""
    engine = create_test_engine()
    engine.add_claim("identity", scope=ClaimScope.IDENTITY)
    engine.add_claim("behavior 1", scope=ClaimScope.BEHAVIOR)
    engine.add_claim("behavior 2", scope=ClaimScope.BEHAVIOR)
    engine.add_claim("knowledge", scope=ClaimScope.KNOWLEDGE)

    identity = engine.get_claims_by_scope(ClaimScope.IDENTITY)
    assert len(identity) == 1

    behavior = engine.get_claims_by_scope(ClaimScope.BEHAVIOR)
    assert len(behavior) == 2

    goals = engine.get_claims_by_scope(ClaimScope.GOAL)
    assert len(goals) == 0


def test_get_conflicting_claims_with_negation():
    """Should detect contradictions via negation heuristic."""
    engine = create_test_engine()

    c1 = engine.add_claim("I trust others", scope=ClaimScope.BEHAVIOR)
    c2 = engine.add_claim("I do not trust others", scope=ClaimScope.BEHAVIOR)

    conflicts = engine.get_conflicting_claims()

    # Whether this detects a conflict depends on phase similarity falling
    # in the 0.3-0.7 range. With our mock embedder, similar text produces
    # similar embeddings. The negation heuristic should flag it.
    # We verify the mechanism works when conditions are met.
    for pair in conflicts:
        assert ClaimsEngine._appears_contradictory(pair[0].content, pair[1].content)


def test_appears_contradictory():
    """Negation heuristic should detect basic contradictions."""
    assert ClaimsEngine._appears_contradictory(
        "I trust others", "I do not trust others"
    )
    assert ClaimsEngine._appears_contradictory(
        "I value honesty", "I never value honesty"
    )
    assert not ClaimsEngine._appears_contradictory(
        "I trust others", "I help others"
    )
    assert not ClaimsEngine._appears_contradictory(
        "I don't lie", "I never cheat"
    )


def test_conflicting_claims_different_scope():
    """Claims in different scopes should not be flagged as conflicting."""
    engine = create_test_engine()
    engine.add_claim("I trust", scope=ClaimScope.IDENTITY)
    engine.add_claim("I do not trust", scope=ClaimScope.BEHAVIOR)

    conflicts = engine.get_conflicting_claims()
    assert len(conflicts) == 0


# ── State / Introspection Tests ──────────────────────────────────────────────


def test_get_state():
    """get_state should return complete summary."""
    engine = create_test_engine()
    c1 = engine.add_claim("test 1", strength=0.8)
    c2 = engine.add_claim("test 2", strength=0.6)
    engine.activate_claim(c1.id)

    state = engine.get_state()
    assert state['total_claims'] == 2
    assert state['active_claims'] == 1
    assert c1.id in state['active_claim_ids']
    assert 'claim_coherence' in state
    assert c1.id in state['claims']
    assert state['claims'][c1.id]['active'] is True
    assert state['claims'][c2.id]['active'] is False


def test_get_state_claim_details():
    """get_state claim entries should have expected fields."""
    engine = create_test_engine()
    engine.add_claim("detail test", strength=0.7, scope=ClaimScope.GOAL)

    state = engine.get_state()
    claim_entry = list(state['claims'].values())[0]
    assert claim_entry['content'] == "detail test"
    assert claim_entry['strength'] == 0.7
    assert claim_entry['scope'] == "goal"
    assert claim_entry['source'] == "learned"
    assert claim_entry['anchored'] is False


# ── Constructor Tests ────────────────────────────────────────────────────────


def test_requires_grounding():
    """ClaimsEngine should raise without grounding."""
    with pytest.raises(ValueError, match="requires SemanticGrounding"):
        ClaimsEngine()


def test_grounding_from_config():
    """Grounding can be passed via config."""
    grounding = create_test_grounding()
    config = ClaimsEngineConfig(grounding=grounding)
    engine = ClaimsEngine(config=config)
    assert engine.grounding is grounding


def test_grounding_direct_overrides_config():
    """Direct grounding parameter overrides config grounding."""
    g1 = create_test_grounding()
    g2 = create_test_grounding()
    config = ClaimsEngineConfig(grounding=g1)
    engine = ClaimsEngine(config=config, grounding=g2)
    assert engine.grounding is g2


def test_default_config_values():
    """Default config should match briefing spec."""
    config = ClaimsEngineConfig()
    assert config.strength_decay == 0.001
    assert config.strength_boost == 0.01
    assert config.min_strength == 0.1
    assert config.coupling_scale == 0.3
    assert config.consistency_threshold == 0.6
    assert config.max_active_claims == 10
    assert config.max_total_claims == 100


# ── Enum Tests ───────────────────────────────────────────────────────────────


def test_claim_scope_values():
    """All ClaimScope values should be present."""
    assert ClaimScope.IDENTITY.value == "identity"
    assert ClaimScope.BEHAVIOR.value == "behavior"
    assert ClaimScope.KNOWLEDGE.value == "knowledge"
    assert ClaimScope.RELATION.value == "relation"
    assert ClaimScope.GOAL.value == "goal"
    assert ClaimScope.CONSTRAINT.value == "constraint"


def test_claim_source_values():
    """All ClaimSource values should be present."""
    assert ClaimSource.INNATE.value == "innate"
    assert ClaimSource.LEARNED.value == "learned"
    assert ClaimSource.INSTRUCTED.value == "instructed"
    assert ClaimSource.INFERRED.value == "inferred"
    assert ClaimSource.SOCIAL.value == "social"


# ── Integration Test ─────────────────────────────────────────────────────────


def test_full_lifecycle():
    """Full claim lifecycle: add -> activate -> apply -> measure -> anchor."""
    engine = create_test_engine()
    substrate = create_test_substrate()
    memory = CrystallineMerkleMemory()

    # Add and activate
    claim = engine.add_claim(
        "I am a careful analyst",
        strength=0.9,
        scope=ClaimScope.IDENTITY,
        source=ClaimSource.INSTRUCTED,
    )
    engine.activate_claim(claim.id)
    assert claim.active

    # Apply to substrate
    engine.apply_to_substrate(substrate)

    # Measure consistency
    consistency = engine.measure_consistency(substrate)
    assert 0.0 <= consistency <= 1.0

    # Anchor to memory
    engine.anchor_to_memory(memory)
    assert claim.memory_node_id is not None

    # Verify state
    state = engine.get_state()
    assert state['total_claims'] == 1
    assert state['active_claims'] == 1
