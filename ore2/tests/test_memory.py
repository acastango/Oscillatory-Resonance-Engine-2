"""Tests for CrystallineMerkleMemory (ORE2-004)."""

import numpy as np
import pytest

from ore2.core.memory import (
    ConsolidationQueue,
    CrystallineMerkleMemory,
    MemoryBranch,
    MemoryNode,
)


# ── Specified test cases from briefing ──────────────────────────────────────


def test_basic_add():
    """Basic node addition."""
    mem = CrystallineMerkleMemory()

    node = mem.add(
        MemoryBranch.EXPERIENCES,
        {"event": "test"},
        immediate=True,
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
        immediate=False,
    )

    # Not in tree yet
    assert node.id not in mem.nodes
    assert not mem.consolidation_queue.is_empty()

    # Consolidate
    result = mem.consolidate()

    assert result["consolidated"] == 1
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
        "global_coherence": 0.75,
        "cross_scale_coherence": 0.5,
        "time": 123.456,
    }

    node = mem.add(
        MemoryBranch.SELF,
        {"claim": "anchored"},
        substrate_state=substrate_state,
    )

    assert node.substrate_anchor is not None
    assert node.coherence_at_creation == 0.75


# ── Additional tests ────────────────────────────────────────────────────────


def test_branch_roots_initialized():
    """All four branch roots should be created on init."""
    mem = CrystallineMerkleMemory()

    assert len(mem.branch_roots) == 4
    for branch in MemoryBranch:
        assert branch in mem.branch_roots
        root_id = mem.branch_roots[branch]
        assert root_id in mem.nodes


def test_root_hash_exists():
    """Root hash should be non-empty after init."""
    mem = CrystallineMerkleMemory()
    assert len(mem.root_hash) == 64  # SHA256 hex digest


def test_root_hash_changes_on_add():
    """Root hash should change when a node is added."""
    mem = CrystallineMerkleMemory()
    hash_before = mem.root_hash

    mem.add(MemoryBranch.EXPERIENCES, {"event": "new"})

    assert mem.root_hash != hash_before


def test_multiple_adds_to_same_branch():
    """Multiple nodes can be added to the same branch."""
    mem = CrystallineMerkleMemory()

    for i in range(5):
        mem.add(MemoryBranch.EXPERIENCES, {"event": f"event_{i}"})

    assert mem.total_nodes == 5

    # All should be children of the EXPERIENCES root
    exp_root_id = mem.branch_roots[MemoryBranch.EXPERIENCES]
    assert len(mem.nodes[exp_root_id].children_ids) == 5


def test_adds_across_branches():
    """Nodes in different branches should not create tension."""
    mem = CrystallineMerkleMemory()

    mem.add(MemoryBranch.SELF, {"info": "the sky is blue and clear"})
    mem.add(MemoryBranch.INSIGHTS, {"info": "the sky is blue and clear"})

    # Different branches = no tension detection
    assert len(mem.grain_boundaries) == 0


def test_query_by_branch():
    """Query should filter by branch."""
    mem = CrystallineMerkleMemory()

    mem.add(MemoryBranch.SELF, {"claim": "I am"})
    mem.add(MemoryBranch.INSIGHTS, {"insight": "water is wet"})
    mem.add(MemoryBranch.INSIGHTS, {"insight": "fire is hot"})

    results = mem.query(branch=MemoryBranch.INSIGHTS)
    assert len(results) == 2

    results = mem.query(branch=MemoryBranch.SELF)
    assert len(results) == 1


def test_query_by_content_filter():
    """Query should filter by content keys."""
    mem = CrystallineMerkleMemory()

    mem.add(MemoryBranch.EXPERIENCES, {"type": "exploration", "detail": "a"})
    mem.add(MemoryBranch.EXPERIENCES, {"type": "conversation", "detail": "b"})
    mem.add(MemoryBranch.EXPERIENCES, {"type": "exploration", "detail": "c"})

    results = mem.query(content_filter={"type": "exploration"})
    assert len(results) == 2


def test_query_no_filter_returns_all():
    """Query with no filter should return all non-root nodes."""
    mem = CrystallineMerkleMemory()

    mem.add(MemoryBranch.SELF, {"a": 1})
    mem.add(MemoryBranch.INSIGHTS, {"b": 2})

    results = mem.query()
    assert len(results) == 2


def test_depth_increases():
    """Tree depth should be at least 2 after adding nodes."""
    mem = CrystallineMerkleMemory()
    assert mem.depth == 1  # Just roots

    mem.add(MemoryBranch.EXPERIENCES, {"event": "first"})
    assert mem.depth == 2  # Root -> child


def test_total_nodes_excludes_roots():
    """total_nodes should not count branch roots."""
    mem = CrystallineMerkleMemory()
    assert mem.total_nodes == 0
    assert len(mem.nodes) == 4  # 4 branch roots

    mem.add(MemoryBranch.SELF, {"claim": "x"})
    assert mem.total_nodes == 1
    assert len(mem.nodes) == 5


def test_no_tension_with_unrelated():
    """Completely different memories should not create tension."""
    mem = CrystallineMerkleMemory()

    mem.add(MemoryBranch.INSIGHTS, {"x": "alpha beta gamma"})
    mem.add(MemoryBranch.INSIGHTS, {"y": "one two three"})

    # Very low word overlap -> no tension
    assert len(mem.grain_boundaries) == 0


def test_consolidation_queue():
    """ConsolidationQueue should track pending items."""
    q = ConsolidationQueue()
    assert q.is_empty()

    node = MemoryNode(
        id="test",
        branch=MemoryBranch.SELF,
        content={"test": True},
        created_at="2024-01-01",
    )
    q.queue_node(node)
    assert not q.is_empty()

    q.queue_tension("a", "b", 0.5)
    assert len(q.pending_tensions) == 1

    q.clear()
    assert q.is_empty()


def test_verify_on_fresh_tree():
    """Fresh tree should verify successfully."""
    mem = CrystallineMerkleMemory()
    valid, msg = mem.verify()
    assert valid
    assert msg == "All nodes verified"


def test_consolidation_empty():
    """Consolidation with nothing to do should return zeros."""
    mem = CrystallineMerkleMemory()
    result = mem.consolidate()
    assert result["consolidated"] == 0
    assert result["tensions_resolved"] == 0
    assert result["remaining_tensions"] == 0


def test_get_state():
    """get_state() should return a complete dict."""
    mem = CrystallineMerkleMemory()
    mem.add(MemoryBranch.EXPERIENCES, {"event": "test"})

    state = mem.get_state()

    assert isinstance(state["root_hash"], str)
    assert state["total_nodes"] == 1
    assert state["depth"] >= 1
    assert isinstance(state["fractal_dimension"], float)
    assert isinstance(state["grain_boundaries"], int)
    assert isinstance(state["pending_consolidation"], int)
    assert state["verified"] is True


def test_fractal_dimension_property():
    """fractal_dimension property should match get_fractal_dimension()."""
    mem = CrystallineMerkleMemory()
    for i in range(10):
        mem.add(MemoryBranch.EXPERIENCES, {"event": f"e_{i}"})

    assert mem.fractal_dimension == mem.get_fractal_dimension()


def test_deferred_then_verify():
    """Deferred add + consolidate should still pass verification."""
    mem = CrystallineMerkleMemory()

    mem.add(MemoryBranch.SELF, {"claim": "immediate"}, immediate=True)
    mem.add(MemoryBranch.INSIGHTS, {"insight": "deferred"}, immediate=False)

    mem.consolidate()

    valid, msg = mem.verify()
    assert valid
    assert mem.total_nodes == 2
