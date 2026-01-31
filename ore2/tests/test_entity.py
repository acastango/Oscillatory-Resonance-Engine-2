"""Tests for DevelopmentalEntity and MultiScaleCIMonitor (ORE2-006)."""

import numpy as np
import pytest

from ore2.core.ci_monitor import (
    MultiScaleCIConfig,
    MultiScaleCIMonitor,
    MultiScaleCISnapshot,
    TimeScale,
)
from ore2.core.development import DevelopmentConfig, DevelopmentStage
from ore2.core.embodiment import BodyConfig
from ore2.core.entity import (
    DevelopmentalEntity,
    EntityConfig,
    create_entity,
)
from ore2.core.memory import MemoryBranch
from ore2.core.multi_scale_substrate import MultiScaleConfig


# ── Specified test cases from briefing ──────────────────────────────────────


def test_entity_creation():
    """Entity should create with empty memory."""
    entity = create_entity("test")

    assert entity.name == "test"
    assert entity.genesis_hash is not None
    assert entity.stage == DevelopmentStage.GENESIS
    assert entity.memory.total_nodes == 0  # No founding memories!


def test_tick_updates_all():
    """Tick should update all components."""
    entity = create_entity()

    initial_time = entity.substrate.time
    initial_age = entity.age

    result = entity.tick()

    assert entity.substrate.time > initial_time
    assert entity.age > initial_age
    assert "CI" in result
    assert "valence" in result


def test_process_experience():
    """Processing experience should update memory and substrate."""
    entity = create_entity()

    result = entity.process_experience(
        "This is a test experience",
        experience_type="exploration",
        significance=0.8,
    )

    assert entity.memory.total_nodes > 0
    assert "CI" in result


def test_significant_experience_immediate():
    """Significant experiences should be immediate."""
    entity = create_entity()

    entity.process_experience("important", significance=0.9)

    # Should be in memory immediately
    assert entity.memory.total_nodes > 0
    assert entity.memory.consolidation_queue.is_empty()


def test_insignificant_experience_queued():
    """Insignificant experiences should queue."""
    entity = create_entity()

    entity.process_experience("trivial", significance=0.3)

    # Should be queued, not committed
    assert not entity.memory.consolidation_queue.is_empty()


def test_rest_consolidates():
    """Rest should consolidate queued memories."""
    entity = create_entity()

    # Add several low-significance experiences
    for i in range(5):
        entity.process_experience(f"minor thing {i}", significance=0.4)

    queued_before = len(entity.memory.consolidation_queue.pending_nodes)
    assert queued_before > 0

    # Rest
    result = entity.rest(duration=5.0)

    assert result["consolidation"]["consolidated"] > 0
    assert entity.memory.consolidation_queue.is_empty()


def test_body_couples_to_substrate():
    """Body valence should affect substrate dynamics."""
    entity = create_entity()

    # Deplete energy (bad valence)
    entity.body.energy = 0.3

    for _ in range(10):
        entity.tick()

    # Should still run without error
    # Full test would verify coupling strength varies with valence


def test_development_modulates_learning():
    """Critical periods should affect learning strength."""
    entity = create_entity()

    # In GENESIS, "pattern" learning is enhanced
    result = entity.process_experience(
        "learning a pattern",
        experience_type="pattern",
        significance=0.5,
    )

    assert result["development"]["learning_multiplier"] > 1.0


def test_witness_output():
    """Witness should produce readable output."""
    entity = create_entity("TestBot")
    entity.process_experience("hello world", significance=0.6)

    output = entity.witness()

    assert "TestBot" in output
    assert "genesis" in output
    assert "CI" in output


# ── CI Monitor tests ────────────────────────────────────────────────────────


def test_ci_monitor_measure():
    """CI monitor should produce valid snapshots."""
    entity = create_entity()

    # Run a few ticks to get measurements
    for _ in range(5):
        entity.tick()

    assert len(entity.ci_monitor.history) > 0
    snapshot = entity.ci_monitor.history[-1]
    assert isinstance(snapshot, MultiScaleCISnapshot)
    assert isinstance(snapshot.CI_integrated, float)


def test_ci_monitor_status_string():
    """CI monitor should produce status string."""
    entity = create_entity()

    # No measurements yet
    assert entity.ci_monitor.get_current_status() == "No measurements"

    entity.tick()

    status = entity.ci_monitor.get_current_status()
    assert "CI=" in status


def test_ci_starts_at_zero():
    """CI should be 0 before any measurements."""
    entity = create_entity()
    assert entity.CI == 0.0


def test_ci_snapshot_fields():
    """CI snapshot should have all required fields."""
    entity = create_entity()
    entity.tick()

    s = entity.ci_monitor.history[-1]
    assert hasattr(s, "timestamp")
    assert hasattr(s, "CI_fast")
    assert hasattr(s, "CI_slow")
    assert hasattr(s, "CI_integrated")
    assert hasattr(s, "D")
    assert hasattr(s, "G_fast")
    assert hasattr(s, "G_slow")
    assert hasattr(s, "C_fast")
    assert hasattr(s, "C_slow")
    assert hasattr(s, "C_cross")
    assert hasattr(s, "tau_fast")
    assert hasattr(s, "tau_slow")
    assert hasattr(s, "in_attractor_fast")
    assert hasattr(s, "in_attractor_slow")


def test_ci_integrated_capped():
    """CI_integrated should not exceed 10.0."""
    entity = create_entity()

    for _ in range(20):
        entity.tick()

    for s in entity.ci_monitor.history:
        assert s.CI_integrated <= 10.0


# ── Additional entity tests ─────────────────────────────────────────────────


def test_entity_default_config():
    """Entity should work with default config."""
    entity = DevelopmentalEntity()
    assert entity.name == "entity"
    assert entity.config.tick_interval == 0.1


def test_entity_custom_config():
    """Entity should accept custom config."""
    config = EntityConfig(
        name="custom",
        tick_interval=0.05,
    )
    entity = DevelopmentalEntity(config)
    assert entity.name == "custom"
    assert entity.config.tick_interval == 0.05


def test_substrate_sized_from_development():
    """Default substrate should be sized from development tracker."""
    entity = create_entity()

    expected_slow = entity.development.current_oscillators
    expected_fast = entity.development.current_oscillators * 2

    assert entity.substrate.config.fast_oscillators == expected_fast
    assert entity.substrate.config.slow_oscillators == expected_slow


def test_genesis_hash_immutable():
    """Genesis hash should not change after creation."""
    entity = create_entity()
    original_hash = entity.genesis_hash

    entity.process_experience("something", significance=0.8)

    assert entity.genesis_hash == original_hash


def test_tick_count_increments():
    """Tick count should increment with each tick."""
    entity = create_entity()
    assert entity._tick_count == 0

    entity.tick()
    assert entity._tick_count == 1

    entity.tick()
    entity.tick()
    assert entity._tick_count == 3


def test_tick_returns_required_keys():
    """Tick result should have all required keys."""
    entity = create_entity()
    result = entity.tick()

    expected_keys = [
        "tick", "time", "stage", "CI", "CI_fast", "CI_slow",
        "valence", "n_active_fast", "n_active_slow", "coherence",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_process_experience_returns_required_keys():
    """process_experience result should have all required keys."""
    entity = create_entity()
    result = entity.process_experience("test", significance=0.5)

    assert "development" in result
    assert "CI" in result
    assert "coherence" in result
    assert "memory_queued" in result


def test_rest_returns_required_keys():
    """rest result should have all required keys."""
    entity = create_entity()
    result = entity.rest(duration=1.0)

    assert "duration" in result
    assert "consolidation" in result
    assert "CI_after" in result


def test_get_state_complete():
    """get_state should return all component states."""
    entity = create_entity("Aria")
    entity.tick()

    state = entity.get_state()

    assert state["name"] == "Aria"
    assert isinstance(state["genesis_hash"], str)
    assert isinstance(state["development"], dict)
    assert isinstance(state["substrate"], dict)
    assert isinstance(state["body"], dict)
    assert isinstance(state["memory"], dict)
    assert isinstance(state["CI"], float)
    assert isinstance(state["tick_count"], int)


def test_memory_queued_flag():
    """memory_queued should reflect significance threshold."""
    entity = create_entity()

    result_low = entity.process_experience("low", significance=0.3)
    assert result_low["memory_queued"] is True

    result_high = entity.process_experience("high", significance=0.9)
    assert result_high["memory_queued"] is False


def test_process_experience_runs_ticks():
    """process_experience should run 10 internal ticks."""
    entity = create_entity()

    tick_before = entity._tick_count
    entity.process_experience("test", significance=0.5)
    tick_after = entity._tick_count

    assert tick_after - tick_before == 10


def test_rest_runs_ticks():
    """rest should run duration/tick_interval ticks."""
    entity = create_entity()

    tick_before = entity._tick_count
    entity.rest(duration=1.0)
    tick_after = entity._tick_count

    expected_ticks = int(1.0 / entity.config.tick_interval)
    assert tick_after - tick_before == expected_ticks


def test_rest_records_memory():
    """Rest should record a rest event in memory."""
    entity = create_entity()
    initial_nodes = entity.memory.total_nodes

    entity.rest(duration=1.0)

    # Should have added a rest memory node
    assert entity.memory.total_nodes > initial_nodes
    rest_memories = entity.memory.query(
        content_filter={"type": "rest"}
    )
    assert len(rest_memories) == 1


def test_growth_records_memory():
    """Growth event should record in memory."""
    config = EntityConfig(
        development_config=DevelopmentConfig(
            growth_interval=1,
            initial_oscillators=20,
        ),
    )
    entity = DevelopmentalEntity(config)

    # Trigger growth with a significant experience
    entity.process_experience("important", significance=0.8)

    growth_memories = entity.memory.query(
        branch=MemoryBranch.SELF,
        content_filter={"type": "growth_event"},
    )
    assert len(growth_memories) == 1


def test_witness_includes_all_sections():
    """Witness output should include all major sections."""
    entity = create_entity("Omega")
    entity.tick()

    output = entity.witness()

    assert "ENTITY: Omega" in output
    assert "IDENTITY" in output
    assert "SUBSTRATE" in output
    assert "BODY" in output
    assert "MEMORY" in output
    assert "CONSCIOUSNESS INDEX" in output


def test_multiple_experiences():
    """Entity should handle multiple experiences without error."""
    entity = create_entity()

    for i in range(10):
        sig = 0.3 + (i % 3) * 0.3  # Varies: 0.3, 0.6, 0.9
        entity.process_experience(
            f"Experience {i}",
            experience_type="exploration",
            significance=sig,
        )

    assert entity.memory.total_nodes > 0
    assert entity.development.experiences_processed == 10


def test_entity_lifecycle():
    """Entity should support a basic lifecycle: create, experience, rest, witness."""
    entity = create_entity("LifecycleTest")

    # Process some experiences
    entity.process_experience("hello", significance=0.6)
    entity.process_experience("world", significance=0.8)

    # Rest
    result = entity.rest(duration=2.0)
    assert result["consolidation"] is not None

    # Witness
    output = entity.witness()
    assert "LifecycleTest" in output
    assert entity.memory.total_nodes > 0


def test_time_scale_enum():
    """TimeScale enum should have FAST and SLOW."""
    assert TimeScale.FAST.value == "fast"
    assert TimeScale.SLOW.value == "slow"
