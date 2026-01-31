"""Tests for EntityPersistence (ORE2-007)."""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from ore2.core.development import DevelopmentStage
from ore2.core.entity import DevelopmentalEntity, EntityConfig, create_entity
from ore2.core.memory import MemoryBranch
from ore2.core.persistence import (
    CheckpointInfo,
    ContinuityError,
    EntityPersistence,
    PersistenceError,
    SaveResult,
    StateCorruptionError,
    VerificationResult,
)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def entity_with_experience():
    """Create an entity with some experiences."""
    entity = create_entity("test")
    entity.process_experience("hello", significance=0.6)
    return entity


# ── Specified test cases from briefing ──────────────────────────────────────


def test_save_load_roundtrip(tmp_dir):
    """Save and load should produce identical entity."""
    entity = create_entity("test")
    entity.process_experience("hello", significance=0.6)

    path = os.path.join(tmp_dir, "test_entity.ore2")

    # Save
    result = EntityPersistence.save(entity, path)
    assert result.verified

    # Load
    restored = EntityPersistence.load(path)

    # Verify identity
    assert restored.genesis_hash == entity.genesis_hash
    assert restored.memory.root_hash == entity.memory.root_hash
    assert restored.development.stage == entity.development.stage
    assert restored.memory.total_nodes == entity.memory.total_nodes


def test_tampering_detected(tmp_dir):
    """Modifying saved file should fail verification."""
    entity = create_entity("test")
    path = os.path.join(tmp_dir, "tamper_test.ore2")
    EntityPersistence.save(entity, path)

    # Tamper with file
    with open(path, "r") as f:
        data = json.load(f)
    data["state"]["name"] = "HACKED"
    with open(path, "w") as f:
        json.dump(data, f)

    # Verification should fail
    result = EntityPersistence.verify_file(path)
    assert not result.valid

    # Load should raise
    with pytest.raises(ContinuityError):
        EntityPersistence.load(path)


def test_checkpoint_lifecycle(tmp_dir):
    """Checkpoint create/list/restore should work."""
    entity = create_entity("test")
    cp_dir = os.path.join(tmp_dir, "checkpoints")

    # Create checkpoints
    cp1 = EntityPersistence.checkpoint(entity, cp_dir)
    entity.process_experience("experience 1", significance=0.8)
    cp2 = EntityPersistence.checkpoint(entity, cp_dir)

    # List
    checkpoints = EntityPersistence.list_checkpoints(cp_dir)
    assert len(checkpoints) >= 2

    # Restore older
    restored = EntityPersistence.restore_checkpoint(cp_dir, cp1)
    assert (
        restored.development.experiences_processed
        < entity.development.experiences_processed
    )


def test_genesis_hash_immutable(tmp_dir):
    """Genesis hash must survive save/load."""
    entity = create_entity("test")
    original_hash = entity.genesis_hash

    path = os.path.join(tmp_dir, "genesis_test.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert restored.genesis_hash == original_hash


# ── Additional tests ────────────────────────────────────────────────────────


def test_save_creates_directories(tmp_dir):
    """Save should create parent directories if needed."""
    entity = create_entity("test")
    path = os.path.join(tmp_dir, "deep", "nested", "dir", "entity.ore2")

    result = EntityPersistence.save(entity, path)
    assert result.verified
    assert os.path.exists(path)


def test_save_result_fields(tmp_dir):
    """SaveResult should have all expected fields."""
    entity = create_entity("test")
    path = os.path.join(tmp_dir, "test.ore2")

    result = EntityPersistence.save(entity, path)

    assert isinstance(result, SaveResult)
    assert result.path == path
    assert result.genesis_hash == entity.genesis_hash
    assert result.merkle_root == entity.memory.root_hash
    assert len(result.state_hash) == 64  # SHA256 hex
    assert result.timestamp  # Non-empty
    assert result.size_bytes > 0
    assert result.verified is True


def test_verify_file_valid(tmp_dir):
    """verify_file should return valid for untampered file."""
    entity = create_entity("test")
    path = os.path.join(tmp_dir, "verify.ore2")
    EntityPersistence.save(entity, path)

    result = EntityPersistence.verify_file(path)
    assert result.valid
    assert result.error is None
    assert len(result.genesis_hash) == 64
    assert len(result.state_hash) == 64


def test_verify_file_nonexistent(tmp_dir):
    """verify_file should return invalid for nonexistent file."""
    result = EntityPersistence.verify_file(os.path.join(tmp_dir, "nope.ore2"))
    assert not result.valid
    assert result.error is not None


def test_verify_file_invalid_json(tmp_dir):
    """verify_file should return invalid for non-JSON file."""
    path = os.path.join(tmp_dir, "bad.ore2")
    with open(path, "w") as f:
        f.write("not json")

    result = EntityPersistence.verify_file(path)
    assert not result.valid


def test_load_nonexistent(tmp_dir):
    """Loading nonexistent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        EntityPersistence.load(os.path.join(tmp_dir, "nope.ore2"))


def test_load_bad_version(tmp_dir):
    """Loading file with unsupported version should raise StateCorruptionError."""
    path = os.path.join(tmp_dir, "old.ore2")
    with open(path, "w") as f:
        json.dump({"version": "1.0", "state": {}, "verification": {}}, f)

    with pytest.raises(StateCorruptionError, match="Unsupported version"):
        EntityPersistence.load(path)


def test_roundtrip_preserves_substrate_state(tmp_dir):
    """Substrate phases and weights should survive roundtrip."""
    entity = create_entity("test")
    # Stimulate to get some non-trivial state
    entity.process_experience("test experience", significance=0.7)

    path = os.path.join(tmp_dir, "substrate.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    np.testing.assert_array_almost_equal(
        restored.substrate.fast.phases, entity.substrate.fast.phases
    )
    np.testing.assert_array_almost_equal(
        restored.substrate.slow.phases, entity.substrate.slow.phases
    )
    np.testing.assert_array_almost_equal(
        restored.substrate.fast.internal_weights,
        entity.substrate.fast.internal_weights,
    )
    np.testing.assert_array_almost_equal(
        restored.substrate.fast_to_slow, entity.substrate.fast_to_slow
    )
    assert restored.substrate.time == pytest.approx(entity.substrate.time)


def test_roundtrip_preserves_body_state(tmp_dir):
    """Body state should survive roundtrip."""
    entity = create_entity("test")
    entity.body.energy = 0.7
    entity.body.arousal = 0.8
    entity.tick()

    path = os.path.join(tmp_dir, "body.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert restored.body.energy == pytest.approx(entity.body.energy, abs=1e-6)
    assert restored.body.arousal == pytest.approx(entity.body.arousal, abs=1e-6)
    assert restored.body.heartbeat_phase == pytest.approx(
        entity.body.heartbeat_phase
    )
    assert restored.body.time == pytest.approx(entity.body.time)


def test_roundtrip_preserves_development_state(tmp_dir):
    """Development state should survive roundtrip."""
    entity = create_entity("test")
    entity.process_experience("exp", significance=0.8)

    path = os.path.join(tmp_dir, "dev.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert restored.development.stage == entity.development.stage
    assert restored.development.age == pytest.approx(entity.development.age)
    assert (
        restored.development.experiences_processed
        == entity.development.experiences_processed
    )
    assert (
        restored.development.significant_experiences
        == entity.development.significant_experiences
    )
    assert (
        restored.development.current_oscillators
        == entity.development.current_oscillators
    )
    assert len(restored.development.milestones) == len(
        entity.development.milestones
    )


def test_roundtrip_preserves_memory(tmp_dir):
    """Memory nodes and merkle root should survive roundtrip."""
    entity = create_entity("test")
    entity.process_experience("memory test", significance=0.9)
    entity.process_experience("another one", significance=0.4)

    path = os.path.join(tmp_dir, "memory.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert restored.memory.root_hash == entity.memory.root_hash
    assert restored.memory.total_nodes == entity.memory.total_nodes
    assert restored.memory.depth == entity.memory.depth
    assert len(restored.memory.nodes) == len(entity.memory.nodes)


def test_roundtrip_preserves_tick_count(tmp_dir):
    """Tick count should survive roundtrip."""
    entity = create_entity("test")
    for _ in range(5):
        entity.tick()

    path = os.path.join(tmp_dir, "ticks.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert restored._tick_count == entity._tick_count


def test_roundtrip_preserves_ci_history(tmp_dir):
    """CI history should survive roundtrip."""
    entity = create_entity("test")
    for _ in range(5):
        entity.tick()

    path = os.path.join(tmp_dir, "ci.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert len(restored.ci_monitor.history) == len(entity.ci_monitor.history)
    if entity.ci_monitor.history:
        assert restored.ci_monitor.history[-1].CI_integrated == pytest.approx(
            entity.ci_monitor.history[-1].CI_integrated
        )


def test_roundtrip_empty_entity(tmp_dir):
    """Fresh entity with no experiences should roundtrip."""
    entity = create_entity("fresh")

    path = os.path.join(tmp_dir, "fresh.ore2")
    result = EntityPersistence.save(entity, path)
    assert result.verified

    restored = EntityPersistence.load(path)
    assert restored.name == "fresh"
    assert restored.genesis_hash == entity.genesis_hash
    assert restored.memory.total_nodes == 0
    assert restored.stage == DevelopmentStage.GENESIS


def test_roundtrip_preserves_consolidation_queue(tmp_dir):
    """Pending consolidation queue should survive roundtrip."""
    entity = create_entity("test")
    # Low significance = queued
    entity.process_experience("queued item", significance=0.3)

    assert not entity.memory.consolidation_queue.is_empty()

    path = os.path.join(tmp_dir, "queue.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert not restored.memory.consolidation_queue.is_empty()
    assert len(restored.memory.consolidation_queue.pending_nodes) == len(
        entity.memory.consolidation_queue.pending_nodes
    )


def test_checkpoint_naming(tmp_dir):
    """Checkpoint ID should follow naming convention."""
    entity = create_entity("test")
    cp_dir = os.path.join(tmp_dir, "checkpoints")

    cp_id = EntityPersistence.checkpoint(entity, cp_dir)

    # Should start with first 8 chars of genesis hash
    assert cp_id.startswith(entity.genesis_hash[:8])
    # Should contain experience count at end
    assert cp_id.endswith("_0")


def test_list_checkpoints_empty(tmp_dir):
    """Listing empty directory should return empty list."""
    cp_dir = os.path.join(tmp_dir, "empty_checkpoints")
    result = EntityPersistence.list_checkpoints(cp_dir)
    assert result == []


def test_list_checkpoints_sorted(tmp_dir):
    """Checkpoints should be sorted newest first."""
    entity = create_entity("test")
    cp_dir = os.path.join(tmp_dir, "checkpoints")

    cp1 = EntityPersistence.checkpoint(entity, cp_dir)
    entity.process_experience("exp", significance=0.5)
    cp2 = EntityPersistence.checkpoint(entity, cp_dir)

    checkpoints = EntityPersistence.list_checkpoints(cp_dir)

    assert len(checkpoints) == 2
    # Newest first
    assert checkpoints[0].experience_count >= checkpoints[1].experience_count


def test_checkpoint_info_fields(tmp_dir):
    """CheckpointInfo should have all expected fields."""
    entity = create_entity("test")
    entity.process_experience("exp", significance=0.8)
    cp_dir = os.path.join(tmp_dir, "checkpoints")

    EntityPersistence.checkpoint(entity, cp_dir)
    checkpoints = EntityPersistence.list_checkpoints(cp_dir)

    info = checkpoints[0]
    assert isinstance(info, CheckpointInfo)
    assert isinstance(info.checkpoint_id, str)
    assert isinstance(info.timestamp, str)
    assert isinstance(info.age, float)
    assert isinstance(info.stage, str)
    assert isinstance(info.experience_count, int)
    assert isinstance(info.merkle_root, str)


def test_file_format_structure(tmp_dir):
    """Saved file should have correct JSON structure."""
    entity = create_entity("test")
    path = os.path.join(tmp_dir, "format.ore2")
    EntityPersistence.save(entity, path)

    with open(path, "r") as f:
        envelope = json.load(f)

    assert envelope["version"] == "2.0"
    assert "state" in envelope
    assert "verification" in envelope

    state = envelope["state"]
    assert "name" in state
    assert "genesis_hash" in state
    assert "substrate" in state
    assert "body" in state
    assert "memory" in state
    assert "development" in state
    assert "ci_history" in state
    assert "tick_count" in state

    verification = envelope["verification"]
    assert "genesis_hash" in verification
    assert "merkle_root" in verification
    assert "state_hash" in verification
    assert "saved_at" in verification


def test_exceptions_hierarchy():
    """Exception classes should have correct inheritance."""
    assert issubclass(ContinuityError, PersistenceError)
    assert issubclass(StateCorruptionError, PersistenceError)
    assert issubclass(PersistenceError, Exception)


def test_roundtrip_with_rest(tmp_dir):
    """Entity that has rested should roundtrip correctly."""
    entity = create_entity("test")
    entity.process_experience("before rest", significance=0.4)
    entity.rest(duration=1.0)

    path = os.path.join(tmp_dir, "rested.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert restored.genesis_hash == entity.genesis_hash
    assert restored.memory.total_nodes == entity.memory.total_nodes
    # Consolidation queue should be empty after rest
    assert restored.memory.consolidation_queue.is_empty()


def test_roundtrip_preserves_name(tmp_dir):
    """Entity name should survive roundtrip."""
    entity = create_entity("MySpecialEntity")

    path = os.path.join(tmp_dir, "name.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    assert restored.name == "MySpecialEntity"


def test_roundtrip_preserves_critical_periods(tmp_dir):
    """Critical periods in development config should survive roundtrip."""
    entity = create_entity("test")

    path = os.path.join(tmp_dir, "cp.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    orig_periods = entity.development.config.critical_periods
    rest_periods = restored.development.config.critical_periods

    assert len(rest_periods) == len(orig_periods)
    for orig, rest in zip(orig_periods, rest_periods):
        assert rest.name == orig.name
        assert rest.stage == orig.stage
        assert rest.learning_type == orig.learning_type
        assert rest.sensitivity == orig.sensitivity


def test_entity_continues_after_load(tmp_dir):
    """Loaded entity should be able to continue processing."""
    entity = create_entity("test")
    entity.process_experience("first", significance=0.6)

    path = os.path.join(tmp_dir, "continue.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    # Should be able to tick and process more experiences
    restored.tick()
    result = restored.process_experience("second", significance=0.7)

    assert "CI" in result
    assert restored.development.experiences_processed == 2


def test_witness_after_load(tmp_dir):
    """witness() should work on loaded entity."""
    entity = create_entity("WitnessTest")
    entity.process_experience("hello", significance=0.6)

    path = os.path.join(tmp_dir, "witness.ore2")
    EntityPersistence.save(entity, path)
    restored = EntityPersistence.load(path)

    output = restored.witness()
    assert "WitnessTest" in output
    assert "ENTITY" in output
