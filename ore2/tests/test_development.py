"""Tests for DevelopmentTracker (ORE2-005)."""

import numpy as np
import pytest

from ore2.core.development import (
    CriticalPeriod,
    DevelopmentConfig,
    DevelopmentStage,
    DevelopmentTracker,
)


# ── Specified test cases from briefing ──────────────────────────────────────


def test_genesis_hash_unique():
    """Each entity should have unique genesis hash."""
    t1 = DevelopmentTracker()
    t2 = DevelopmentTracker()

    assert t1.genesis_hash != t2.genesis_hash


def test_initial_state():
    """Should start in GENESIS with minimal oscillators."""
    t = DevelopmentTracker()

    assert t.stage == DevelopmentStage.GENESIS
    assert t.current_oscillators == t.config.initial_oscillators
    assert t.age == 0.0


def test_stage_transition():
    """Should transition stages at correct ages."""
    config = DevelopmentConfig(genesis_duration=10.0)
    t = DevelopmentTracker(config)

    # Advance past genesis duration
    t.advance_age(15.0)
    result = t.process_experience({"type": "test"}, 0.5)

    assert result["stage_transition"] == DevelopmentStage.BABBLING
    assert t.stage == DevelopmentStage.BABBLING


def test_growth_trigger():
    """Should grow after N significant experiences."""
    config = DevelopmentConfig(growth_interval=5, initial_oscillators=10)
    t = DevelopmentTracker(config)

    # Process 4 significant experiences (not enough)
    for i in range(4):
        result = t.process_experience({"type": "test"}, 0.8)
        assert not result["growth_triggered"]

    # 5th should trigger growth
    result = t.process_experience({"type": "test"}, 0.8)
    assert result["growth_triggered"]
    assert t.current_oscillators > 10


def test_significance_threshold():
    """Only experiences with significance > 0.7 should count."""
    config = DevelopmentConfig(growth_interval=3)
    t = DevelopmentTracker(config)

    # Low significance - shouldn't count
    t.process_experience({"type": "test"}, 0.5)
    t.process_experience({"type": "test"}, 0.6)
    t.process_experience({"type": "test"}, 0.7)  # Exactly 0.7 = not > 0.7

    assert t.significant_experiences == 0

    # High significance
    t.process_experience({"type": "test"}, 0.8)
    assert t.significant_experiences == 1


def test_critical_period_multiplier():
    """Critical periods should enhance learning."""
    t = DevelopmentTracker()
    t.stage = DevelopmentStage.GENESIS

    # "pattern" learning has critical period in GENESIS
    mult = t.get_learning_multiplier("pattern")
    assert mult > 1.0

    # "social" learning doesn't have critical period in GENESIS
    mult = t.get_learning_multiplier("social")
    assert mult == 1.0

    # Move to IMITATION stage
    t.stage = DevelopmentStage.IMITATION
    mult = t.get_learning_multiplier("social")
    assert mult > 1.0


def test_mature_no_transition():
    """MATURE stage should never transition."""
    t = DevelopmentTracker()
    t.stage = DevelopmentStage.MATURE
    t._stage_start_age = 0.0

    t.advance_age(100000.0)  # Huge age
    result = t.process_experience({"type": "test"}, 0.5)

    assert result["stage_transition"] is None
    assert t.stage == DevelopmentStage.MATURE


def test_milestones_recorded():
    """Stage transitions and growth should be recorded."""
    config = DevelopmentConfig(genesis_duration=10.0, growth_interval=2)
    t = DevelopmentTracker(config)

    # Trigger growth
    t.process_experience({"type": "test"}, 0.8)
    t.process_experience({"type": "test"}, 0.8)

    # Trigger transition
    t.advance_age(15.0)
    t.process_experience({"type": "test"}, 0.5)

    assert len(t.milestones) >= 2
    types = [m["type"] for m in t.milestones]
    assert "growth" in types
    assert "stage_transition" in types


# ── Additional tests ────────────────────────────────────────────────────────


def test_genesis_hash_is_sha256():
    """Genesis hash should be 64 hex characters (SHA256)."""
    t = DevelopmentTracker()
    assert len(t.genesis_hash) == 64
    assert all(c in "0123456789abcdef" for c in t.genesis_hash)


def test_default_config():
    """DevelopmentTracker should work with default config."""
    t = DevelopmentTracker()
    assert t.config.initial_oscillators == 20
    assert t.config.max_oscillators == 200
    assert t.config.genesis_duration == 100.0
    assert len(t.config.critical_periods) == 4


def test_advance_age():
    """advance_age should accumulate correctly."""
    t = DevelopmentTracker()
    t.advance_age(10.0)
    t.advance_age(5.0)
    assert t.age == 15.0


def test_stage_progress_genesis():
    """Stage progress should reflect position in current stage."""
    config = DevelopmentConfig(genesis_duration=100.0)
    t = DevelopmentTracker(config)

    assert t.stage_progress == 0.0

    t.advance_age(50.0)
    assert abs(t.stage_progress - 0.5) < 0.001

    t.advance_age(50.0)
    assert abs(t.stage_progress - 1.0) < 0.001


def test_stage_progress_mature():
    """MATURE stage progress should always be 1.0."""
    t = DevelopmentTracker()
    t.stage = DevelopmentStage.MATURE

    assert t.stage_progress == 1.0


def test_stage_progress_capped():
    """Stage progress should not exceed 1.0."""
    config = DevelopmentConfig(genesis_duration=10.0)
    t = DevelopmentTracker(config)
    t.advance_age(100.0)

    assert t.stage_progress == 1.0


def test_full_stage_progression():
    """Should progress through all stages correctly."""
    config = DevelopmentConfig(
        genesis_duration=10.0,
        babbling_duration=10.0,
        imitation_duration=10.0,
        autonomy_duration=10.0,
    )
    t = DevelopmentTracker(config)

    expected_stages = [
        DevelopmentStage.BABBLING,
        DevelopmentStage.IMITATION,
        DevelopmentStage.AUTONOMY,
        DevelopmentStage.MATURE,
    ]

    for expected in expected_stages:
        t.advance_age(15.0)
        result = t.process_experience({"type": "test"}, 0.5)
        assert result["stage_transition"] == expected
        assert t.stage == expected


def test_growth_caps_at_max():
    """Oscillators should not exceed max_oscillators."""
    config = DevelopmentConfig(
        growth_interval=1,
        initial_oscillators=195,
        max_oscillators=200,
        growth_rate=0.1,
    )
    t = DevelopmentTracker(config)

    # Process many significant experiences
    for _ in range(20):
        t.process_experience({"type": "test"}, 0.8)

    assert t.current_oscillators <= config.max_oscillators


def test_no_growth_at_max():
    """should_grow should return False when at max oscillators."""
    config = DevelopmentConfig(
        growth_interval=1,
        initial_oscillators=200,
        max_oscillators=200,
    )
    t = DevelopmentTracker(config)
    t.significant_experiences = 1

    assert not t.should_grow()


def test_no_growth_without_significant():
    """should_grow should return False with zero significant experiences."""
    t = DevelopmentTracker()
    assert not t.should_grow()


def test_experiences_processed_count():
    """All experiences should be counted regardless of significance."""
    t = DevelopmentTracker()

    t.process_experience({"type": "a"}, 0.1)
    t.process_experience({"type": "b"}, 0.5)
    t.process_experience({"type": "c"}, 0.9)

    assert t.experiences_processed == 3
    assert t.significant_experiences == 1


def test_critical_period_is_active():
    """CriticalPeriod.is_active should check stage correctly."""
    period = CriticalPeriod("test", DevelopmentStage.BABBLING, "novelty", 2.0)

    assert not period.is_active(DevelopmentStage.GENESIS)
    assert period.is_active(DevelopmentStage.BABBLING)
    assert not period.is_active(DevelopmentStage.IMITATION)


def test_critical_period_sensitivity_values():
    """Default critical periods should have correct sensitivity values."""
    config = DevelopmentConfig()

    sensitivities = {p.learning_type: p.sensitivity for p in config.critical_periods}

    assert sensitivities["pattern"] == 3.0
    assert sensitivities["novelty"] == 2.5
    assert sensitivities["social"] == 2.0
    assert sensitivities["planning"] == 1.5


def test_get_state():
    """get_state() should return a complete dict."""
    t = DevelopmentTracker()
    t.process_experience({"type": "test"}, 0.8)
    t.advance_age(5.0)

    state = t.get_state()

    assert isinstance(state["genesis_hash"], str)
    assert state["stage"] == "genesis"
    assert state["age"] == 5.0
    assert isinstance(state["stage_progress"], float)
    assert state["current_oscillators"] == 20
    assert state["experiences_processed"] == 1
    assert state["significant_experiences"] == 1
    assert isinstance(state["milestones"], list)


def test_get_state_milestones_are_copy():
    """get_state milestones should be a copy, not a reference."""
    t = DevelopmentTracker()
    state = t.get_state()
    state["milestones"].append({"fake": True})

    assert len(t.milestones) == 0


def test_learning_multiplier_no_match():
    """Unknown learning type should return 1.0."""
    t = DevelopmentTracker()
    mult = t.get_learning_multiplier("unknown_type")
    assert mult == 1.0


def test_growth_amount():
    """Growth should add growth_rate * 10 oscillators."""
    config = DevelopmentConfig(
        growth_interval=1,
        initial_oscillators=20,
        growth_rate=0.5,
    )
    t = DevelopmentTracker(config)

    t.process_experience({"type": "test"}, 0.8)

    expected = 20 + int(0.5 * 10)  # 25
    assert t.current_oscillators == expected


def test_stage_transition_resets_progress():
    """After transition, stage_progress should reset near 0."""
    config = DevelopmentConfig(genesis_duration=10.0, babbling_duration=100.0)
    t = DevelopmentTracker(config)

    t.advance_age(10.0)
    t.process_experience({"type": "test"}, 0.5)

    assert t.stage == DevelopmentStage.BABBLING
    # Stage start age was set to current age, so progress should be 0
    assert t.stage_progress == 0.0


def test_milestone_growth_records_oscillator_count():
    """Growth milestone should record current oscillator count."""
    config = DevelopmentConfig(growth_interval=1, initial_oscillators=20)
    t = DevelopmentTracker(config)

    t.process_experience({"type": "test"}, 0.8)

    growth_milestones = [m for m in t.milestones if m["type"] == "growth"]
    assert len(growth_milestones) == 1
    assert growth_milestones[0]["oscillators"] == t.current_oscillators


def test_milestone_transition_records_stages():
    """Transition milestone should record from and to stages."""
    config = DevelopmentConfig(genesis_duration=5.0)
    t = DevelopmentTracker(config)

    t.advance_age(10.0)
    t.process_experience({"type": "test"}, 0.5)

    transition_milestones = [m for m in t.milestones if m["type"] == "stage_transition"]
    assert len(transition_milestones) == 1
    assert transition_milestones[0]["from"] == "genesis"
    assert transition_milestones[0]["to"] == "babbling"
