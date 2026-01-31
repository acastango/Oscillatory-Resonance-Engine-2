"""Tests for EmbodimentLayer (ORE2-003)."""

import numpy as np
import pytest

from ore2.core.embodiment import BodyConfig, EmbodimentLayer


# ── Specified test cases from briefing ──────────────────────────────────────


def test_rhythm_advance():
    """Rhythms should advance at correct frequency."""
    body = EmbodimentLayer(BodyConfig(heartbeat_freq=1.0))

    initial_phase = body.heartbeat_phase
    body.step(dt=0.25)  # Quarter second

    # Should have advanced by pi/2 (quarter cycle at 1Hz)
    expected_advance = 2 * np.pi * 1.0 * 0.25
    actual_advance = (body.heartbeat_phase - initial_phase) % (2 * np.pi)

    assert abs(actual_advance - expected_advance) < 0.01


def test_valence_at_baseline():
    """Valence should be 0 at baseline."""
    body = EmbodimentLayer()
    body.energy = body.config.energy_baseline
    body.arousal = body.config.arousal_baseline

    assert abs(body.valence) < 0.001


def test_valence_deviation():
    """Valence should be negative when deviated."""
    body = EmbodimentLayer()
    body.energy = 0.5  # Below baseline of 1.0
    body.arousal = 0.5  # At baseline

    assert body.valence < 0


def test_action_costs_energy():
    """Actions should reduce energy."""
    body = EmbodimentLayer()
    initial_energy = body.energy

    body.step(0.1, action="do_something")

    assert body.energy < initial_energy


def test_novel_perception_raises_arousal():
    """Novel perceptions should increase arousal."""
    body = EmbodimentLayer()
    initial_arousal = body.arousal

    body.step(0.1, perception="something_new")

    assert body.arousal > initial_arousal

    # Same perception again should NOT raise arousal
    current_arousal = body.arousal
    body.step(0.1, perception="something_new")

    assert body.arousal <= current_arousal  # May have decayed


def test_recovery():
    """Should recover toward baseline over time."""
    body = EmbodimentLayer()
    body.energy = 0.5  # Below baseline

    for _ in range(100):
        body.step(0.1)

    # Should have recovered toward baseline
    assert body.energy > 0.5
    assert body.energy <= body.config.energy_baseline


def test_coupling_signal_shape():
    """Coupling signal should match input shape."""
    body = EmbodimentLayer()
    phases = np.random.uniform(0, 2 * np.pi, 50)

    signal = body.get_cognitive_coupling_signal(phases)

    assert signal.shape == phases.shape


def test_coupling_modulated_by_valence():
    """Coupling should be weaker with bad valence."""
    body = EmbodimentLayer()
    phases = np.random.uniform(0, 2 * np.pi, 50)

    # Good state
    body.energy = body.config.energy_baseline
    body.arousal = body.config.arousal_baseline
    good_signal = body.get_cognitive_coupling_signal(phases)

    # Bad state
    body.energy = 0.3
    bad_signal = body.get_cognitive_coupling_signal(phases)

    # Bad state should have weaker coupling
    assert np.mean(np.abs(bad_signal)) < np.mean(np.abs(good_signal))


# ── Additional tests ────────────────────────────────────────────────────────


def test_default_config():
    """EmbodimentLayer should work with default config."""
    body = EmbodimentLayer()
    assert body.energy == 1.0
    assert body.arousal == 0.5
    assert body.time == 0.0


def test_body_signal_shape():
    """Body signal should be length 4."""
    body = EmbodimentLayer()
    assert body.body_signal.shape == (4,)


def test_body_signal_range():
    """Body signal components should be in [-1, 1]."""
    body = EmbodimentLayer()
    for _ in range(20):
        body.step(0.1)
    signal = body.body_signal
    assert np.all(signal >= -1.0)
    assert np.all(signal <= 1.0)


def test_respiration_advance():
    """Respiration should advance at its own frequency."""
    body = EmbodimentLayer(BodyConfig(respiration_freq=0.25))

    initial_phase = body.respiration_phase
    body.step(dt=1.0)  # One second

    # Should have advanced by pi/2 (quarter cycle at 0.25Hz)
    expected_advance = 2 * np.pi * 0.25 * 1.0
    actual_advance = (body.respiration_phase - initial_phase) % (2 * np.pi)

    assert abs(actual_advance - expected_advance) < 0.01


def test_is_depleted():
    """is_depleted should trigger when energy < 0.2."""
    body = EmbodimentLayer()
    assert not body.is_depleted

    body.energy = 0.1
    assert body.is_depleted

    body.energy = 0.2
    assert not body.is_depleted


def test_is_overaroused():
    """is_overaroused should trigger when arousal > 0.8."""
    body = EmbodimentLayer()
    assert not body.is_overaroused

    body.arousal = 0.9
    assert body.is_overaroused

    body.arousal = 0.8
    assert not body.is_overaroused


def test_energy_depletion_from_actions():
    """Extended activity without rest should deplete energy."""
    body = EmbodimentLayer(BodyConfig(energy_recovery=0.0))

    for _ in range(200):
        body.step(0.1, action="work")

    assert body.energy < 0.5
    assert body.valence < -0.3


def test_arousal_from_novel_input():
    """Constant novel input should raise arousal."""
    body = EmbodimentLayer(BodyConfig(arousal_recovery=0.0))

    for i in range(10):
        body.step(0.1, perception=f"novel_{i}")

    assert body.arousal > body.config.arousal_baseline
    assert body.is_overaroused


def test_idle_recovery():
    """Idling should recover both energy and arousal to baseline."""
    body = EmbodimentLayer()
    body.energy = 0.3
    body.arousal = 0.9

    for _ in range(1000):
        body.step(0.1)

    np.testing.assert_allclose(body.energy, body.config.energy_baseline, atol=0.05)
    np.testing.assert_allclose(body.arousal, body.config.arousal_baseline, atol=0.05)


def test_energy_clamp():
    """Energy should be clamped to [0, 2]."""
    body = EmbodimentLayer()
    body.energy = 2.5
    body.step(0.1)
    assert body.energy <= 2.0

    body.energy = -1.0
    body.step(0.1)
    assert body.energy >= 0.0


def test_arousal_clamp():
    """Arousal should be clamped to [0, 1]."""
    body = EmbodimentLayer()
    body.arousal = 1.5
    body.step(0.1)
    assert body.arousal <= 1.0


def test_get_state():
    """get_state() should return a complete dict."""
    body = EmbodimentLayer()
    state = body.get_state()

    assert isinstance(state["time"], float)
    assert isinstance(state["heartbeat_phase"], float)
    assert isinstance(state["respiration_phase"], float)
    assert isinstance(state["energy"], float)
    assert isinstance(state["arousal"], float)
    assert isinstance(state["valence"], float)
    assert isinstance(state["is_depleted"], bool)
    assert isinstance(state["is_overaroused"], bool)
    assert len(state["body_signal"]) == 4


def test_time_advances():
    """Time should accumulate across steps."""
    body = EmbodimentLayer()
    body.step(0.1)
    body.step(0.1)
    body.step(0.1)
    np.testing.assert_allclose(body.time, 0.3)
