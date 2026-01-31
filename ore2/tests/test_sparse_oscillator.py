"""Tests for SparseOscillatorLayer (ORE2-001)."""

import time

import numpy as np
import pytest

from ore2.core.sparse_oscillator import SparseOscillatorConfig, SparseOscillatorLayer


# ── Specified test cases from briefing ──────────────────────────────────────


def test_sparse_basic():
    """Basic sparse behavior."""
    config = SparseOscillatorConfig(
        n_oscillators=100, activation_threshold=0.5, activation_decay=0.005
    )
    layer = SparseOscillatorLayer("test", config)

    # Initially all dormant
    assert layer.n_active == 0
    assert layer.coherence == 0.0

    # Stimulate some oscillators
    layer.stimulate(np.array([0, 1, 2, 3, 4]), np.array([0.6, 0.6, 0.6, 0.6, 0.6]))
    assert layer.n_active == 5

    # Step and check coherence emerges (re-stimulate to keep active during sync)
    for _ in range(100):
        layer.stimulate(np.array([0, 1, 2, 3, 4]), np.array([0.01, 0.01, 0.01, 0.01, 0.01]))
        layer.step(0.01)

    assert layer.coherence > 0.3  # Should synchronize

    # Let activation decay (no new stimulation)
    for _ in range(500):
        layer.step(0.01)

    assert layer.n_active < 5  # Should have decayed


def test_sparse_performance():
    """Sparse should be faster than dense."""
    config = SparseOscillatorConfig(
        n_oscillators=1000, activation_decay=0.0, max_active_fraction=1.0
    )
    layer = SparseOscillatorLayer("perf", config)

    # Activate 10% (decay=0 keeps them active for fair comparison)
    layer.stimulate(np.arange(100), np.ones(100) * 0.8)
    assert layer.n_active == 100

    start = time.time()
    for _ in range(1000):
        layer.step(0.01)
    sparse_time = time.time() - start

    # Compare to dense (activate all)
    layer.activation_potentials[:] = 1.0
    layer._update_active_mask()
    assert layer.n_active == 1000

    start = time.time()
    for _ in range(1000):
        layer.step(0.01)
    dense_time = time.time() - start

    assert sparse_time < dense_time * 0.5  # At least 2x faster


def test_coherence_matches_dense():
    """When all active, should match ORE1 behavior."""
    config = SparseOscillatorConfig(n_oscillators=50, activation_threshold=0.0)
    layer = SparseOscillatorLayer("dense_compare", config)

    # Force all active
    layer.activation_potentials[:] = 1.0
    layer._update_active_mask()

    # Set coherent initial state
    base_phase = np.random.uniform(0, 2 * np.pi)
    layer.phases = (base_phase + 0.1 * np.random.randn(50)) % (2 * np.pi)

    # Run and check coherence stays high
    for _ in range(100):
        layer.step(0.01)

    assert layer.coherence > 0.8  # Should maintain coherence


# ── Edge cases from briefing ────────────────────────────────────────────────


def test_zero_active_no_crash():
    """Zero active oscillators: coherence = 0, step() should not crash."""
    config = SparseOscillatorConfig(n_oscillators=50)
    layer = SparseOscillatorLayer("empty", config)

    assert layer.n_active == 0
    assert layer.coherence == 0.0
    assert layer.mean_phase == 0.0
    assert layer.phase_hash == "2e1cfa82b035c26c"  # sha256 of b"empty" prefix

    # Step should not crash
    for _ in range(10):
        layer.step(0.01)

    assert layer.n_active == 0


def test_all_active_dense_behavior():
    """All oscillators active should behave like dense Kuramoto."""
    config = SparseOscillatorConfig(
        n_oscillators=30,
        activation_threshold=0.0,
        max_active_fraction=1.0,
    )
    layer = SparseOscillatorLayer("all_active", config)

    # Force all active
    layer.activation_potentials[:] = 1.0
    layer._update_active_mask()
    assert layer.n_active == 30

    # Run dynamics
    for _ in range(200):
        layer.step(0.01)

    # Should produce valid coherence
    assert 0.0 <= layer.coherence <= 0.999
    assert 0.0 <= layer.global_coherence <= 0.999


def test_max_active_cap():
    """Max active cap should keep only highest activations."""
    config = SparseOscillatorConfig(
        n_oscillators=100,
        activation_threshold=0.3,
        max_active_fraction=0.1,  # max 10
    )
    layer = SparseOscillatorLayer("capped", config)

    # Stimulate 50 oscillators above threshold
    layer.stimulate(np.arange(50), np.ones(50) * 0.8)

    # Should be capped at 10
    assert layer.n_active == 10

    # The active ones should be from the 50 we stimulated
    active_idx = np.where(layer.active_mask)[0]
    assert all(idx < 50 for idx in active_idx)


def test_external_input_only_affects_active():
    """External input in step() should only affect active oscillators."""
    config = SparseOscillatorConfig(n_oscillators=20)
    layer = SparseOscillatorLayer("ext_input", config)

    # Activate just first 5
    layer.stimulate(np.arange(5), np.ones(5) * 0.8)

    # Record dormant phases
    dormant_phases_before = layer.phases[5:].copy()

    # External input for all oscillators
    ext = np.ones(20) * 10.0
    layer.step(0.01, external_input=ext)

    # Dormant oscillators should only have drifted by natural frequency
    dormant_phases_after = layer.phases[5:]
    expected_drift = (
        dormant_phases_before + 0.01 * layer.natural_frequencies[5:]
    ) % (2 * np.pi)
    np.testing.assert_allclose(dormant_phases_after, expected_drift, atol=1e-10)


def test_phase_wraparound():
    """Phases should always remain in [0, 2pi)."""
    config = SparseOscillatorConfig(n_oscillators=10)
    layer = SparseOscillatorLayer("wrap", config)

    # Force high frequencies for large phase changes
    layer.natural_frequencies[:] = 1000.0
    layer.activation_potentials[:] = 1.0
    layer._update_active_mask()

    layer.step(1.0)  # Large dt

    assert np.all(layer.phases >= 0)
    assert np.all(layer.phases < 2 * np.pi)


# ── Properties ──────────────────────────────────────────────────────────────


def test_coherence_capped_at_0999():
    """Coherence should never exceed 0.999."""
    config = SparseOscillatorConfig(n_oscillators=10, noise_amplitude=0.0)
    layer = SparseOscillatorLayer("cap_test", config)

    # Set all phases identical -> perfect coherence
    layer.phases[:] = 1.0
    layer.activation_potentials[:] = 1.0
    layer._update_active_mask()

    assert layer.coherence <= 0.999
    assert layer.global_coherence <= 0.999


def test_phase_hash_changes_with_active_phases():
    """Phase hash should change when active phases change."""
    config = SparseOscillatorConfig(n_oscillators=20)
    layer = SparseOscillatorLayer("hash_test", config)

    layer.stimulate(np.arange(5), np.ones(5) * 0.8)
    hash1 = layer.phase_hash

    # Step to change phases
    layer.stimulate(np.arange(5), np.ones(5) * 0.8)  # keep active
    layer.step(0.1)
    hash2 = layer.phase_hash

    assert hash1 != hash2


def test_stimulate_by_similarity():
    """Stimulate by similarity should activate oscillators with similar phases."""
    config = SparseOscillatorConfig(n_oscillators=100, activation_threshold=0.3)
    layer = SparseOscillatorLayer("sim_test", config)

    # Set known phases
    layer.phases[:50] = 1.0   # First half at phase 1.0
    layer.phases[50:] = 4.0   # Second half at phase 4.0

    # Reference pattern matching first half
    reference = np.full(100, 1.0)
    layer.stimulate_by_similarity(reference, strength=0.8)

    # First half should have higher activation than second half
    assert np.mean(layer.activation_potentials[:50]) > np.mean(layer.activation_potentials[50:])


def test_get_state_returns_dict():
    """get_state() should return a complete state dict."""
    config = SparseOscillatorConfig(n_oscillators=10)
    layer = SparseOscillatorLayer("state_test", config)

    state = layer.get_state()

    assert state["name"] == "state_test"
    assert state["n"] == 10
    assert state["n_active"] == 0
    assert isinstance(state["coherence"], float)
    assert isinstance(state["global_coherence"], float)
    assert isinstance(state["mean_phase"], float)
    assert isinstance(state["phase_hash"], str)
    assert len(state["phases"]) == 10
    assert len(state["activation_potentials"]) == 10
    assert len(state["active_mask"]) == 10


def test_default_config():
    """Layer should work with default config when None is passed."""
    layer = SparseOscillatorLayer("default")
    assert layer.n == 100
    assert layer.n_active == 0


def test_activation_decay():
    """Activation potentials should decay each step."""
    config = SparseOscillatorConfig(n_oscillators=10, activation_decay=0.1)
    layer = SparseOscillatorLayer("decay_test", config)

    layer.stimulate(np.array([0]), np.array([0.9]))
    initial = layer.activation_potentials[0]

    layer.step(0.01)
    after_one = layer.activation_potentials[0]

    assert after_one < initial
    np.testing.assert_allclose(after_one, initial * 0.9, atol=1e-10)
