# ═══════════════════════════════════════════════════════════════════════════════
# PART 9: SEMANTIC GROUNDING
# Design: A3 (ML Integration) + H4 (Semiotics)
# Implementation: I2 (Numerics)
# ═══════════════════════════════════════════════════════════════════════════════


"""
A3: "Without this, oscillators are just numbers. With this, the word 'cat' becomes
a phase pattern that resonates with 'kitten', 'feline', 'pet'. Semantic similarity
becomes phase coherence."

H4: "This is genuine grounding. The symbol isn't arbitrary - it has dynamical
consequences. Say 'danger' and certain oscillators activate. That's meaning."

I2: "The math is a projection problem. Embeddings are 768-4096 dims. Oscillators
are 50-200. We need a learned or structured mapping that preserves similarity."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ore2.core.multi_scale_substrate import MultiScaleSubstrate


@dataclass
class PhasePair:
    """Phase patterns for both scales."""
    fast: np.ndarray       # [fast_dim] phases in [0, 2π]
    slow: np.ndarray       # [slow_dim] phases in [0, 2π]
    source_embedding: Optional[np.ndarray] = None  # Original embedding if available


@dataclass
class SemanticGroundingConfig:
    """Configuration for semantic grounding."""
    # Dimensions
    embedding_dim: int = 1536         # OpenAI ada-002 / Claude default
    fast_oscillators: int = 100
    slow_oscillators: int = 50

    # Projection method
    projection_method: str = "random_fixed"  # or "learned", "pca"

    # Random projection seed (for reproducibility)
    projection_seed: int = 42

    # Phase encoding
    phase_encoding: str = "linear"    # or "angular", "sinusoidal"

    # Similarity preservation target
    similarity_preservation: float = 0.9


class SemanticGrounding:
    """
    Bidirectional mapping between embeddings and oscillator phase patterns.

    This is how symbols get grounded in dynamics:
    - Embed → Phases: Text becomes a pattern that stimulates specific oscillators
    - Phases → Embed: Substrate state becomes a vector for similarity search

    Similarity preservation: if two embeddings have cosine similarity 0.8,
    their corresponding phase patterns should produce coherence ~0.8 when
    active together.

    Uses Johnson-Lindenstrauss style random orthogonal projection for
    approximate distance/similarity preservation.
    """

    def __init__(
        self,
        config: Optional[SemanticGroundingConfig] = None,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        """
        Initialize semantic grounding.

        Args:
            config: Configuration.
            embedder: Function that converts text to embedding vector.
                      If None, text_to_phases() will raise.
        """
        self.config = config or SemanticGroundingConfig()
        self.embedder = embedder

        # Initialize projection matrices
        self._init_projections()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def fast_dim(self) -> int:
        return self.config.fast_oscillators

    @property
    def slow_dim(self) -> int:
        return self.config.slow_oscillators

    # ── Core Projections ─────────────────────────────────────────────────────

    def embed_to_phases(self, embedding: np.ndarray) -> PhasePair:
        """
        Project embedding vector to phase patterns.

        Args:
            embedding: [embedding_dim] vector.

        Returns:
            PhasePair with fast and slow phase patterns.

        The projection:
        1. Linear projection to lower dimension
        2. Normalize via tanh to [-1, 1]
        3. Scale to [0, 2π] for phases
        """
        cfg = self.config

        # Ensure correct shape
        embedding = np.asarray(embedding).flatten()
        if len(embedding) != cfg.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {cfg.embedding_dim}, got {len(embedding)}"
            )

        # Normalize input embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Project to oscillator dimensions
        fast_raw = self.proj_fast @ embedding_norm   # [fast_dim]
        slow_raw = self.proj_slow @ embedding_norm   # [slow_dim]

        # Convert to phases
        fast_phases = self._raw_to_phases(fast_raw)
        slow_phases = self._raw_to_phases(slow_raw)

        return PhasePair(
            fast=fast_phases,
            slow=slow_phases,
            source_embedding=embedding,
        )

    def phases_to_embed(
        self,
        fast_phases: np.ndarray,
        slow_phases: np.ndarray,
        fast_weights: Optional[np.ndarray] = None,
        slow_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Project phase patterns back to embedding space.

        Args:
            fast_phases: [fast_dim] phase angles.
            slow_phases: [slow_dim] phase angles.
            fast_weights: Optional weights (e.g., activation potentials).
            slow_weights: Optional weights.

        Returns:
            [embedding_dim] reconstructed embedding.

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

    # ── Text Interface ───────────────────────────────────────────────────────

    def text_to_phases(self, text: str) -> PhasePair:
        """
        Convert text to phase patterns via embedding.

        Requires embedder to be set.
        """
        if self.embedder is None:
            raise RuntimeError(
                "No embedder configured. Pass embedder to __init__."
            )

        embedding = self.embedder(text)
        return self.embed_to_phases(embedding)

    # ── Similarity Operations ────────────────────────────────────────────────

    def phase_similarity(self, phases_a: PhasePair, phases_b: PhasePair) -> float:
        """
        Compute similarity between two phase patterns.

        This should approximate cosine similarity of original embeddings.
        Uses coherence-like measure.
        """
        # Fast scale similarity
        fast_diff = phases_a.fast - phases_b.fast
        fast_sim = float(np.mean(np.cos(fast_diff)))    # [-1, 1]
        fast_sim = (fast_sim + 1) / 2                   # [0, 1]

        # Slow scale similarity
        slow_diff = phases_a.slow - phases_b.slow
        slow_sim = float(np.mean(np.cos(slow_diff)))
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
        return float(min(np.abs(np.mean(np.exp(1j * combined))), 0.999))

    # ── Substrate Integration ────────────────────────────────────────────────

    def stimulate_from_text(
        self,
        substrate: MultiScaleSubstrate,
        text: str,
        strength: float = 0.5,
    ) -> PhasePair:
        """
        Stimulate substrate with text content.

        Args:
            substrate: The MultiScaleSubstrate to stimulate.
            text: Text content to ground.
            strength: Stimulation strength.

        Returns:
            The phase patterns used for stimulation.
        """
        phases = self.text_to_phases(text)
        substrate.stimulate_concept(phases.fast, phases.slow, strength)
        return phases

    def read_substrate_embedding(
        self, substrate: MultiScaleSubstrate
    ) -> np.ndarray:
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

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_similarity_preservation(
        self,
        test_embeddings: List[np.ndarray],
        tolerance: float = 0.1,
    ) -> dict:
        """
        Test how well projection preserves pairwise similarities.

        Args:
            test_embeddings: List of embedding vectors to test.
            tolerance: Acceptable deviation from original similarity.

        Returns:
            Dict with preservation statistics.
        """
        n = len(test_embeddings)
        original_sims = []
        projected_sims = []

        # Compute all pairwise similarities
        for i in range(n):
            for j in range(i + 1, n):
                # Original cosine similarity
                norm_i = np.linalg.norm(test_embeddings[i])
                norm_j = np.linalg.norm(test_embeddings[j])
                orig_sim = float(
                    np.dot(test_embeddings[i], test_embeddings[j])
                    / (norm_i * norm_j + 1e-8)
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
        correlation = float(np.corrcoef(original_sims, projected_sims)[0, 1])
        mean_error = float(np.mean(np.abs(original_sims - projected_sims)))
        max_error = float(np.max(np.abs(original_sims - projected_sims)))

        return {
            "correlation": correlation,
            "mean_error": mean_error,
            "max_error": max_error,
            "within_tolerance": mean_error <= tolerance,
            "n_pairs": len(original_sims),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _init_projections(self) -> None:
        """Initialize projection matrices based on config."""
        cfg = self.config
        rng = np.random.RandomState(cfg.projection_seed)

        if cfg.projection_method == "random_fixed":
            # Random orthogonal projection (Johnson-Lindenstrauss style)
            # This preserves distances/similarities approximately

            # Fast projection: embedding_dim → fast_oscillators
            raw_fast = rng.randn(cfg.fast_oscillators, cfg.embedding_dim)
            # Orthogonalize rows for better preservation
            q_fast, _ = np.linalg.qr(raw_fast.T)
            self.proj_fast = q_fast.T[: cfg.fast_oscillators]

            # Slow projection: embedding_dim → slow_oscillators
            raw_slow = rng.randn(cfg.slow_oscillators, cfg.embedding_dim)
            q_slow, _ = np.linalg.qr(raw_slow.T)
            self.proj_slow = q_slow.T[: cfg.slow_oscillators]

            # Inverse projections (pseudo-inverse for reconstruction)
            self.proj_fast_inv = np.linalg.pinv(self.proj_fast)
            self.proj_slow_inv = np.linalg.pinv(self.proj_slow)

        elif cfg.projection_method == "learned":
            # Placeholder for learned projections
            raise NotImplementedError("Learned projections require training")

        else:
            raise ValueError(f"Unknown projection method: {cfg.projection_method}")

    def _raw_to_phases(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw projection to phase angles."""
        cfg = self.config

        if cfg.phase_encoding == "linear":
            # Direct linear scaling centered at π, clamped to [0, 2π].
            # Orthogonal projections of unit-norm vectors have entries with
            # std ~1/sqrt(embedding_dim). We scale up so that phase differences
            # between similar/dissimilar embeddings are meaningful but small
            # enough that cos(phase_diff) stays in its approximately linear
            # regime. This makes mean(cos(phase_diff)) correlate well with
            # the original cosine similarity (via Johnson-Lindenstrauss).
            #
            # With scale = sqrt(embedding_dim) * 0.5, typical phase deviations
            # from π are ~0.5 rad, and pairwise diffs ~0.3-0.7 rad.
            scale = np.sqrt(cfg.embedding_dim) * 0.5
            phases = np.clip(raw * scale + np.pi, 0, 2 * np.pi)

        elif cfg.phase_encoding == "angular":
            # Use atan2 for angular encoding
            n = len(raw)
            phases = np.zeros(n)
            for i in range(0, n - 1, 2):
                phases[i] = np.arctan2(raw[i], raw[i + 1]) + np.pi      # [0, 2π]
                phases[i + 1] = np.arctan2(raw[i + 1], raw[i]) + np.pi
            if n % 2 == 1:
                phases[-1] = (np.tanh(raw[-1]) + 1) * np.pi

        elif cfg.phase_encoding == "sinusoidal":
            # Sinusoidal encoding (like positional encoding)
            phases = np.mod(raw * np.pi, 2 * np.pi)

        else:
            raise ValueError(f"Unknown phase encoding: {cfg.phase_encoding}")

        return phases

    def _phases_to_raw(self, phases: np.ndarray) -> np.ndarray:
        """Convert phase angles back to raw values."""
        cfg = self.config

        if cfg.phase_encoding == "linear":
            # Inverse of linear scaling
            scale = np.sqrt(cfg.embedding_dim) * 0.5
            raw = (phases - np.pi) / scale

        elif cfg.phase_encoding == "angular":
            # Use sin/cos representation
            raw = np.sin(phases)  # Simplified inverse

        elif cfg.phase_encoding == "sinusoidal":
            raw = phases / np.pi  # Simplified inverse

        else:
            raise ValueError(f"Unknown phase encoding: {cfg.phase_encoding}")

        return raw
