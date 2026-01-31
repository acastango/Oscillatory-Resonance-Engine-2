# ═══════════════════════════════════════════════════════════════════════════════
# PART 10: CLAIMS ENGINE
# Design: N4 (Computational Neuro) + A3 (ML Integration) + P1 (Dynamical Systems)
# Implementation: I3 (State Management)
# ═══════════════════════════════════════════════════════════════════════════════

"""
N4: "Claims are the bridge between declarative and operational knowledge. Saying
'I am cautious' should actually make the system more cautious - not just remember
that it said so."

A3: "This is how we inject roles into ORE. The COGNIZEN 5-role pattern (Analyst,
Creative, Skeptic, Integrator, Meta) becomes claims that shape dynamics."

P1: "Claims modify the energy landscape. A strong claim creates an attractor basin.
The substrate *wants* to be consistent with its claims."

C6: "Claims should be merkle-anchored. Auditable belief history."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ore2.core.memory import CrystallineMerkleMemory, MemoryBranch
from ore2.core.semantic_grounding import PhasePair, SemanticGrounding


class ClaimScope(Enum):
    """What a claim affects."""
    IDENTITY = "identity"       # Core self-conception
    BEHAVIOR = "behavior"       # Action tendencies
    KNOWLEDGE = "knowledge"     # Factual beliefs
    RELATION = "relation"       # Beliefs about others
    GOAL = "goal"               # Objectives/drives
    CONSTRAINT = "constraint"   # Limitations/rules


class ClaimSource(Enum):
    """Where a claim originated."""
    INNATE = "innate"           # Built-in (rare in developmental model)
    LEARNED = "learned"         # From experience
    INSTRUCTED = "instructed"   # Explicitly told
    INFERRED = "inferred"       # Derived from other claims
    SOCIAL = "social"           # From other entities


@dataclass
class Claim:
    """A single claim that can shape dynamics."""
    id: str
    content: str                    # Natural language content
    strength: float                 # 0-1, confidence/importance
    scope: ClaimScope               # What it affects
    source: ClaimSource             # Where it came from

    # Grounding
    embedding: Optional[np.ndarray] = None
    phase_pattern: Optional[PhasePair] = None

    # State
    active: bool = False
    created_at: str = ""
    activated_at: Optional[str] = None

    # Memory anchoring
    memory_node_id: Optional[str] = None
    coherence_at_creation: float = 0.0


@dataclass
class ClaimsEngineConfig:
    """Configuration for the claims engine."""
    # Grounding (can be passed here or directly to __init__)
    grounding: Optional[SemanticGrounding] = None

    # Strength dynamics
    strength_decay: float = 0.001       # Per tick when inactive
    strength_boost: float = 0.01        # Per tick when consistent
    min_strength: float = 0.1           # Below this, claim is removed

    # Coupling modification
    coupling_scale: float = 0.3         # How much claims affect weights

    # Consistency
    consistency_threshold: float = 0.6  # Below = conflict

    # Limits
    max_active_claims: int = 10         # Cognitive load limit
    max_total_claims: int = 100


class ClaimsEngine:
    """
    Manages claims that shape entity dynamics.

    Claims are injectable knowledge - declarative statements that become
    operational in the substrate. A claim like "I am a helpful assistant"
    isn't just stored - it modifies coupling weights, creates attractors,
    and biases dynamics.

    Claims have three aspects:
    1. Content - What the claim asserts (text/embedding)
    2. Strength - How confidently it's held (0 to 1)
    3. Scope - What it affects (substrate regions, behaviors)

    When a claim is activated, it:
    - Generates a phase pattern (via SemanticGrounding)
    - Modifies coupling weights toward that pattern
    - Creates an attractor that pulls dynamics toward consistency
    """

    def __init__(
        self,
        config: Optional[ClaimsEngineConfig] = None,
        grounding: Optional[SemanticGrounding] = None,
    ):
        self.config = config or ClaimsEngineConfig()
        self.grounding = grounding or self.config.grounding

        if self.grounding is None:
            raise ValueError("ClaimsEngine requires SemanticGrounding")

        self.claims: Dict[str, Claim] = {}
        self.active_claims: List[str] = []

        # Role templates (COGNIZEN-style)
        self._role_templates = self._init_role_templates()

    # ── Role Templates ────────────────────────────────────────────────────────

    @staticmethod
    def _init_role_templates() -> Dict[str, List[str]]:
        """Initialize standard role claim sets."""
        return {
            'analyst': [
                "I examine problems systematically and thoroughly",
                "I seek evidence before drawing conclusions",
                "I notice patterns and inconsistencies",
            ],
            'creative': [
                "I generate novel ideas and perspectives",
                "I make unexpected connections between concepts",
                "I value originality and experimentation",
            ],
            'skeptic': [
                "I question assumptions and challenge claims",
                "I look for flaws and edge cases",
                "I resist premature conclusions",
            ],
            'integrator': [
                "I synthesize diverse viewpoints into coherent wholes",
                "I find common ground between conflicting ideas",
                "I build bridges between perspectives",
            ],
            'meta': [
                "I observe and reflect on my own thinking processes",
                "I monitor the quality and coherence of my reasoning",
                "I adjust my approach based on self-observation",
            ],
        }

    # ── Claim Management ──────────────────────────────────────────────────────

    def add_claim(
        self,
        content: str,
        strength: float = 0.5,
        scope: ClaimScope = ClaimScope.KNOWLEDGE,
        source: ClaimSource = ClaimSource.LEARNED,
    ) -> Claim:
        """
        Add a new claim.

        Args:
            content: Natural language claim content.
            strength: Initial strength (0-1).
            scope: What the claim affects.
            source: Where it came from.

        Returns:
            The created Claim.
        """
        # Check limits
        if len(self.claims) >= self.config.max_total_claims:
            self._prune_weakest()

        # Generate ID
        claim_id = (
            f"claim_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            f"_{len(self.claims)}"
        )

        # Ground the claim
        embedding = (
            self.grounding.embedder(content)
            if self.grounding.embedder is not None
            else None
        )
        phase_pattern = (
            self.grounding.embed_to_phases(embedding)
            if embedding is not None
            else None
        )

        claim = Claim(
            id=claim_id,
            content=content,
            strength=float(np.clip(strength, 0, 1)),
            scope=scope,
            source=source,
            embedding=embedding,
            phase_pattern=phase_pattern,
            created_at=datetime.now().isoformat(),
        )

        self.claims[claim_id] = claim
        return claim

    def remove_claim(self, claim_id: str) -> None:
        """Remove a claim entirely."""
        if claim_id not in self.claims:
            raise KeyError(f"Unknown claim: {claim_id}")

        # Deactivate first if active
        if claim_id in self.active_claims:
            self.deactivate_claim(claim_id)

        del self.claims[claim_id]

    def update_strength(self, claim_id: str, new_strength: float) -> None:
        """Update a claim's strength."""
        if claim_id not in self.claims:
            raise KeyError(f"Unknown claim: {claim_id}")

        self.claims[claim_id].strength = float(np.clip(new_strength, 0, 1))

    def _prune_weakest(self) -> None:
        """Remove the weakest inactive claim."""
        inactive = [c for c in self.claims.values() if not c.active]
        if inactive:
            weakest = min(inactive, key=lambda c: c.strength)
            del self.claims[weakest.id]

    # ── Activation ────────────────────────────────────────────────────────────

    def activate_claim(self, claim_id: str) -> None:
        """Activate a claim, making it influence dynamics."""
        if claim_id not in self.claims:
            raise KeyError(f"Unknown claim: {claim_id}")

        claim = self.claims[claim_id]

        if claim.active:
            return  # Already active

        # Check capacity
        if len(self.active_claims) >= self.config.max_active_claims:
            # Deactivate weakest active claim
            weakest_id = min(
                self.active_claims,
                key=lambda cid: self.claims[cid].strength,
            )
            self.deactivate_claim(weakest_id)

        claim.active = True
        claim.activated_at = datetime.now().isoformat()
        self.active_claims.append(claim_id)

    def deactivate_claim(self, claim_id: str) -> None:
        """Deactivate a claim."""
        if claim_id not in self.claims:
            return

        claim = self.claims[claim_id]
        claim.active = False
        claim.activated_at = None

        if claim_id in self.active_claims:
            self.active_claims.remove(claim_id)

    def activate_role(self, role_name: str) -> None:
        """
        Activate a predefined role (set of claims).

        COGNIZEN roles: analyst, creative, skeptic, integrator, meta
        """
        if role_name not in self._role_templates:
            raise ValueError(
                f"Unknown role: {role_name}. "
                f"Available: {list(self._role_templates.keys())}"
            )

        # Add and activate role claims
        for content in self._role_templates[role_name]:
            # Check if similar claim exists
            existing = self._find_similar_claim(content)

            if existing:
                # Boost existing claim
                existing.strength = min(1.0, existing.strength + 0.2)
                self.activate_claim(existing.id)
            else:
                # Create new claim
                claim = self.add_claim(
                    content=content,
                    strength=0.7,
                    scope=ClaimScope.BEHAVIOR,
                    source=ClaimSource.INSTRUCTED,
                )
                self.activate_claim(claim.id)

    def _find_similar_claim(
        self, content: str, threshold: float = 0.85
    ) -> Optional[Claim]:
        """Find existing claim similar to content."""
        if self.grounding.embedder is None:
            return None

        new_embedding = self.grounding.embedder(content)

        for claim in self.claims.values():
            if claim.embedding is not None:
                similarity = np.dot(new_embedding, claim.embedding) / (
                    np.linalg.norm(new_embedding)
                    * np.linalg.norm(claim.embedding)
                    + 1e-8
                )
                if similarity > threshold:
                    return claim

        return None

    # ── Substrate Integration ─────────────────────────────────────────────────

    def apply_to_substrate(self, substrate: Any) -> None:
        """
        Apply active claims to substrate dynamics.

        This modifies coupling weights to create attractors
        toward claim-consistent states.
        """
        if not self.active_claims:
            return

        cfg = self.config

        for claim_id in self.active_claims:
            claim = self.claims[claim_id]

            if claim.phase_pattern is None:
                continue

            # Claim creates an attractor toward its phase pattern.
            # Modify internal weights to favor this pattern.

            # For fast scale
            self._apply_claim_to_layer(
                substrate.fast,
                claim.phase_pattern.fast,
                claim.strength * cfg.coupling_scale,
            )

            # For slow scale (stronger effect - identity level)
            self._apply_claim_to_layer(
                substrate.slow,
                claim.phase_pattern.slow,
                claim.strength * cfg.coupling_scale * 1.5,
            )

    @staticmethod
    def _apply_claim_to_layer(
        layer: Any,
        target_phases: np.ndarray,
        strength: float,
    ) -> None:
        """
        Modify layer coupling to favor target phase pattern.

        The idea: oscillators that should be in-phase get positive coupling,
        oscillators that should be anti-phase get negative coupling.
        """
        n = layer.n

        # Compute pairwise target phase differences
        target_diff = target_phases[:, np.newaxis] - target_phases[np.newaxis, :]

        # Coupling modification: positive for in-phase targets, negative for anti-phase.
        # This creates an attractor toward the target pattern.
        coupling_mod = strength * np.cos(target_diff)

        # Apply modification (additive, preserving existing structure)
        layer.internal_weights += coupling_mod / n

        # Also stimulate oscillators aligned with target.
        # This activates the claim's pattern.
        current_alignment = np.cos(layer.phases - target_phases)
        aligned_mask = current_alignment > 0.5

        if np.any(aligned_mask):
            indices = np.where(aligned_mask)[0]
            strengths = strength * current_alignment[aligned_mask]
            layer.stimulate(indices, strengths)

    def measure_consistency(self, substrate: Any) -> float:
        """
        Measure how consistent substrate state is with active claims.

        Returns 0-1 where 1 = perfectly consistent.
        """
        if not self.active_claims:
            return 1.0  # No claims = vacuously consistent

        consistencies = []

        for claim_id in self.active_claims:
            claim = self.claims[claim_id]

            if claim.phase_pattern is None:
                continue

            # Measure phase alignment with claim pattern
            # Fast scale
            fast_alignment = float(np.mean(np.cos(
                substrate.fast.phases - claim.phase_pattern.fast
            )))
            fast_alignment = (fast_alignment + 1) / 2  # [0, 1]

            # Slow scale (weighted more)
            slow_alignment = float(np.mean(np.cos(
                substrate.slow.phases - claim.phase_pattern.slow
            )))
            slow_alignment = (slow_alignment + 1) / 2

            # Weight by claim strength
            consistency = (
                (0.4 * fast_alignment + 0.6 * slow_alignment)
                * claim.strength
            )
            consistencies.append(consistency)

        return float(np.mean(consistencies)) if consistencies else 1.0

    # ── Memory Integration ────────────────────────────────────────────────────

    def anchor_to_memory(
        self,
        memory: CrystallineMerkleMemory,
        substrate_state: Optional[dict] = None,
    ) -> None:
        """Anchor all unanchored claims to memory."""
        for claim in self.claims.values():
            if claim.memory_node_id is not None:
                continue  # Already anchored

            # Store in SELF branch
            node = memory.add(
                MemoryBranch.SELF,
                {
                    'type': 'claim',
                    'claim_id': claim.id,
                    'content': claim.content,
                    'strength': claim.strength,
                    'scope': claim.scope.value,
                    'source': claim.source.value,
                    'created_at': claim.created_at,
                },
                substrate_state=substrate_state,
                immediate=True,  # Claims are important
            )

            claim.memory_node_id = node.id
            claim.coherence_at_creation = node.coherence_at_creation

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_claims_by_scope(self, scope: ClaimScope) -> List[Claim]:
        """Get all claims with a given scope."""
        return [c for c in self.claims.values() if c.scope == scope]

    def get_conflicting_claims(self) -> List[Tuple[Claim, Claim]]:
        """
        Find pairs of claims that are semantically inconsistent.

        Uses embedding similarity - very similar claims with
        opposite implications are conflicts.
        """
        conflicts = []
        claim_list = list(self.claims.values())

        for i in range(len(claim_list)):
            for j in range(i + 1, len(claim_list)):
                c1, c2 = claim_list[i], claim_list[j]

                if c1.embedding is None or c2.embedding is None:
                    continue

                # Check if same scope (conflicts most relevant within scope)
                if c1.scope != c2.scope:
                    continue

                # Compute phase similarity
                if c1.phase_pattern and c2.phase_pattern:
                    similarity = self.grounding.phase_similarity(
                        c1.phase_pattern, c2.phase_pattern
                    )

                    # Medium similarity = potential conflict
                    # (Very high = same claim, very low = unrelated)
                    if 0.3 < similarity < 0.7:
                        if self._appears_contradictory(c1.content, c2.content):
                            conflicts.append((c1, c2))

        return conflicts

    @staticmethod
    def _appears_contradictory(content1: str, content2: str) -> bool:
        """Simple heuristic for contradiction detection."""
        negations = ['not', "don't", "never", "avoid", "refuse", "cannot"]

        c1_lower = content1.lower()
        c2_lower = content2.lower()

        # One has negation, other doesn't, on similar topic
        c1_negated = any(neg in c1_lower for neg in negations)
        c2_negated = any(neg in c2_lower for neg in negations)

        return c1_negated != c2_negated

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def claim_coherence(self) -> float:
        """
        Internal coherence of active claims.
        High = claims are mutually consistent.
        Low = claims conflict with each other.
        """
        if len(self.active_claims) < 2:
            return 1.0

        # Pairwise phase coherence between active claims
        coherences = []

        for i, cid1 in enumerate(self.active_claims):
            for cid2 in self.active_claims[i + 1:]:
                c1, c2 = self.claims[cid1], self.claims[cid2]

                if c1.phase_pattern and c2.phase_pattern:
                    coh = self.grounding.phase_similarity(
                        c1.phase_pattern, c2.phase_pattern
                    )
                    coherences.append(coh)

        return float(np.mean(coherences)) if coherences else 1.0

    # ── State ─────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Serialize current state for introspection."""
        return {
            'total_claims': len(self.claims),
            'active_claims': len(self.active_claims),
            'active_claim_ids': list(self.active_claims),
            'claim_coherence': self.claim_coherence,
            'claims': {
                cid: {
                    'content': c.content,
                    'strength': c.strength,
                    'scope': c.scope.value,
                    'source': c.source.value,
                    'active': c.active,
                    'anchored': c.memory_node_id is not None,
                }
                for cid, c in self.claims.items()
            },
        }
