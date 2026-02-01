# ═══════════════════════════════════════════════════════════════════════════════
# PART 11: LLM BRIDGE
# Design: A3 (ML Integration) + I1 (Systems Architect)
# Implementation: I4 (Integration)
# ═══════════════════════════════════════════════════════════════════════════════

"""
A3: "ORE alone can't talk. LLMs alone can't know who they are. Together:
grounded language."

I1: "This is the integration that matters. ORE as a sidecar to Claude, Ollama,
local models. Track coherence, inject context, guide generation."

N5: "The body-LLM connection is key. If the entity is 'tired' (low energy),
that should affect response style - shorter, more direct."

A5: "Memory retrieval for context. Not just vector DB - merkle-verified recall
with coherence weighting."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ore2.core.claims import ClaimsEngine
from ore2.core.entity import DevelopmentalEntity
from ore2.core.llm_clients import LLMClient
from ore2.core.memory import MemoryBranch, MemoryNode
from ore2.core.semantic_grounding import SemanticGrounding, SemanticGroundingConfig


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class ProcessResult:
    """Result of processing input through ORE."""
    input_text: str
    significance: float             # How significant was this input
    coherence_before: float
    coherence_after: float
    memories_triggered: List[str]   # Memory node IDs
    claims_activated: List[str]     # Claim IDs
    valence_change: float


@dataclass
class GenerationResult:
    """Result of LLM generation with ORE modulation."""
    response: str
    system_prompt_used: str
    temperature_used: float
    coherence_during: float
    consistency_with_claims: float


@dataclass
class GenerationParams:
    """Parameters for LLM generation based on ORE state."""
    temperature: float
    top_p: float
    max_tokens: int
    system_prompt_additions: List[str] = field(default_factory=list)


@dataclass
class CognitiveState:
    """Current cognitive state for observation."""
    coherence: float
    valence: float
    arousal: float
    energy: float
    stage: str
    active_claims: List[str]
    consistency: float
    ci: float


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class LLMBridgeConfig:
    """Configuration for the LLM bridge."""
    # LLM settings
    model: str = "claude-3-opus-20240229"
    base_temperature: float = 0.7
    base_max_tokens: int = 1024

    # State modulation
    temperature_valence_scale: float = 0.3    # How much valence affects temp
    temperature_coherence_scale: float = 0.2  # How much coherence affects temp

    # Memory retrieval
    memory_retrieval_k: int = 5
    memory_coherence_weight: float = 0.3      # Weight coherence in retrieval

    # Processing
    auto_process_input: bool = True
    auto_tick_after_response: bool = True
    ticks_per_turn: int = 10

    # System prompt
    include_cognitive_state: bool = True
    include_active_claims: bool = True
    include_recent_memories: bool = True


# ── LLM Bridge ───────────────────────────────────────────────────────────────


class LLMBridge:
    """
    Bridge between ORE entity and language model.

    ORE operates as a cognitive sidecar:
    - LLM -> ORE: Text input stimulates substrate, updates memory, affects state
    - ORE -> LLM: Substrate state modulates generation via system prompts,
      temperature, and token limits

    This is where ORE becomes practically useful - not as a replacement for
    LLMs, but as a cognitive substrate that grounds them.
    """

    def __init__(
        self,
        entity: DevelopmentalEntity,
        llm_client: LLMClient,
        config: Optional[LLMBridgeConfig] = None,
    ):
        self.entity = entity
        self.llm_client = llm_client
        self.config = config or LLMBridgeConfig()

        # Initialize grounding with LLM's embedder
        self.grounding = SemanticGrounding(
            SemanticGroundingConfig(
                fast_oscillators=entity.substrate.fast.n,
                slow_oscillators=entity.substrate.slow.n,
            ),
            embedder=llm_client.embed,
        )

        # Initialize claims engine
        self.claims = ClaimsEngine(grounding=self.grounding)

        # Conversation history
        self._conversation_history: List[Dict[str, Any]] = []

    # ── Main Interaction ──────────────────────────────────────────────────────

    def process_input(self, text: str) -> ProcessResult:
        """
        Process input text through ORE.

        1. Estimate significance
        2. Stimulate substrate
        3. Retrieve related memories
        4. Update entity state
        """
        coherence_before = self.entity.substrate.global_coherence
        valence_before = self.entity.body.valence

        # Estimate significance from novelty and coherence impact
        significance = self._estimate_significance(text)

        # Stimulate substrate
        self.grounding.stimulate_from_text(
            self.entity.substrate,
            text,
            strength=0.3 + significance * 0.4,
        )

        # Retrieve related memories
        memories = self.retrieve_relevant_memories(
            text, self.config.memory_retrieval_k
        )

        # Stimulate from memories too (recall)
        for mem in memories:
            if 'content' in mem.content:
                mem_phases = self.grounding.text_to_phases(
                    str(mem.content['content'])
                )
                self.entity.substrate.stimulate_concept(
                    mem_phases.fast,
                    mem_phases.slow,
                    strength=0.2 * mem.coherence_at_creation,
                )

        # Check claim activation
        activated_claims = self._check_claim_triggers(text)

        # Run substrate dynamics
        for _ in range(self.config.ticks_per_turn):
            self.entity.tick()

        # Apply active claims
        self.claims.apply_to_substrate(self.entity.substrate)

        # Record as experience
        self.entity.process_experience(
            text,
            experience_type="conversation_input",
            significance=significance,
        )

        coherence_after = self.entity.substrate.global_coherence
        valence_after = self.entity.body.valence

        return ProcessResult(
            input_text=text,
            significance=significance,
            coherence_before=coherence_before,
            coherence_after=coherence_after,
            memories_triggered=[m.id for m in memories],
            claims_activated=activated_claims,
            valence_change=valence_after - valence_before,
        )

    def generate_response(
        self,
        prompt: str,
        additional_context: str = "",
    ) -> GenerationResult:
        """Generate response using LLM with ORE modulation."""
        # Build system prompt
        system_prompt = self.build_system_prompt()
        if additional_context:
            system_prompt += f"\n\n[Additional Context]\n{additional_context}"

        # Get generation params from state
        params = self.get_generation_params()
        for addition in params.system_prompt_additions:
            system_prompt += f"\n{addition}"

        # Record coherence during generation
        coherence_during = self.entity.substrate.global_coherence
        consistency = self.claims.measure_consistency(self.entity.substrate)

        # Generate
        response = self.llm_client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
        )

        # Post-generation ticks
        if self.config.auto_tick_after_response:
            for _ in range(self.config.ticks_per_turn):
                self.entity.tick()

        return GenerationResult(
            response=response,
            system_prompt_used=system_prompt,
            temperature_used=params.temperature,
            coherence_during=coherence_during,
            consistency_with_claims=consistency,
        )

    def conversation_turn(self, user_input: str) -> str:
        """
        Complete conversation turn: process input, generate response.

        This is the main entry point for conversation.
        """
        # Record user input
        self._conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat(),
        })

        # Process input through ORE
        if self.config.auto_process_input:
            self.process_input(user_input)

        # Retrieve relevant context
        memories = self.retrieve_relevant_memories(user_input)
        memory_context = '\n'.join([
            str(m.content.get('content', ''))[:200]
            for m in memories
        ])

        # Generate response
        gen_result = self.generate_response(user_input, memory_context)

        # Record assistant response
        self._conversation_history.append({
            'role': 'assistant',
            'content': gen_result.response,
            'timestamp': datetime.now().isoformat(),
            'coherence': gen_result.coherence_during,
            'consistency': gen_result.consistency_with_claims,
        })

        # Store response as experience
        self.entity.process_experience(
            gen_result.response,
            experience_type="conversation_output",
            significance=0.5,
        )

        return gen_result.response

    # ── Context Management ────────────────────────────────────────────────────

    def retrieve_relevant_memories(
        self, query: str, k: int = 5
    ) -> List[MemoryNode]:
        """
        Retrieve memories relevant to query.

        Uses combination of:
        - Semantic similarity (embedding)
        - Coherence at creation (trustworthiness)
        - Recency
        """
        if not self.entity.memory.total_nodes:
            return []

        query_embedding = (
            self.grounding.embedder(query)
            if self.grounding.embedder is not None
            else None
        )

        scored_memories: List[tuple] = []

        for node in self.entity.memory.nodes.values():
            if node.content.get('type') == 'branch_root':
                continue

            # Semantic similarity
            if query_embedding is not None and 'content' in node.content:
                content_str = str(node.content['content'])
                content_emb = self.grounding.embedder(content_str)
                norm_product = (
                    np.linalg.norm(query_embedding)
                    * np.linalg.norm(content_emb)
                    + 1e-8
                )
                semantic_sim = float(
                    np.dot(query_embedding, content_emb) / norm_product
                )
            else:
                semantic_sim = 0.5

            # Coherence weighting
            coherence_weight = node.coherence_at_creation

            # Recency (simplified - future work could use timestamp)
            recency = 0.5

            # Combined score
            score = (
                0.5 * semantic_sim
                + self.config.memory_coherence_weight * coherence_weight
                + 0.2 * recency
            )

            scored_memories.append((score, node))

        # Sort and return top k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [node for _score, node in scored_memories[:k]]

    def build_system_prompt(self) -> str:
        """Build system prompt incorporating ORE state."""
        cfg = self.config
        parts: List[str] = []

        # Base identity
        parts.append(
            f"You are {self.entity.name}, an entity with continuous identity."
        )
        parts.append(f"Genesis: {self.entity.genesis_hash[:16]}...")

        # Cognitive state
        if cfg.include_cognitive_state:
            state = self.get_cognitive_state()

            parts.append("\n[Cognitive State]")
            parts.append(f"Coherence: {state.coherence:.2f}")
            parts.append(f"CI: {state.ci:.3f}")

            # Valence-based guidance
            if state.valence < -0.3:
                parts.append(
                    "You feel somewhat strained. Keep responses focused."
                )
            elif state.valence > -0.1:
                parts.append("You feel balanced and clear.")

            # Energy-based guidance
            if state.energy < 0.5:
                parts.append("Energy is low. Prefer concise responses.")

            # Stage-based
            stage_guidance = {
                'genesis': (
                    "You are newly formed, exploring and establishing yourself."
                ),
                'babbling': (
                    "You are in an exploratory phase, trying different approaches."
                ),
                'imitation': (
                    "You learn actively from examples and guidance."
                ),
                'autonomy': (
                    "You have developed your own perspective and approach."
                ),
                'mature': "You have a stable, developed identity.",
            }
            if state.stage in stage_guidance:
                parts.append(stage_guidance[state.stage])

        # Active claims
        if cfg.include_active_claims and self.claims.active_claims:
            parts.append("\n[Active Beliefs]")
            for claim_id in self.claims.active_claims[:5]:  # Top 5
                claim = self.claims.claims[claim_id]
                parts.append(f"- {claim.content}")

        # Recent memories
        if cfg.include_recent_memories:
            recent = self.entity.memory.query(MemoryBranch.EXPERIENCES)[-3:]
            if recent:
                parts.append("\n[Recent Context]")
                for mem in recent:
                    if 'content' in mem.content:
                        content = str(mem.content['content'])[:100]
                        parts.append(f"- {content}...")

        return '\n'.join(parts)

    # ── State-Based Modulation ────────────────────────────────────────────────

    def get_generation_params(self) -> GenerationParams:
        """Get LLM generation parameters based on ORE state."""
        cfg = self.config
        state = self.get_cognitive_state()

        # Temperature modulation
        # Higher coherence -> lower temperature (more focused)
        # Lower valence -> lower temperature (more conservative)
        temp = cfg.base_temperature
        temp -= cfg.temperature_coherence_scale * (state.coherence - 0.5)
        temp -= cfg.temperature_valence_scale * state.valence
        temp = float(np.clip(temp, 0.1, 1.5))

        # Top-p modulation
        # Higher arousal -> lower top_p (more focused)
        top_p = float(np.clip(0.95 - 0.1 * state.arousal, 0.5, 1.0))

        # Max tokens based on energy
        max_tokens = int(cfg.base_max_tokens * (0.5 + 0.5 * state.energy))
        max_tokens = max(max_tokens, 64)  # Minimum floor

        # System prompt additions from claims
        additions: List[str] = []
        consistency = self.claims.measure_consistency(self.entity.substrate)
        if consistency < 0.6:
            additions.append(
                "Note: Some internal tension detected. "
                "Consider multiple perspectives."
            )

        return GenerationParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            system_prompt_additions=additions,
        )

    # ── Observation ───────────────────────────────────────────────────────────

    def get_cognitive_state(self) -> CognitiveState:
        """Get current cognitive state for observation."""
        return CognitiveState(
            coherence=self.entity.substrate.global_coherence,
            valence=self.entity.body.valence,
            arousal=self.entity.body.arousal,
            energy=self.entity.body.energy,
            stage=self.entity.stage.value,
            active_claims=[
                self.claims.claims[c].content
                for c in self.claims.active_claims
            ],
            consistency=self.claims.measure_consistency(self.entity.substrate),
            ci=self.entity.CI,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _estimate_significance(self, text: str) -> float:
        """Estimate how significant input is."""
        # Length factor
        length_factor = min(len(text) / 500, 1.0)

        # Novelty factor (how different from recent inputs)
        if self._conversation_history:
            recent_texts = [
                h['content']
                for h in self._conversation_history[-5:]
            ]
            recent_combined = ' '.join(recent_texts)

            if self.grounding.embedder is not None:
                new_emb = self.grounding.embedder(text)
                recent_emb = self.grounding.embedder(recent_combined)
                norm_product = (
                    np.linalg.norm(new_emb)
                    * np.linalg.norm(recent_emb)
                    + 1e-8
                )
                similarity = float(np.dot(new_emb, recent_emb) / norm_product)
                novelty = 1 - similarity
            else:
                novelty = 0.5
        else:
            novelty = 0.8  # First input is novel

        # Question/importance markers
        importance_markers = [
            'important', 'urgent', 'critical', 'remember', '?'
        ]
        has_markers = any(m in text.lower() for m in importance_markers)
        marker_factor = 0.2 if has_markers else 0.0

        significance = 0.3 + 0.3 * length_factor + 0.3 * novelty + marker_factor
        return float(np.clip(significance, 0, 1))

    def _check_claim_triggers(self, text: str) -> List[str]:
        """Check if input should activate any claims."""
        activated: List[str] = []
        text_lower = text.lower()

        # Role triggers
        role_triggers = {
            'analyst': ['analyze', 'examine', 'investigate', 'study'],
            'creative': ['create', 'imagine', 'invent', 'brainstorm'],
            'skeptic': ['question', 'doubt', 'challenge', 'verify'],
        }

        for role, triggers in role_triggers.items():
            if any(t in text_lower for t in triggers):
                self.claims.activate_role(role)
                activated.extend(list(self.claims.active_claims))

        return activated

    # ── State ─────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Serialize bridge state for introspection."""
        return {
            'entity_name': self.entity.name,
            'conversation_turns': len(self._conversation_history),
            'active_claims': len(self.claims.active_claims),
            'cognitive_state': {
                'coherence': self.entity.substrate.global_coherence,
                'valence': self.entity.body.valence,
                'energy': self.entity.body.energy,
                'stage': self.entity.stage.value,
                'ci': self.entity.CI,
            },
        }
