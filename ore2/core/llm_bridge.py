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

from ore2.core.claims import Claim, ClaimScope, ClaimSource, ClaimsEngine
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
    model: str = "claude-sonnet-4-20250514"
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
        embedder: Optional[Any] = None,
    ):
        self.entity = entity
        self.llm_client = llm_client
        self.config = config or LLMBridgeConfig()

        # Resolve embedder: explicit param > llm_client.embed > None
        resolved_embedder = embedder
        if resolved_embedder is None:
            resolved_embedder = self._probe_embedder(llm_client)

        # Initialize grounding with embedder (may be None)
        self.grounding = SemanticGrounding(
            SemanticGroundingConfig(
                fast_oscillators=entity.substrate.fast.n,
                slow_oscillators=entity.substrate.slow.n,
            ),
            embedder=resolved_embedder,
        )

        # Initialize claims engine
        self.claims = ClaimsEngine(grounding=self.grounding)

        # Conversation history
        self._conversation_history: List[Dict[str, Any]] = []

    @staticmethod
    def _probe_embedder(llm_client: LLMClient) -> Optional[Any]:
        """
        Test if llm_client.embed works. If it raises NotImplementedError,
        return None so the bridge operates in completion-only mode.
        """
        try:
            llm_client.embed("test")
            return llm_client.embed
        except NotImplementedError:
            return None

    # ── Main Interaction ──────────────────────────────────────────────────────

    def process_input(self, text: str) -> ProcessResult:
        """
        Process input text through ORE.

        1. Estimate significance
        2. Stimulate substrate from text and memories
        3. Apply claims to shape dynamics
        4. Record as experience (which runs substrate ticks internally)
        """
        coherence_before = self.entity.substrate.global_coherence
        valence_before = self.entity.body.valence

        # Estimate significance from novelty and coherence impact
        significance = self._estimate_significance(text)

        # Stimulate substrate (requires embedder for text -> phases)
        if self.grounding.embedder is not None:
            self.grounding.stimulate_from_text(
                self.entity.substrate,
                text,
                strength=0.6 + significance * 0.4,
            )

        # Retrieve related memories
        memories = self.retrieve_relevant_memories(
            text, self.config.memory_retrieval_k
        )

        # Stimulate from memories too (recall, requires embedder)
        if self.grounding.embedder is not None:
            for mem in memories:
                if 'content' in mem.content:
                    mem_phases = self.grounding.text_to_phases(
                        str(mem.content['content'])
                    )
                    self.entity.substrate.stimulate_concept(
                        mem_phases.fast,
                        mem_phases.slow,
                        strength=0.3 * mem.coherence_at_creation,
                    )

        # Check claim activation
        activated_claims = self._check_claim_triggers(text)

        # Apply active claims BEFORE dynamics so they shape the trajectory
        self.claims.apply_to_substrate(self.entity.substrate)

        # Record as experience — skip_stimulation and run_ticks=False when
        # we have grounding, because:
        # 1. Hash-based patterns would fight the grounded ones
        # 2. The internal tick loop would decay activation without re-stim
        # The bridge handles its own tick loop with sustained stimulation.
        has_grounding = self.grounding.embedder is not None
        self.entity.process_experience(
            text,
            experience_type="conversation_input",
            significance=significance,
            skip_stimulation=has_grounding,
            run_ticks=not has_grounding,
        )

        # Bridge-controlled tick loop: sustain activation every tick to
        # counteract the 5x fast decay from nesting ratio, plus periodic
        # content-based stimulation to reinforce phase coherence.
        if has_grounding:
            for i in range(self.config.ticks_per_turn):
                self.entity.substrate.sustain_activation()
                if i % 3 == 0:
                    self.grounding.stimulate_from_text(
                        self.entity.substrate, text, strength=0.2,
                    )
                self.entity.tick()

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

        # Build conversation history for multi-turn
        messages = self._build_messages_for_llm()

        # Generate
        response = self.llm_client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            messages=messages if messages else None,
        )

        # Post-generation ticks with sustained activation
        if self.config.auto_tick_after_response:
            for i in range(self.config.ticks_per_turn):
                if self.grounding.embedder is not None:
                    self.entity.substrate.sustain_activation()
                    if i % 3 == 0:
                        self.grounding.stimulate_from_text(
                            self.entity.substrate, prompt[:200],
                            strength=0.15,
                        )
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

        Full feedback loop:
        1. Record user input in history
        2. Process input through ORE (stimulate substrate, trigger claims)
        3. Retrieve relevant memories for context
        4. Generate response with ORE-modulated params
        5. Feed response back through substrate (bidirectional grounding)
        6. Store in appropriate memory branches
        7. Anchor unanchored claims to memory
        8. Update arousal based on conversation intensity
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

        # ── Response feedback loop ───────────────────────────────────────
        # Feed response back through substrate (bidirectional grounding)
        if self.grounding.embedder is not None:
            self.grounding.stimulate_from_text(
                self.entity.substrate,
                gen_result.response[:500],  # Truncate for efficiency
                strength=0.5,
            )

        # Estimate response significance
        resp_significance = self._estimate_significance(gen_result.response)

        # Store response as experience — skip hash-based stimulation and
        # internal ticks when we have grounding, for the same reasons as
        # in process_input: hash patterns fight grounded ones and the
        # internal tick loop decays activation without re-stimulation.
        has_grounding = self.grounding.embedder is not None
        self.entity.process_experience(
            gen_result.response,
            experience_type="conversation_output",
            significance=resp_significance,
            skip_stimulation=has_grounding,
            run_ticks=not has_grounding,
        )

        # Sustain activation after response processing
        if has_grounding:
            for i in range(self.config.ticks_per_turn):
                self.entity.substrate.sustain_activation()
                if i % 3 == 0:
                    self.grounding.stimulate_from_text(
                        self.entity.substrate,
                        gen_result.response[:200],
                        strength=0.15,
                    )
                self.entity.tick()

        # Store high-significance responses as insights
        if resp_significance > 0.7:
            self.entity.memory.add(
                MemoryBranch.INSIGHTS,
                {
                    'type': 'conversation_insight',
                    'content': gen_result.response[:500],
                    'coherence': gen_result.coherence_during,
                    'prompt': user_input[:200],
                },
                substrate_state=self.entity.substrate.get_state(),
                immediate=True,
            )

        # Anchor unanchored claims to memory
        if self.claims.active_claims:
            self.claims.anchor_to_memory(
                self.entity.memory,
                substrate_state=self.entity.substrate.get_state(),
            )

        # Update arousal based on conversation pace
        turn_count = len(self._conversation_history) // 2
        if turn_count > 3:
            # Conversation is getting longer — arousal rises
            self.entity.body.arousal = min(
                self.entity.body.arousal + 0.02, 1.0
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
        """Build system prompt incorporating full ORE state."""
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

            # CI trajectory
            ci_history = self.entity.ci_monitor.history
            if len(ci_history) >= 2:
                recent_ci = [s.CI_integrated for s in ci_history[-10:]]
                trend = recent_ci[-1] - recent_ci[0]
                if trend > 0.001:
                    parts.append("Consciousness is rising.")
                elif trend < -0.001:
                    parts.append("Consciousness is fading.")

            # Valence-based guidance
            if state.valence < -0.3:
                parts.append(
                    "You feel somewhat strained. Keep responses focused."
                )
            elif state.valence > -0.1:
                parts.append("You feel balanced and clear.")

            # Body state constraints
            body = self.entity.body
            if body.is_depleted:
                parts.append(
                    "You are depleted. Keep responses brief and essential."
                )
            elif state.energy < 0.5:
                parts.append("Energy is low. Prefer concise responses.")

            if body.is_overaroused:
                parts.append(
                    "You feel overstimulated. Ground yourself, be deliberate."
                )

            # Stage-based with progress
            progress = self.entity.development.stage_progress
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
                if progress > 0.8:
                    parts.append("You are nearing a developmental transition.")

        # Active claims with coherence
        if cfg.include_active_claims and self.claims.active_claims:
            coherence = self.claims.claim_coherence
            parts.append(f"\n[Active Beliefs] (coherence: {coherence:.2f})")
            for claim_id in self.claims.active_claims[:5]:  # Top 5
                claim = self.claims.claims[claim_id]
                parts.append(f"- {claim.content}")

            # Conflict detection
            conflicts = self.claims.get_conflicting_claims()
            if conflicts:
                parts.append(
                    f"Note: {len(conflicts)} belief conflict(s) detected. "
                    "Acknowledge tensions honestly."
                )

        # Recent memories across branches
        if cfg.include_recent_memories:
            recent_exp = self.entity.memory.query(
                MemoryBranch.EXPERIENCES
            )[-3:]
            recent_insights = self.entity.memory.query(
                MemoryBranch.INSIGHTS
            )[-2:]

            if recent_exp or recent_insights:
                parts.append("\n[Recent Context]")
                for mem in recent_exp:
                    if 'content' in mem.content:
                        content = str(mem.content['content'])[:100]
                        parts.append(f"- {content}...")
                for mem in recent_insights:
                    if 'content' in mem.content:
                        content = str(mem.content['content'])[:100]
                        parts.append(f"- (insight) {content}...")

        return '\n'.join(parts)

    # ── Claims & Beliefs ──────────────────────────────────────────────────────

    def add_belief(
        self,
        content: str,
        scope: str = "knowledge",
        source: str = "instructed",
        strength: float = 0.8,
        activate: bool = True,
    ) -> Claim:
        """
        Add a new belief/claim that shapes entity dynamics.

        Args:
            content: Natural language belief content.
            scope: One of identity, behavior, knowledge, relation, goal, constraint.
            source: One of innate, learned, instructed, inferred, social.
            strength: 0-1 confidence level.
            activate: Whether to immediately activate this claim.

        Returns:
            The created Claim.
        """
        scope_enum = ClaimScope(scope)
        source_enum = ClaimSource(source)

        claim = self.claims.add_claim(
            content,
            strength=strength,
            scope=scope_enum,
            source=source_enum,
        )

        if activate:
            self.claims.activate_claim(claim.id)

        # Apply claim to substrate immediately
        self.claims.apply_to_substrate(self.entity.substrate)

        return claim

    def remove_belief(self, claim_id: str) -> None:
        """Remove a belief/claim by ID."""
        self.claims.remove_claim(claim_id)

    def list_beliefs(self) -> List[Dict[str, Any]]:
        """List all beliefs with their status."""
        beliefs = []
        for claim in self.claims.claims.values():
            beliefs.append({
                'id': claim.id,
                'content': claim.content,
                'strength': claim.strength,
                'scope': claim.scope.value,
                'source': claim.source.value,
                'active': claim.id in self.claims.active_claims,
                'anchored': claim.memory_node_id is not None,
            })
        return beliefs

    def activate_role(self, role_name: str) -> List[str]:
        """
        Activate a COGNIZEN role (analyst, creative, skeptic, integrator, meta).

        Returns list of activated claim IDs.
        """
        self.claims.activate_role(role_name)
        self.claims.apply_to_substrate(self.entity.substrate)
        return list(self.claims.active_claims)

    def deactivate_all_roles(self) -> None:
        """Deactivate all currently active claims."""
        for claim_id in list(self.claims.active_claims):
            self.claims.deactivate_claim(claim_id)

    def list_roles(self) -> List[str]:
        """List available COGNIZEN roles."""
        return list(self.claims._role_templates.keys())

    def get_conflicts(self) -> List[Dict[str, str]]:
        """Find conflicting beliefs."""
        conflicts = self.claims.get_conflicting_claims()
        return [
            {
                'claim_a': f"[{c1.id[:8]}] {c1.content}",
                'claim_b': f"[{c2.id[:8]}] {c2.content}",
            }
            for c1, c2 in conflicts
        ]

    # ── Memory Access ────────────────────────────────────────────────────────

    def store_insight(self, content: str, significance: float = 0.7) -> None:
        """Store an insight in the INSIGHTS memory branch."""
        self.entity.memory.add(
            MemoryBranch.INSIGHTS,
            {
                'type': 'insight',
                'content': content,
                'significance': significance,
            },
            substrate_state=self.entity.substrate.get_state(),
            immediate=significance > 0.5,
        )

    def store_relationship(self, content: str, significance: float = 0.6) -> None:
        """Store a relationship observation in the RELATIONS branch."""
        self.entity.memory.add(
            MemoryBranch.RELATIONS,
            {
                'type': 'relationship',
                'content': content,
                'significance': significance,
            },
            substrate_state=self.entity.substrate.get_state(),
            immediate=significance > 0.5,
        )

    def get_memories(
        self, branch: Optional[str] = None, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories, optionally from a specific branch.

        Args:
            branch: One of 'self', 'relations', 'insights', 'experiences', or None for all.
            k: Max memories to return.
        """
        branch_map = {
            'self': MemoryBranch.SELF,
            'relations': MemoryBranch.RELATIONS,
            'insights': MemoryBranch.INSIGHTS,
            'experiences': MemoryBranch.EXPERIENCES,
        }

        if branch:
            mem_branch = branch_map.get(branch.lower())
            if mem_branch is None:
                return []
            nodes = self.entity.memory.query(mem_branch)
        else:
            nodes = [
                n for n in self.entity.memory.nodes.values()
                if n.content.get('type') != 'branch_root'
            ]

        # Sort by creation order (newest first) and limit
        results = []
        for node in nodes[-k:]:
            entry = {
                'id': node.id[:12],
                'branch': node.branch.value if hasattr(node, 'branch') else 'unknown',
                'content': str(node.content.get('content', node.content.get('type', '?')))[:200],
                'coherence': node.coherence_at_creation,
            }
            results.append(entry)
        return results

    def verify_memory(self) -> bool:
        """Cryptographically verify memory integrity."""
        result = self.entity.memory.verify()
        # verify() returns (bool, str) tuple
        if isinstance(result, tuple):
            return result[0]
        return bool(result)

    # ── Introspection ────────────────────────────────────────────────────────

    def reflect(self) -> Dict[str, Any]:
        """
        Deep introspection: CI breakdown, claim coherence, memory stats,
        body state, development progress.
        """
        ci_status = self.entity.ci_monitor.get_current_status()
        ci_history = self.entity.ci_monitor.history

        # CI trajectory
        if len(ci_history) >= 2:
            recent_ci = [s.CI_integrated for s in ci_history[-10:]]
            ci_trend = recent_ci[-1] - recent_ci[0]
        else:
            ci_trend = 0.0

        # Per-scale CI
        if ci_history:
            latest = ci_history[-1]
            ci_fast = latest.CI_fast
            ci_slow = latest.CI_slow
            cross_scale = latest.C_cross
        else:
            ci_fast = ci_slow = cross_scale = 0.0

        # Claim analysis
        conflicts = self.get_conflicts()

        # Memory stats
        mem_state = self.entity.memory.get_state()

        # Body state
        body = self.entity.body

        return {
            'consciousness': {
                'ci': self.entity.CI,
                'ci_fast': ci_fast,
                'ci_slow': ci_slow,
                'cross_scale_coherence': cross_scale,
                'ci_trend': ci_trend,
                'status': ci_status,
            },
            'body': {
                'valence': body.valence,
                'arousal': body.arousal,
                'energy': body.energy,
                'depleted': body.is_depleted,
                'overaroused': body.is_overaroused,
            },
            'claims': {
                'total': len(self.claims.claims),
                'active': len(self.claims.active_claims),
                'coherence': self.claims.claim_coherence,
                'conflicts': conflicts,
            },
            'memory': {
                'total_nodes': mem_state['total_nodes'],
                'depth': mem_state['depth'],
                'fractal_dimension': mem_state['fractal_dimension'],
                'grain_boundaries': mem_state['grain_boundaries'],
                'verified': self.verify_memory(),
                'pending_consolidation': len(
                    self.entity.memory.consolidation_queue.pending_nodes
                ),
            },
            'development': {
                'stage': self.entity.stage.value,
                'progress': self.entity.development.stage_progress,
                'age': self.entity.age,
                'oscillators': self.entity.development.current_oscillators,
            },
            'substrate': {
                'coherence': self.entity.substrate.global_coherence,
                'fast_active': self.entity.substrate.fast.n_active,
                'slow_active': self.entity.substrate.slow.n_active,
                'loop_coherence': self.entity.substrate.loop_coherence,
            },
        }

    def read_mind(self) -> Optional[np.ndarray]:
        """
        Read current substrate state as an embedding vector.

        Returns the semantic embedding of what the substrate is
        currently "thinking about", or None if no embedder.
        """
        if self.grounding.embedder is None:
            return None
        return self.grounding.read_substrate_embedding(self.entity.substrate)

    def rest(self, duration: float = 10.0) -> Dict[str, Any]:
        """
        Trigger rest/consolidation cycle.

        Anchors claims, consolidates memory, returns results.
        """
        # Anchor claims before rest
        if self.claims.claims:
            self.claims.anchor_to_memory(
                self.entity.memory,
                substrate_state=self.entity.substrate.get_state(),
            )

        result = self.entity.rest(duration)

        return {
            'duration': duration,
            'consolidated': result.get('consolidation', {}).get('consolidated', 0),
            'tensions_resolved': result.get('consolidation', {}).get(
                'tensions_resolved', 0
            ),
            'ci_after': result.get('CI_after', self.entity.CI),
            'memory_verified': self.verify_memory(),
        }

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

    def _build_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Build conversation messages for the LLM from history.

        Returns a list of {"role": ..., "content": ...} dicts representing
        prior completed turns. The current (trailing) user message is excluded
        because it's passed separately as the `prompt` argument to complete().
        """
        history = self._conversation_history
        # Exclude trailing user message — it will be sent as `prompt`
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        return [
            {"role": entry["role"], "content": entry["content"]}
            for entry in history
        ]

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
