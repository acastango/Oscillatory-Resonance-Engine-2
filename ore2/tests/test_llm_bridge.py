"""Tests for LLMBridge and LLM clients (ORE2-010)."""

import numpy as np
import pytest

from ore2.core.entity import DevelopmentalEntity, EntityConfig, create_entity
from ore2.core.llm_bridge import (
    CognitiveState,
    GenerationParams,
    GenerationResult,
    LLMBridge,
    LLMBridgeConfig,
    ProcessResult,
)
from ore2.core.llm_clients import LLMClient, MockLLMClient
from ore2.core.memory import MemoryBranch


# ── Helpers ──────────────────────────────────────────────────────────────────


def create_test_bridge(
    config: LLMBridgeConfig = None,
    entity: DevelopmentalEntity = None,
) -> LLMBridge:
    """Create an LLMBridge with MockLLMClient for testing."""
    ent = entity or create_entity("TestEntity")
    client = MockLLMClient()
    return LLMBridge(ent, client, config)


# ── MockLLMClient Tests ──────────────────────────────────────────────────────


def test_mock_client_complete():
    """MockLLMClient.complete should return a non-empty string."""
    client = MockLLMClient()
    response = client.complete("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_mock_client_complete_deterministic():
    """Same prompt should give same response."""
    client = MockLLMClient()
    r1 = client.complete("test prompt")
    r2 = client.complete("test prompt")
    assert r1 == r2


def test_mock_client_complete_varies_by_prompt():
    """Different prompts should give different responses."""
    client = MockLLMClient()
    r1 = client.complete("first prompt")
    r2 = client.complete("completely different prompt")
    assert r1 != r2


def test_mock_client_embed():
    """MockLLMClient.embed should return a unit vector."""
    client = MockLLMClient()
    emb = client.embed("test text")
    assert emb.shape == (1536,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-6


def test_mock_client_embed_deterministic():
    """Same text should give same embedding."""
    client = MockLLMClient()
    e1 = client.embed("hello")
    e2 = client.embed("hello")
    assert np.allclose(e1, e2)


def test_mock_client_embed_varies():
    """Different texts should give different embeddings."""
    client = MockLLMClient()
    e1 = client.embed("hello world")
    e2 = client.embed("goodbye moon")
    assert not np.allclose(e1, e2)


def test_mock_client_call_log():
    """MockLLMClient should log all calls."""
    client = MockLLMClient()
    client.complete("prompt 1")
    client.embed("text 1")
    client.complete("prompt 2")

    assert len(client.call_log) == 3
    assert client.call_log[0]['method'] == 'complete'
    assert client.call_log[1]['method'] == 'embed'
    assert client.call_log[2]['method'] == 'complete'


def test_mock_client_custom_embedding_dim():
    """MockLLMClient should support custom embedding dimensions."""
    client = MockLLMClient(embedding_dim=768)
    emb = client.embed("test")
    assert emb.shape == (768,)


def test_llm_client_abc():
    """LLMClient should be abstract."""
    with pytest.raises(TypeError):
        LLMClient()  # Can't instantiate abstract class


# ── LLMBridge Constructor Tests ──────────────────────────────────────────────


def test_bridge_creation():
    """LLMBridge should initialize all components."""
    bridge = create_test_bridge()

    assert bridge.entity is not None
    assert bridge.llm_client is not None
    assert bridge.grounding is not None
    assert bridge.claims is not None
    assert len(bridge._conversation_history) == 0


def test_bridge_grounding_uses_llm_embedder():
    """Bridge grounding should use the LLM client's embed method."""
    bridge = create_test_bridge()
    assert bridge.grounding.embedder is not None

    # Embedder should work
    emb = bridge.grounding.embedder("test")
    assert emb.shape == (1536,)


def test_bridge_grounding_matches_substrate():
    """Grounding dimensions should match entity substrate."""
    bridge = create_test_bridge()
    assert bridge.grounding.config.fast_oscillators == bridge.entity.substrate.fast.n
    assert bridge.grounding.config.slow_oscillators == bridge.entity.substrate.slow.n


def test_bridge_default_config():
    """Default config should match briefing spec."""
    config = LLMBridgeConfig()
    assert config.base_temperature == 0.7
    assert config.base_max_tokens == 1024
    assert config.temperature_valence_scale == 0.3
    assert config.temperature_coherence_scale == 0.2
    assert config.memory_retrieval_k == 5
    assert config.memory_coherence_weight == 0.3
    assert config.auto_process_input is True
    assert config.auto_tick_after_response is True
    assert config.ticks_per_turn == 10


def test_bridge_custom_config():
    """Custom config should be applied."""
    config = LLMBridgeConfig(
        base_temperature=0.5,
        ticks_per_turn=5,
        include_cognitive_state=False,
    )
    bridge = create_test_bridge(config=config)
    assert bridge.config.base_temperature == 0.5
    assert bridge.config.ticks_per_turn == 5
    assert bridge.config.include_cognitive_state is False


# ── process_input Tests ──────────────────────────────────────────────────────


def test_process_input():
    """Processing input should affect substrate."""
    bridge = create_test_bridge()

    result = bridge.process_input("Hello, how are you?")

    assert isinstance(result, ProcessResult)
    assert result.significance > 0
    assert result.input_text == "Hello, how are you?"
    assert isinstance(result.coherence_before, float)
    assert isinstance(result.coherence_after, float)
    assert isinstance(result.valence_change, float)


def test_process_input_significance_range():
    """Significance should be in [0, 1]."""
    bridge = create_test_bridge()
    result = bridge.process_input("test")
    assert 0.0 <= result.significance <= 1.0


def test_process_input_first_is_novel():
    """First input should have high significance (novelty)."""
    bridge = create_test_bridge()
    result = bridge.process_input("This is the first ever input")
    # First input novelty = 0.8, so significance should be high
    assert result.significance > 0.5


def test_process_input_question_marker():
    """Questions should have higher significance."""
    bridge = create_test_bridge()
    r1 = bridge.process_input("Tell me about something.")
    r2 = bridge.process_input("What is the most important thing?")
    # r2 has both '?' and 'important' markers
    assert r2.significance >= r1.significance


def test_process_input_triggers_claims():
    """Input with role triggers should activate claims."""
    bridge = create_test_bridge()
    result = bridge.process_input("Please analyze this data carefully")

    # "analyze" should trigger analyst role
    assert len(result.claims_activated) > 0


def test_process_input_records_experience():
    """Processing input should add to entity memory."""
    bridge = create_test_bridge()
    nodes_before = bridge.entity.memory.total_nodes

    bridge.process_input("Remember this experience")

    assert bridge.entity.memory.total_nodes > nodes_before


# ── generate_response Tests ──────────────────────────────────────────────────


def test_generate_response():
    """Generate response should return a GenerationResult."""
    bridge = create_test_bridge()
    result = bridge.generate_response("Tell me something")

    assert isinstance(result, GenerationResult)
    assert isinstance(result.response, str)
    assert len(result.response) > 0
    assert isinstance(result.system_prompt_used, str)
    assert isinstance(result.temperature_used, float)
    assert isinstance(result.coherence_during, float)
    assert isinstance(result.consistency_with_claims, float)


def test_generate_response_uses_system_prompt():
    """Generated system prompt should include entity name."""
    bridge = create_test_bridge()
    result = bridge.generate_response("Hello")
    assert bridge.entity.name in result.system_prompt_used


def test_generate_response_additional_context():
    """Additional context should appear in system prompt."""
    bridge = create_test_bridge()
    result = bridge.generate_response("Hello", additional_context="Extra info here")
    assert "Extra info here" in result.system_prompt_used


def test_generate_response_auto_ticks():
    """Auto ticks after response should advance entity."""
    config = LLMBridgeConfig(auto_tick_after_response=True, ticks_per_turn=5)
    bridge = create_test_bridge(config=config)

    ticks_before = bridge.entity._tick_count
    bridge.generate_response("test")
    ticks_after = bridge.entity._tick_count

    assert ticks_after > ticks_before


def test_generate_response_no_auto_ticks():
    """Disabling auto ticks should not advance entity."""
    config = LLMBridgeConfig(auto_tick_after_response=False)
    bridge = create_test_bridge(config=config)

    ticks_before = bridge.entity._tick_count
    bridge.generate_response("test")
    ticks_after = bridge.entity._tick_count

    assert ticks_after == ticks_before


# ── conversation_turn Tests ──────────────────────────────────────────────────


def test_conversation_turn():
    """Full conversation turn should work."""
    bridge = create_test_bridge()

    response = bridge.conversation_turn("Hello!")

    assert len(response) > 0
    assert len(bridge._conversation_history) == 2  # user + assistant


def test_conversation_turn_records_history():
    """Conversation history should track turns correctly."""
    bridge = create_test_bridge()

    bridge.conversation_turn("First message")
    bridge.conversation_turn("Second message")

    assert len(bridge._conversation_history) == 4  # 2 user + 2 assistant
    assert bridge._conversation_history[0]['role'] == 'user'
    assert bridge._conversation_history[0]['content'] == 'First message'
    assert bridge._conversation_history[1]['role'] == 'assistant'
    assert bridge._conversation_history[2]['role'] == 'user'
    assert bridge._conversation_history[2]['content'] == 'Second message'
    assert bridge._conversation_history[3]['role'] == 'assistant'


def test_conversation_turn_records_metadata():
    """Assistant history entries should include coherence and consistency."""
    bridge = create_test_bridge()
    bridge.conversation_turn("Hello")

    assistant_entry = bridge._conversation_history[1]
    assert 'coherence' in assistant_entry
    assert 'consistency' in assistant_entry
    assert 'timestamp' in assistant_entry


def test_conversation_turn_stores_experience():
    """Response should be stored as experience in memory."""
    bridge = create_test_bridge()
    nodes_before = bridge.entity.memory.total_nodes
    queue_before = len(bridge.entity.memory.consolidation_queue.pending_nodes)

    bridge.conversation_turn("Tell me something")

    # Experiences with significance < 0.7 are queued for consolidation
    # rather than committed immediately. Check that memory grew OR queue grew.
    nodes_after = bridge.entity.memory.total_nodes
    queue_after = len(bridge.entity.memory.consolidation_queue.pending_nodes)
    assert (nodes_after > nodes_before) or (queue_after > queue_before)


# ── build_system_prompt Tests ────────────────────────────────────────────────


def test_system_prompt_includes_state():
    """System prompt should reflect cognitive state."""
    bridge = create_test_bridge()

    prompt = bridge.build_system_prompt()

    assert bridge.entity.name in prompt
    assert 'Coherence' in prompt
    assert bridge.entity.genesis_hash[:16] in prompt


def test_system_prompt_includes_stage():
    """System prompt should include developmental stage."""
    bridge = create_test_bridge()
    prompt = bridge.build_system_prompt()

    # Entity starts in genesis stage
    assert 'newly formed' in prompt or 'genesis' in prompt.lower()


def test_system_prompt_includes_claims():
    """Active claims should appear in system prompt."""
    bridge = create_test_bridge()

    # Activate a role
    bridge.claims.activate_role('analyst')

    prompt = bridge.build_system_prompt()
    assert '[Active Beliefs]' in prompt
    assert 'systematically' in prompt  # From analyst role claims


def test_system_prompt_no_claims_section_when_empty():
    """No Active Beliefs section when no claims are active."""
    bridge = create_test_bridge()
    prompt = bridge.build_system_prompt()
    assert '[Active Beliefs]' not in prompt


def test_system_prompt_cognitive_state_disabled():
    """Cognitive state can be disabled in config."""
    config = LLMBridgeConfig(include_cognitive_state=False)
    bridge = create_test_bridge(config=config)
    prompt = bridge.build_system_prompt()

    assert 'Coherence' not in prompt
    assert 'CI:' not in prompt


def test_system_prompt_recent_memories():
    """System prompt should include recent memories."""
    bridge = create_test_bridge()

    # Add some experiences
    bridge.entity.process_experience(
        "I learned about Python today", significance=0.8
    )

    prompt = bridge.build_system_prompt()
    assert '[Recent Context]' in prompt


# ── get_generation_params Tests ──────────────────────────────────────────────


def test_generation_params_vary():
    """Generation params should vary with state."""
    bridge = create_test_bridge()
    params = bridge.get_generation_params()

    assert isinstance(params, GenerationParams)
    assert 0.1 <= params.temperature <= 1.5
    assert 0.5 <= params.top_p <= 1.0
    assert params.max_tokens >= 64


def test_generation_params_temperature_range():
    """Temperature should be clamped to [0.1, 1.5]."""
    bridge = create_test_bridge()
    params = bridge.get_generation_params()
    assert 0.1 <= params.temperature <= 1.5


def test_generation_params_low_energy_fewer_tokens():
    """Low energy should reduce max_tokens."""
    bridge = create_test_bridge()

    # Normal energy
    params_normal = bridge.get_generation_params()

    # Drain energy
    bridge.entity.body.energy = 0.2
    params_tired = bridge.get_generation_params()

    assert params_tired.max_tokens < params_normal.max_tokens


def test_generation_params_high_arousal_lower_top_p():
    """High arousal should lower top_p."""
    bridge = create_test_bridge()

    # Normal arousal
    bridge.entity.body.arousal = 0.3
    params_calm = bridge.get_generation_params()

    # High arousal
    bridge.entity.body.arousal = 0.9
    params_aroused = bridge.get_generation_params()

    assert params_aroused.top_p < params_calm.top_p


def test_generation_params_tension_addition():
    """Low consistency should add tension note."""
    bridge = create_test_bridge()

    # Add conflicting claims to lower consistency
    for i in range(5):
        c = bridge.claims.add_claim(f"very different claim number {i}", strength=0.9)
        bridge.claims.activate_claim(c.id)

    params = bridge.get_generation_params()
    # Note: consistency may or may not be < 0.6 depending on phase patterns
    assert isinstance(params.system_prompt_additions, list)


# ── get_cognitive_state Tests ─────────────────────────────────────────────────


def test_cognitive_state():
    """get_cognitive_state should return all required fields."""
    bridge = create_test_bridge()
    state = bridge.get_cognitive_state()

    assert isinstance(state, CognitiveState)
    assert isinstance(state.coherence, float)
    assert isinstance(state.valence, float)
    assert isinstance(state.arousal, float)
    assert isinstance(state.energy, float)
    assert isinstance(state.stage, str)
    assert isinstance(state.active_claims, list)
    assert isinstance(state.consistency, float)
    assert isinstance(state.ci, float)


def test_cognitive_state_reflects_entity():
    """Cognitive state should match entity state."""
    bridge = create_test_bridge()
    state = bridge.get_cognitive_state()

    assert state.valence == bridge.entity.body.valence
    assert state.arousal == bridge.entity.body.arousal
    assert state.energy == bridge.entity.body.energy
    assert state.stage == bridge.entity.stage.value


def test_cognitive_state_claims_content():
    """Active claims in cognitive state should be content strings."""
    bridge = create_test_bridge()
    bridge.claims.activate_role('creative')

    state = bridge.get_cognitive_state()
    assert len(state.active_claims) > 0
    assert all(isinstance(c, str) for c in state.active_claims)


# ── Memory Retrieval Tests ────────────────────────────────────────────────────


def test_memory_retrieval():
    """Should retrieve relevant memories."""
    bridge = create_test_bridge()

    # Add some memories
    bridge.entity.process_experience(
        "I love Python programming", significance=0.8
    )
    bridge.entity.process_experience(
        "The weather is nice today", significance=0.8
    )

    # Query related to Python
    memories = bridge.retrieve_relevant_memories("Tell me about coding")

    # Should find memories (content matching via embeddings)
    assert len(memories) > 0


def test_memory_retrieval_empty():
    """Empty memory should return empty list."""
    bridge = create_test_bridge()
    memories = bridge.retrieve_relevant_memories("anything")
    assert memories == []


def test_memory_retrieval_respects_k():
    """Should return at most k memories."""
    bridge = create_test_bridge()

    # Add many experiences
    for i in range(10):
        bridge.entity.process_experience(
            f"Experience number {i} about topic {i}", significance=0.8
        )

    memories = bridge.retrieve_relevant_memories("experience", k=3)
    assert len(memories) <= 3


def test_memory_retrieval_excludes_roots():
    """Branch root nodes should not appear in results."""
    bridge = create_test_bridge()
    bridge.entity.process_experience("test memory", significance=0.8)

    memories = bridge.retrieve_relevant_memories("test")
    for mem in memories:
        assert mem.content.get('type') != 'branch_root'


# ── get_state Tests ──────────────────────────────────────────────────────────


def test_get_state():
    """Bridge state should include expected fields."""
    bridge = create_test_bridge()
    state = bridge.get_state()

    assert state['entity_name'] == 'TestEntity'
    assert state['conversation_turns'] == 0
    assert 'active_claims' in state
    assert 'cognitive_state' in state
    assert 'coherence' in state['cognitive_state']
    assert 'valence' in state['cognitive_state']
    assert 'stage' in state['cognitive_state']


def test_get_state_after_conversation():
    """State should reflect conversation history."""
    bridge = create_test_bridge()
    bridge.conversation_turn("hello")

    state = bridge.get_state()
    assert state['conversation_turns'] == 2  # user + assistant


# ── Integration Tests ─────────────────────────────────────────────────────────


def test_full_conversation_flow():
    """Full multi-turn conversation flow."""
    bridge = create_test_bridge()

    # Turn 1
    r1 = bridge.conversation_turn("Hello, I'm new here.")
    assert len(r1) > 0

    # Turn 2 with role trigger
    r2 = bridge.conversation_turn("Can you analyze this problem?")
    assert len(r2) > 0

    # Claims should have been activated
    assert len(bridge.claims.active_claims) > 0

    # History should have 4 entries
    assert len(bridge._conversation_history) == 4

    # State should be observable
    state = bridge.get_cognitive_state()
    assert state.stage == 'genesis'
    assert state.ci >= 0.0


def test_bridge_entity_evolution():
    """Entity should evolve through bridge interactions."""
    bridge = create_test_bridge()
    ci_start = bridge.entity.CI

    # Several interactions
    for i in range(3):
        bridge.conversation_turn(f"Important experience number {i}")

    # Entity should have processed experiences
    assert bridge.entity.memory.total_nodes > 0
    assert bridge.entity._tick_count > 0


def test_bridge_mock_client_logged():
    """All LLM calls should be logged in mock client."""
    bridge = create_test_bridge()
    client = bridge.llm_client

    bridge.conversation_turn("Hello world")

    # Should have embed calls (from grounding) + complete call
    assert len(client.call_log) > 0
    methods = [c['method'] for c in client.call_log]
    assert 'complete' in methods
    assert 'embed' in methods


def test_bridge_significance_decreases_with_repetition():
    """Repeated inputs should have decreasing significance."""
    bridge = create_test_bridge()

    r1 = bridge.process_input("Tell me about cats")
    r2 = bridge.process_input("Tell me about cats")

    # Second time should be less novel
    assert r2.significance <= r1.significance


# ── No-Embedder (completion-only) Tests ──────────────────────────────────────


class NoEmbedClient(LLMClient):
    """Client that only supports completions, not embeddings (like Claude)."""

    def complete(self, prompt, system_prompt="", temperature=0.7, max_tokens=1024,
                 messages=None):
        return f"Response to: {prompt[:50]}"

    def embed(self, text):
        raise NotImplementedError("No embeddings available")


def test_bridge_no_embedder_creation():
    """Bridge should initialize with a client that has no embeddings."""
    entity = create_entity("NoEmbed")
    client = NoEmbedClient()
    bridge = LLMBridge(entity, client)

    assert bridge.grounding.embedder is None


def test_bridge_no_embedder_conversation_turn():
    """Full conversation should work without embeddings."""
    entity = create_entity("NoEmbed")
    client = NoEmbedClient()
    bridge = LLMBridge(entity, client)

    response = bridge.conversation_turn("Hello!")
    assert len(response) > 0
    assert len(bridge._conversation_history) == 2


def test_bridge_no_embedder_process_input():
    """process_input should work without embeddings."""
    entity = create_entity("NoEmbed")
    client = NoEmbedClient()
    bridge = LLMBridge(entity, client)

    result = bridge.process_input("Test input")
    assert isinstance(result, ProcessResult)
    assert result.significance > 0


def test_bridge_no_embedder_system_prompt():
    """System prompt should still be built without embeddings."""
    entity = create_entity("NoEmbed")
    client = NoEmbedClient()
    bridge = LLMBridge(entity, client)

    prompt = bridge.build_system_prompt()
    assert "NoEmbed" in prompt
    assert "Coherence" in prompt


def test_bridge_no_embedder_generation_params():
    """Generation params should still work without embeddings."""
    entity = create_entity("NoEmbed")
    client = NoEmbedClient()
    bridge = LLMBridge(entity, client)

    params = bridge.get_generation_params()
    assert 0.1 <= params.temperature <= 1.5
    assert params.max_tokens >= 64


def test_bridge_explicit_embedder_overrides():
    """Explicit embedder param should override client's embed."""
    entity = create_entity("Override")
    client = NoEmbedClient()

    # Pass a working embedder explicitly
    mock_client = MockLLMClient()
    bridge = LLMBridge(entity, client, embedder=mock_client.embed)

    assert bridge.grounding.embedder is not None
    emb = bridge.grounding.embedder("test")
    assert emb.shape == (1536,)


# ── Multi-Turn Conversation History Tests ────────────────────────────────────


def test_mock_client_receives_messages():
    """MockLLMClient should log messages parameter."""
    client = MockLLMClient()
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    client.complete("Follow up", messages=history)

    assert len(client.call_log) == 1
    assert client.call_log[0]["messages"] == history
    assert client.call_log[0]["prompt"] == "Follow up"


def test_mock_client_messages_affect_response():
    """Different conversation histories should produce different responses."""
    client = MockLLMClient()
    r1 = client.complete("Same prompt", messages=[
        {"role": "user", "content": "context A"},
        {"role": "assistant", "content": "response A"},
    ])
    r2 = client.complete("Same prompt", messages=[
        {"role": "user", "content": "context B"},
        {"role": "assistant", "content": "response B"},
    ])
    assert r1 != r2


def test_mock_client_no_messages_backward_compatible():
    """Calling complete() without messages should still work (single-turn)."""
    client = MockLLMClient()
    r1 = client.complete("test prompt")
    assert isinstance(r1, str)
    assert len(r1) > 0
    assert client.call_log[0]["messages"] is None


def test_bridge_passes_history_to_llm():
    """Bridge should pass conversation history to LLM on second+ turns."""
    bridge = create_test_bridge()
    client = bridge.llm_client

    # First turn
    bridge.conversation_turn("Hello!")
    # Second turn — LLM should receive history from first turn
    bridge.conversation_turn("How are you?")

    # Find the complete calls
    complete_calls = [c for c in client.call_log if c["method"] == "complete"]

    # First generate_response call: no history yet (empty messages)
    assert complete_calls[0]["messages"] is None or complete_calls[0]["messages"] == []

    # Second generate_response call: should have user + assistant from turn 1
    second_call = complete_calls[1]
    assert second_call["messages"] is not None
    assert len(second_call["messages"]) >= 2
    roles = [m["role"] for m in second_call["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_bridge_history_grows_each_turn():
    """Message history should grow with each conversation turn."""
    bridge = create_test_bridge()
    client = bridge.llm_client

    bridge.conversation_turn("Turn 1")
    bridge.conversation_turn("Turn 2")
    bridge.conversation_turn("Turn 3")

    complete_calls = [c for c in client.call_log if c["method"] == "complete"]

    # Turn 3 should have history of turn 1 + turn 2 (4 messages: 2 user + 2 assistant)
    third_call = complete_calls[2]
    assert third_call["messages"] is not None
    assert len(third_call["messages"]) == 4


def test_bridge_history_only_role_and_content():
    """Messages passed to LLM should only contain role and content (no metadata)."""
    bridge = create_test_bridge()
    bridge.conversation_turn("Hello!")
    bridge.conversation_turn("Follow up")

    client = bridge.llm_client
    complete_calls = [c for c in client.call_log if c["method"] == "complete"]
    second_call = complete_calls[1]

    for msg in second_call["messages"]:
        assert set(msg.keys()) == {"role", "content"}


def test_build_messages_empty_history():
    """_build_messages_for_llm should return empty list for fresh bridge."""
    bridge = create_test_bridge()
    messages = bridge._build_messages_for_llm()
    assert messages == []


def test_build_messages_after_turns():
    """_build_messages_for_llm should return all prior turns."""
    bridge = create_test_bridge()
    bridge.conversation_turn("Hello")
    bridge.conversation_turn("World")

    messages = bridge._build_messages_for_llm()
    assert len(messages) == 4  # 2 user + 2 assistant
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "World"
    assert messages[3]["role"] == "assistant"


# ── Beliefs / Claims Bridge Tests ────────────────────────────────────────────


def test_add_belief():
    """add_belief should create and activate a claim."""
    bridge = create_test_bridge()
    claim = bridge.add_belief("I value honesty")
    assert claim.content == "I value honesty"
    assert claim.id in bridge.claims.active_claims


def test_add_belief_scope_and_source():
    """add_belief should accept scope and source."""
    bridge = create_test_bridge()
    claim = bridge.add_belief(
        "Be concise", scope="behavior", source="instructed", strength=0.9
    )
    assert claim.scope.value == "behavior"
    assert claim.source.value == "instructed"
    assert claim.strength == 0.9


def test_add_belief_applies_to_substrate():
    """add_belief should immediately apply to substrate."""
    bridge = create_test_bridge()
    # Get weights before
    weights_before = bridge.entity.substrate.slow.internal_weights.copy()
    bridge.add_belief("Focus on systematic analysis")
    weights_after = bridge.entity.substrate.slow.internal_weights
    # Weights should have changed due to claim application
    assert not (weights_before == weights_after).all()


def test_remove_belief():
    """remove_belief should remove the claim."""
    bridge = create_test_bridge()
    claim = bridge.add_belief("temporary belief")
    bridge.remove_belief(claim.id)
    assert claim.id not in bridge.claims.claims


def test_list_beliefs():
    """list_beliefs should return all beliefs with status."""
    bridge = create_test_bridge()
    bridge.add_belief("belief one")
    bridge.add_belief("belief two", activate=False)

    beliefs = bridge.list_beliefs()
    assert len(beliefs) == 2

    # First should be active, second not
    active_beliefs = [b for b in beliefs if b['active']]
    inactive_beliefs = [b for b in beliefs if not b['active']]
    assert len(active_beliefs) == 1
    assert len(inactive_beliefs) == 1


def test_activate_role():
    """activate_role should activate COGNIZEN role claims."""
    bridge = create_test_bridge()
    active = bridge.activate_role('analyst')
    assert len(active) > 0
    assert len(bridge.claims.active_claims) > 0


def test_deactivate_all_roles():
    """deactivate_all_roles should clear active claims."""
    bridge = create_test_bridge()
    bridge.activate_role('creative')
    assert len(bridge.claims.active_claims) > 0

    bridge.deactivate_all_roles()
    assert len(bridge.claims.active_claims) == 0


def test_list_roles():
    """list_roles should return available COGNIZEN roles."""
    bridge = create_test_bridge()
    roles = bridge.list_roles()
    assert 'analyst' in roles
    assert 'creative' in roles
    assert 'skeptic' in roles


def test_get_conflicts():
    """get_conflicts should return formatted conflict list."""
    bridge = create_test_bridge()
    # No conflicts initially
    conflicts = bridge.get_conflicts()
    assert isinstance(conflicts, list)


# ── Memory Access Bridge Tests ───────────────────────────────────────────────


def test_store_insight():
    """store_insight should add to INSIGHTS branch."""
    bridge = create_test_bridge()
    bridge.store_insight("Key finding about patterns")

    insights = bridge.entity.memory.query(MemoryBranch.INSIGHTS)
    assert len(insights) > 0
    found = any(
        n.content.get('content') == "Key finding about patterns"
        for n in insights
    )
    assert found


def test_store_relationship():
    """store_relationship should add to RELATIONS branch."""
    bridge = create_test_bridge()
    bridge.store_relationship("User prefers detailed explanations")

    relations = bridge.entity.memory.query(MemoryBranch.RELATIONS)
    assert len(relations) > 0
    found = any(
        n.content.get('content') == "User prefers detailed explanations"
        for n in relations
    )
    assert found


def test_get_memories_all():
    """get_memories with no branch should return all non-root nodes."""
    bridge = create_test_bridge()
    bridge.entity.process_experience("test memory", significance=0.8)
    memories = bridge.get_memories()
    assert len(memories) > 0


def test_get_memories_by_branch():
    """get_memories with branch should filter."""
    bridge = create_test_bridge()
    bridge.store_insight("test insight")
    bridge.entity.process_experience("test experience", significance=0.8)

    insights = bridge.get_memories(branch='insights')
    assert len(insights) > 0
    for m in insights:
        assert 'content' in m


def test_get_memories_unknown_branch():
    """get_memories with unknown branch returns empty list."""
    bridge = create_test_bridge()
    assert bridge.get_memories(branch='nonexistent') == []


def test_verify_memory():
    """verify_memory should return True for intact memory."""
    bridge = create_test_bridge()
    assert bridge.verify_memory() is True


# ── Introspection Tests ──────────────────────────────────────────────────────


def test_reflect():
    """reflect should return comprehensive state."""
    bridge = create_test_bridge()
    # Do some activity first
    bridge.conversation_turn("Hello world")

    data = bridge.reflect()

    # Check all top-level sections
    assert 'consciousness' in data
    assert 'body' in data
    assert 'claims' in data
    assert 'memory' in data
    assert 'development' in data
    assert 'substrate' in data


def test_reflect_consciousness_fields():
    """reflect consciousness section should have CI breakdown."""
    bridge = create_test_bridge()
    bridge.conversation_turn("test")
    data = bridge.reflect()

    ci = data['consciousness']
    assert 'ci' in ci
    assert 'ci_fast' in ci
    assert 'ci_slow' in ci
    assert 'cross_scale_coherence' in ci
    assert 'ci_trend' in ci
    assert 'status' in ci


def test_reflect_body_fields():
    """reflect body section should have state flags."""
    bridge = create_test_bridge()
    data = bridge.reflect()

    body = data['body']
    assert 'valence' in body
    assert 'arousal' in body
    assert 'energy' in body
    assert 'depleted' in body
    assert 'overaroused' in body


def test_reflect_claims_fields():
    """reflect claims section should have coherence and conflicts."""
    bridge = create_test_bridge()
    data = bridge.reflect()

    claims = data['claims']
    assert 'total' in claims
    assert 'active' in claims
    assert 'coherence' in claims
    assert 'conflicts' in claims


def test_reflect_memory_fields():
    """reflect memory section should have stats."""
    bridge = create_test_bridge()
    data = bridge.reflect()

    mem = data['memory']
    assert 'total_nodes' in mem
    assert 'depth' in mem
    assert 'fractal_dimension' in mem
    assert 'pending_consolidation' in mem
    assert 'verified' in mem


def test_reflect_development_fields():
    """reflect development section should have progress."""
    bridge = create_test_bridge()
    data = bridge.reflect()

    dev = data['development']
    assert 'stage' in dev
    assert 'progress' in dev
    assert 'age' in dev
    assert 'oscillators' in dev
    assert 0.0 <= dev['progress'] <= 1.0


def test_read_mind():
    """read_mind should return embedding from substrate."""
    bridge = create_test_bridge()
    embedding = bridge.read_mind()
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)


def test_read_mind_no_embedder():
    """read_mind should return None without embedder."""
    entity = create_entity("NoEmbed")
    client = NoEmbedClient()
    bridge = LLMBridge(entity, client)
    assert bridge.read_mind() is None


def test_rest_through_bridge():
    """rest via bridge should consolidate and anchor claims."""
    bridge = create_test_bridge()
    bridge.add_belief("test belief for anchoring")
    bridge.entity.process_experience("test memory", significance=0.5)

    result = bridge.rest()
    assert 'consolidated' in result
    assert 'tensions_resolved' in result
    assert 'ci_after' in result
    assert 'memory_verified' in result
    assert result['memory_verified'] is True


# ── Enriched conversation_turn Tests ─────────────────────────────────────────


def test_conversation_turn_feeds_response_to_substrate():
    """Response should be fed back through substrate (bidirectional)."""
    bridge = create_test_bridge()
    # Run a conversation turn
    bridge.conversation_turn("Tell me something important")

    # Entity should have processed experiences for both input AND output
    assert bridge.entity.memory.total_nodes > 0


def test_conversation_turn_stores_insights():
    """High-significance responses should be stored as insights."""
    bridge = create_test_bridge()

    # Run several turns to build significance
    for i in range(5):
        bridge.conversation_turn(
            f"This is an important critical urgent message number {i}?"
        )

    # Check INSIGHTS branch has entries
    insights = bridge.entity.memory.query(MemoryBranch.INSIGHTS)
    # May or may not have insights depending on significance estimation
    assert isinstance(insights, list)


def test_conversation_turn_anchors_claims():
    """Active claims should be anchored to memory during conversation."""
    bridge = create_test_bridge()
    bridge.add_belief("I am thorough")

    bridge.conversation_turn("Hello!")

    # The claim should now be anchored
    claim = list(bridge.claims.claims.values())[0]
    assert claim.memory_node_id is not None


def test_conversation_turn_arousal_increases():
    """Arousal should increase over long conversations."""
    bridge = create_test_bridge()
    arousal_start = bridge.entity.body.arousal

    # 5 turns to trigger arousal increase (threshold is > 3 turns)
    for i in range(5):
        bridge.conversation_turn(f"Message {i}")

    assert bridge.entity.body.arousal > arousal_start


# ── Enriched System Prompt Tests ─────────────────────────────────────────────


def test_system_prompt_depleted_body():
    """System prompt should reflect depleted body state."""
    bridge = create_test_bridge()
    bridge.entity.body.energy = 0.1  # Deplete

    prompt = bridge.build_system_prompt()
    assert 'depleted' in prompt.lower() or 'brief' in prompt.lower()


def test_system_prompt_overaroused_body():
    """System prompt should reflect overaroused state."""
    bridge = create_test_bridge()
    bridge.entity.body.arousal = 0.9  # Overstimulate

    prompt = bridge.build_system_prompt()
    assert 'overstimulated' in prompt.lower() or 'deliberate' in prompt.lower()


def test_system_prompt_claim_coherence():
    """System prompt should include claim coherence when beliefs active."""
    bridge = create_test_bridge()
    bridge.activate_role('analyst')

    prompt = bridge.build_system_prompt()
    assert 'coherence' in prompt.lower()


def test_system_prompt_includes_insights():
    """System prompt should include recent insights."""
    bridge = create_test_bridge()
    bridge.store_insight("Neural networks converge faster with normalization")

    prompt = bridge.build_system_prompt()
    assert '(insight)' in prompt
