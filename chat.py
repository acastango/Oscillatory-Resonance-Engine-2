#!/usr/bin/env python3
"""
Interactive chat with an ORE-grounded entity.

Usage:
    # With Claude (requires ANTHROPIC_API_KEY env var):
    python chat.py

    # With Claude, specific model:
    python chat.py --model claude-sonnet-4-20250514

    # With mock client (no API key needed):
    python chat.py --mock

    # With Ollama (requires Ollama running locally):
    python chat.py --ollama --ollama-model llama2

    # Name your entity:
    python chat.py --name Athena

    # Show cognitive state after each turn:
    python chat.py --show-state
"""

import argparse
import sys

from ore2.core.entity import create_entity
from ore2.core.llm_bridge import LLMBridge, LLMBridgeConfig
from ore2.core.llm_clients import MockLLMClient


def create_client(args):
    """Create the appropriate LLM client based on CLI args."""
    if args.mock:
        print("[Using MockLLMClient - no API key needed]")
        return MockLLMClient()
    elif args.ollama:
        from ore2.core.llm_clients import OllamaClient
        model = args.ollama_model or "llama2"
        url = args.ollama_url or "http://localhost:11434"
        print(f"[Using Ollama: {model} at {url}]")
        return OllamaClient(model=model, base_url=url)
    else:
        from ore2.core.llm_clients import ClaudeClient
        model = args.model or "claude-sonnet-4-20250514"
        print(f"[Using Claude: {model}]")
        return ClaudeClient(model=model)


def format_state(bridge):
    """Format cognitive state for display."""
    state = bridge.get_cognitive_state()
    parts = [
        f"  Coherence: {state.coherence:.3f}",
        f"  CI: {state.ci:.4f}",
        f"  Valence: {state.valence:.3f}",
        f"  Energy: {state.energy:.3f}",
        f"  Stage: {state.stage}",
    ]
    if state.active_claims:
        parts.append(f"  Claims: {', '.join(state.active_claims[:3])}")
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description="Chat with an ORE-grounded entity")
    parser.add_argument("--name", default="Omega", help="Entity name (default: Omega)")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API key)")
    parser.add_argument("--model", default=None, help="Claude model name")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama")
    parser.add_argument("--ollama-model", default=None, help="Ollama model name")
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL")
    parser.add_argument("--show-state", action="store_true", help="Show cognitive state")
    args = parser.parse_args()

    # Create entity and bridge
    entity = create_entity(args.name)
    client = create_client(args)
    bridge = LLMBridge(entity, client)

    print(f"\n{'=' * 60}")
    print(f"  {entity.name} is online.")
    print(f"  Genesis: {entity.genesis_hash[:16]}...")
    print(f"  Stage: {entity.stage.value}")
    print(f"{'=' * 60}")
    print("  Type 'quit' or 'exit' to end.")
    print("  Type '/state' to see cognitive state.")
    print("  Type '/rest' to trigger consolidation.")
    print(f"{'=' * 60}\n")

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye.")
            break

        if user_input == "/state":
            print(f"\n[Cognitive State]\n{format_state(bridge)}\n")
            continue

        if user_input == "/rest":
            result = entity.rest()
            print(f"\n[Rest cycle complete]")
            print(f"  Consolidated: {result.get('memory', {}).get('consolidated', 0)}")
            print(f"  Tensions resolved: {result.get('memory', {}).get('tensions_resolved', 0)}\n")
            continue

        # Conversation turn
        turn += 1
        response = bridge.conversation_turn(user_input)
        print(f"\n{entity.name}: {response}\n")

        if args.show_state:
            print(f"[State] {format_state(bridge)}\n")


if __name__ == "__main__":
    main()
