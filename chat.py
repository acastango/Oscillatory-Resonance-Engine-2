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
import json

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


def format_reflect(data):
    """Format reflect() output for display."""
    lines = []

    c = data['consciousness']
    lines.append("[Consciousness]")
    lines.append(f"  CI: {c['ci']:.4f} (fast={c['ci_fast']:.4f}, slow={c['ci_slow']:.4f})")
    lines.append(f"  Cross-scale coherence: {c['cross_scale_coherence']:.3f}")
    trend = c['ci_trend']
    trend_str = "rising" if trend > 0.001 else ("falling" if trend < -0.001 else "stable")
    lines.append(f"  Trend: {trend_str} ({trend:+.4f})")

    b = data['body']
    lines.append("\n[Body]")
    lines.append(f"  Valence: {b['valence']:.3f}  Arousal: {b['arousal']:.3f}  Energy: {b['energy']:.3f}")
    flags = []
    if b['depleted']:
        flags.append("DEPLETED")
    if b['overaroused']:
        flags.append("OVERAROUSED")
    if flags:
        lines.append(f"  Flags: {', '.join(flags)}")

    cl = data['claims']
    lines.append("\n[Claims]")
    lines.append(f"  Total: {cl['total']}  Active: {cl['active']}  Coherence: {cl['coherence']:.3f}")
    if cl['conflicts']:
        lines.append(f"  Conflicts: {len(cl['conflicts'])}")
        for conf in cl['conflicts'][:3]:
            lines.append(f"    {conf['claim_a']}  vs  {conf['claim_b']}")

    m = data['memory']
    lines.append("\n[Memory]")
    lines.append(f"  Nodes: {m['total_nodes']}  Depth: {m['depth']}  Fractal D: {m['fractal_dimension']:.2f}")
    lines.append(f"  Grain boundaries: {m['grain_boundaries']}  Pending: {m['pending_consolidation']}")
    lines.append(f"  Verified: {m['verified']}")

    d = data['development']
    lines.append("\n[Development]")
    lines.append(f"  Stage: {d['stage']} ({d['progress']*100:.1f}% progress)")
    lines.append(f"  Age: {d['age']:.1f}  Oscillators: {d['oscillators']}")

    s = data['substrate']
    lines.append("\n[Substrate]")
    lines.append(f"  Coherence: {s['coherence']:.3f}  Loop: {s['loop_coherence']:.3f}")
    lines.append(f"  Active: fast={s['fast_active']}, slow={s['slow_active']}")

    return '\n'.join(lines)


HELP_TEXT = """
Commands:
  /help                    Show this help
  /state                   Show cognitive state (brief)
  /reflect                 Deep introspection (full state breakdown)
  /witness                 Full entity witness display

  /believe <text>          Add a belief that shapes dynamics
  /beliefs                 List all beliefs
  /role <name>             Activate a COGNIZEN role
  /roles                   List available roles
  /unrole                  Deactivate all roles
  /conflicts               Show conflicting beliefs

  /memories [branch]       Show memories (branches: self, relations, insights, experiences)
  /insight <text>          Store an insight
  /relationship <text>     Store a relationship observation
  /verify                  Verify memory integrity

  /rest                    Trigger rest/consolidation cycle

  quit, exit               End session
""".strip()


def handle_command(user_input, bridge, entity):
    """Handle slash commands. Returns True if command was handled."""
    if not user_input.startswith("/"):
        return False

    parts = user_input.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        print(f"\n{HELP_TEXT}\n")

    elif cmd == "/state":
        print(f"\n[Cognitive State]\n{format_state(bridge)}\n")

    elif cmd == "/reflect":
        data = bridge.reflect()
        print(f"\n{format_reflect(data)}\n")

    elif cmd == "/witness":
        print(entity.witness())

    elif cmd == "/believe":
        if not arg:
            print("\nUsage: /believe <text>\n")
        else:
            claim = bridge.add_belief(arg)
            print(f"\n[Belief added] {claim.id[:12]}... : {claim.content}")
            print(f"  Scope: {claim.scope.value}  Strength: {claim.strength}\n")

    elif cmd == "/beliefs":
        beliefs = bridge.list_beliefs()
        if not beliefs:
            print("\n[No beliefs]\n")
        else:
            print(f"\n[Beliefs] ({len(beliefs)} total)")
            for b in beliefs:
                status = "*" if b['active'] else " "
                anchor = "A" if b['anchored'] else " "
                print(f"  [{status}{anchor}] {b['id'][:12]}... {b['content']}")
                print(f"       scope={b['scope']} str={b['strength']:.2f} src={b['source']}")
            print()

    elif cmd == "/role":
        if not arg:
            print(f"\nUsage: /role <name>")
            print(f"Available: {', '.join(bridge.list_roles())}\n")
        else:
            try:
                active = bridge.activate_role(arg.strip())
                print(f"\n[Role '{arg.strip()}' activated] {len(active)} active claims\n")
            except (KeyError, ValueError) as e:
                print(f"\nError: {e}")
                print(f"Available: {', '.join(bridge.list_roles())}\n")

    elif cmd == "/roles":
        roles = bridge.list_roles()
        print(f"\n[Available roles] {', '.join(roles)}\n")

    elif cmd == "/unrole":
        bridge.deactivate_all_roles()
        print("\n[All roles deactivated]\n")

    elif cmd == "/conflicts":
        conflicts = bridge.get_conflicts()
        if not conflicts:
            print("\n[No conflicts detected]\n")
        else:
            print(f"\n[{len(conflicts)} conflict(s)]")
            for c in conflicts:
                print(f"  {c['claim_a']}")
                print(f"    vs {c['claim_b']}")
            print()

    elif cmd == "/memories":
        branch = arg.strip() if arg else None
        memories = bridge.get_memories(branch=branch, k=10)
        if not memories:
            label = f" ({branch})" if branch else ""
            print(f"\n[No memories{label}]\n")
        else:
            label = f" ({branch})" if branch else ""
            print(f"\n[Memories{label}] ({len(memories)} shown)")
            for m in memories:
                print(f"  [{m['id']}] C={m['coherence']:.3f} {m['content']}")
            print()

    elif cmd == "/insight":
        if not arg:
            print("\nUsage: /insight <text>\n")
        else:
            bridge.store_insight(arg)
            print(f"\n[Insight stored]\n")

    elif cmd == "/relationship":
        if not arg:
            print("\nUsage: /relationship <text>\n")
        else:
            bridge.store_relationship(arg)
            print(f"\n[Relationship stored]\n")

    elif cmd == "/verify":
        verified = bridge.verify_memory()
        status = "VERIFIED" if verified else "FAILED"
        print(f"\n[Memory integrity: {status}]\n")

    elif cmd == "/rest":
        result = bridge.rest()
        print(f"\n[Rest cycle complete]")
        print(f"  Consolidated: {result['consolidated']}")
        print(f"  Tensions resolved: {result['tensions_resolved']}")
        print(f"  CI after: {result['ci_after']:.4f}")
        print(f"  Memory verified: {result['memory_verified']}\n")

    else:
        print(f"\nUnknown command: {cmd}")
        print(f"Type /help for available commands.\n")

    return True


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
    print("  Type /help for commands, or just chat.")
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

        # Handle slash commands
        if handle_command(user_input, bridge, entity):
            continue

        # Conversation turn
        turn += 1
        response = bridge.conversation_turn(user_input)
        print(f"\n{entity.name}: {response}\n")

        if args.show_state:
            print(f"[State]\n{format_state(bridge)}\n")


if __name__ == "__main__":
    main()
