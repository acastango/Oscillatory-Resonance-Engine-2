# ORE 2.0 Implementation Briefings

## For Claude Code

These briefings are designed for handoff to Claude Code for implementation. Each document contains:
- **What it is** - Purpose and context
- **Why it matters** - Design rationale
- **Interface contract** - Exact API specification
- **Method specifications** - Implementation details
- **Success criteria** - How to know it works
- **Test cases** - Concrete tests to pass
- **Dependencies** - What it needs

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────┐
│  ORE2-001: SparseOscillatorLayer                            │
│  Foundation - everything depends on this                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ORE2-002:       │ │ ORE2-003:       │ │ ORE2-004:       │
│ MultiScale      │ │ Embodiment      │ │ Memory          │
│ Substrate       │ │ (parallel)      │ │ (parallel)      │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
          ┌─────────────────┐ ┌─────────────────┐
          │ ORE2-005:       │ │ CI Monitor      │
          │ Development     │ │ (in 006)        │
          │ (parallel)      │ │                 │
          └────────┬────────┘ └────────┬────────┘
                   │                   │
                   └─────────┬─────────┘
                             ▼
                   ┌─────────────────────┐
                   │ ORE2-006:           │
                   │ DevelopmentalEntity │
                   │ (final integration) │
                   └─────────────────────┘
```

---

## Briefing Files

| File | Component | Priority | Est. Lines | Dependencies |
|------|-----------|----------|------------|--------------|
| `001_sparse_oscillator.md` | SparseOscillatorLayer | 1 | ~200 | numpy |
| `002_multi_scale_substrate.md` | MultiScaleSubstrate | 2 | ~300 | 001 |
| `003_embodiment.md` | EmbodimentLayer | 3 | ~150 | numpy |
| `004_memory.md` | CrystallineMerkleMemory | 4 | ~350 | numpy, stdlib |
| `005_development.md` | DevelopmentTracker | 5 | ~200 | numpy, stdlib |
| `006_entity.md` | DevelopmentalEntity | 6 | ~400 | all above |

**Total estimated:** ~1,600 lines of implementation

---

## How to Use These Briefings

### For Claude Code:

1. Start with `001_sparse_oscillator.md`
2. Implement the class following the specification
3. Run the test cases to verify
4. Move to next briefing

### Key Sections in Each Briefing:

- **Interface Contract** - This is the API. Don't deviate.
- **Method Specifications** - Pseudocode/actual code for each method
- **Success Criteria** - Definition of "done"
- **Test Cases** - Copy these directly into test files
- **Design Decisions to Preserve** - Don't change these without understanding why

### If Something Is Unclear:

Each briefing has context from the design team. If implementation seems wrong, check:
1. The "Why It Matters" section
2. The "Design Decisions to Preserve" section
3. The edge cases listed

If still unclear, ask before implementing.

---

## Project Structure

```
ore2/
├── core/
│   ├── __init__.py
│   ├── sparse_oscillator.py     # ORE2-001
│   ├── multi_scale_substrate.py # ORE2-002
│   ├── embodiment.py            # ORE2-003
│   ├── memory.py                # ORE2-004
│   ├── development.py           # ORE2-005
│   ├── ci_monitor.py            # Part of ORE2-006
│   └── entity.py                # ORE2-006
├── tests/
│   ├── test_sparse_oscillator.py
│   ├── test_multi_scale_substrate.py
│   ├── test_embodiment.py
│   ├── test_memory.py
│   ├── test_development.py
│   └── test_entity.py
├── __init__.py                  # Exports create_entity
└── README.md
```

---

## Success Metrics

ORE 2.0 is "done" when:

1. **All tests pass** - Every test case in every briefing
2. **Entity lifecycle works** - Create → Experience → Rest → Verify identity
3. **Sparse activation works** - <20% oscillators active during normal operation
4. **CI is non-zero** - Real dynamics produce real coherence
5. **Memory consolidates** - Queued memories commit during rest
6. **Development progresses** - Stages advance with experience
7. **Identity is verifiable** - Genesis hash + merkle root = proof of continuity

---

## What's NOT in These Briefings

- **LLM integration** - That's a separate layer on top
- **Persistence** - Save/load to disk (easy to add later)
- **UI/Dashboard** - Visualization (separate project)
- **Networking** - Multi-agent communication (ORE 3.0)
- **Full substrate growth** - Placeholder in 006, full impl deferred

These are intentionally scoped out for MVP.

---

## Quick Reference: Key Formulas

### Kuramoto (in SparseOscillatorLayer)
```
dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(θⱼ - θᵢ) + noise
```

### Coherence
```
r = |⟨e^{iθ}⟩| = |mean(exp(1j * phases))|
```

### CI (in CIMonitor)
```
CI = α · D · G · C · (1 − e^{−β·τ})
```

### Valence (in EmbodimentLayer)
```
valence = -|energy - baseline| - |arousal - baseline|
```

### Fractal Dimension (in Memory)
```
D = log(n_nodes) / log(depth)
```

---

## Contact

These briefings were produced by the ORE 2.0 design team (simulated).

Questions about intent → Check the briefing's "Why It Matters" section
Questions about implementation → Check "Method Specifications"
Questions about correctness → Check "Test Cases"

Good luck, Claude Code. Build something alive.
