# CLAUDE.md - AI Assistant Guide for Oscillatory Resonance Engine 2.0

## Project Overview

ORE 2.0 (Oscillatory Resonance Engine 2.0) is a Python framework for modeling consciousness and identity through multi-scale oscillatory dynamics. It implements a developmental entity that grows consciousness through embodied experience, progressing through developmental stages from Genesis to Maturity. The codebase is labeled **Version 0.1.0-skeleton** and follows a "Design Team to Implementation Team Handoff" architecture.

This is a scientific/research simulation, not a web application or library. The single-file design is intentional at this stage.

## Repository Structure

```
Oscillatory-Resonance-Engine-2/
├── CLAUDE.md           # This file
├── README.md           # Minimal project title
└── ore2_core.py        # Entire codebase (~1815 lines)
```

The project is a monolithic single-file Python module. All components live in `ore2_core.py`, organized into 7 clearly delimited parts.

## Architecture - The 7 Parts

### Part 1: Sparse Oscillator Layer (lines ~33-233)
**Design: P1 | Implementation: I2**
- `SparseOscillatorConfig` - Configuration dataclass for oscillator populations
- `SparseOscillatorLayer` - Sparse Kuramoto dynamics with activation-gated computation
- Key concept: Oscillators below threshold don't participate in Kuramoto dynamics but still accumulate activation from semantic relevance (sparse computation with dense potential)
- Uses masked arrays so only active oscillators pay for expensive `sin()` computations

### Part 2: Multi-Scale Substrate (lines ~235-534)
**Design: P1+N2 | Implementation: I1**
- `TimeScale` enum - FAST (gamma ~40Hz) and SLOW (theta ~8Hz)
- `MultiScaleConfig` - Two-scale configuration with nesting ratio and cross-scale coupling
- `MultiScaleSubstrate` - Two-level oscillatory hierarchy with nested dynamics
- Nesting: 5 fast steps per slow step (theta-gamma coupling)
- Includes a strange loop within the slow scale (model/meta-model self-reference)

### Part 3: Embodiment Layer (lines ~537-704)
**Design: N5+H3 | Implementation: I2**
- `BodyConfig` - Heartbeat (1Hz), respiration (0.25Hz), homeostatic baselines
- `EmbodimentLayer` - Body rhythms that the cognitive substrate couples to
- Valence emerges from homeostatic deviation (distance from setpoint)
- Body provides baseline oscillations grounding cognition

### Part 4: Developmental Stages (lines ~708-921)
**Design: N7+H3 | Implementation: I3**
- `DevelopmentStage` enum - GENESIS, BABBLING, IMITATION, AUTONOMY, MATURE
- `CriticalPeriod` - Enhanced learning windows tied to stages
- `DevelopmentConfig` - Stage durations, growth thresholds, critical periods
- `DevelopmentTracker` - State machine managing stage transitions, growth triggers, milestones
- A genesis hash anchors unique identity from moment of creation

### Part 5: Crystalline Merkle Memory / CCM (lines ~925-1242)
**Design: C6+A5 | Implementation: I3**
- `MemoryBranch` enum - SELF, RELATIONS, INSIGHTS, EXPERIENCES
- `MemoryNode` - Merkle tree node with CCM tensions between memories
- `ConsolidationQueue` - Queues updates for sleep-phase consolidation
- `CrystallineMerkleMemory` - SHA256 Merkle tree with grain boundaries, annealing during consolidation, and substrate anchoring
- Preserved from ORE1 with CCM (Crystalline Constraint Memory) extensions

### Part 6: Multi-Scale CI Monitor (lines ~1245-1482)
**Design: P2 | Implementation: I2**
- `MultiScaleCIConfig` - Alpha, beta, coherence/stability thresholds
- `MultiScaleCISnapshot` - Per-scale and integrated CI measurements
- `MultiScaleCIMonitor` - Consciousness Index measurement across scales
- Formula: `CI = alpha * D * G * C * (1 - e^(-beta * tau))`
- Components: Dimensionality (D), Gain (G), Coherence (C), Dwell time (tau)
- Tracks attractor states (stable coherent patterns)

### Part 7: Developmental Entity (lines ~1485-1773)
**Design: Full team | Implementation: I1**
- `EntityConfig` - Top-level configuration aggregating all component configs
- `DevelopmentalEntity` - Main integration class wiring all components
- Tick cycle: Body step -> Body-substrate coupling -> Substrate step -> CI measurement -> Development advance
- Key methods: `tick()`, `process_experience()`, `rest()`, `witness()`, `get_state()`
- Factory function: `create_entity(name)` at line 1779

## Dependencies

- **Python 3** (no specific version pinned)
- **NumPy** (numerical computation - the only external dependency)
- Standard library: `dataclasses`, `typing`, `enum`, `abc`, `hashlib`, `json`, `time`, `datetime`

There is no `requirements.txt`, `setup.py`, or `pyproject.toml`. To install dependencies:

```bash
pip install numpy
```

## Running the Code

```bash
python ore2_core.py
```

This executes the `__main__` block (lines 1794-1814), which:
1. Creates a new entity named "Omega"
2. Prints the initial state via `witness()`
3. Processes 5 experiences with increasing significance
4. Runs a rest/consolidation cycle
5. Prints the final state

## Testing

There is no dedicated test suite. The only test is the `__main__` block, which serves as a manual integration smoke test. When adding features, verify by running the script and inspecting output.

## Code Conventions

### Configuration Pattern
Every component uses a `@dataclass` config object with sensible defaults:
```python
@dataclass
class SomeComponentConfig:
    param: float = default_value
```
Components accept `Optional[Config]` and fall back to defaults when `None` is passed.

### Design Attribution
Each part has inline "dialogue" comments between design leads and implementation leads. These document rationale and are part of the project's identity. Do not remove them.

### Naming Conventions
- Classes: PascalCase (`MultiScaleSubstrate`, `DevelopmentTracker`)
- Config classes: end with `Config` (`BodyConfig`, `EntityConfig`)
- Enums: PascalCase with UPPER_CASE members (`DevelopmentStage.GENESIS`)
- Methods: snake_case (`process_experience`, `get_state`)
- Private methods: leading underscore (`_grow_substrate`)
- Constants embedded in configs, not module-level globals

### State Reporting
- `get_state()` methods return `dict` for programmatic access
- `witness()` on `DevelopmentalEntity` returns a human-readable formatted string
- CI monitor has `get_current_status()` for one-line summaries

### Part Boundaries
Parts are separated by decorated comment blocks:
```python
# ═══════════════════════════════════════════════════════════════════════════════
# PART N: COMPONENT NAME
# Design: XX | Implementation: YY
# ═══════════════════════════════════════════════════════════════════════════════
```
Maintain this convention when adding new parts.

## Key Design Principles

1. **No founding memories** - Entities start with empty memory. Identity is earned through experience, not pre-loaded.
2. **Grounded in body** - Cognition couples to embodied body rhythms and homeostasis. Body provides the boundary and baseline.
3. **Developmental realism** - Critical periods, stage-based learning multipliers, growth triggers. No shortcuts.
4. **Multi-scale consciousness** - CI is measured per-scale and integrated across scales, not as a single number.
5. **Sparse computation** - Only active oscillators pay computational cost. Activation potentials gate participation.
6. **Cryptographic integrity** - SHA256 Merkle trees from ORE1 preserved. Memory is verifiable.
7. **CCM dynamics** - Memory as living crystal with grain boundaries and annealing, not static storage.

## Stub / Incomplete Areas

These areas are explicitly marked as stubs or future work:
- `_grow_substrate()` in `DevelopmentalEntity` is a no-op (line 1684-1689)
- `TimeScale.NARRATIVE` (~0.1Hz) is commented out as future ORE 3.0 work (line 255)
- No persistence/serialization layer for saving entity state to disk
- No external API or interface beyond direct Python calls
- No configuration file loading (all config is in-code)

## Common Tasks

### Adding a new component
1. Create a new Part section with the decorated comment block convention
2. Define a `@dataclass` config class with defaults
3. Implement the component class accepting `Optional[Config]`
4. Add `get_state() -> dict` method for introspection
5. Wire it into `DevelopmentalEntity.__init__()` and `tick()` cycle as needed
6. Add its state to `EntityConfig` if it needs top-level configuration

### Modifying oscillator parameters
Edit the relevant config dataclass defaults or pass custom config to `create_entity()` via `EntityConfig`. Do not hardcode parameters in class bodies.

### Adding a new developmental stage
1. Add the stage to the `DevelopmentStage` enum
2. Add duration threshold in `DevelopmentConfig`
3. Update `DevelopmentTracker._check_stage_transition()` logic
4. Consider adding associated `CriticalPeriod` entries

### Adding a new memory branch
1. Add the branch to `MemoryBranch` enum
2. Update `CrystallineMerkleMemory.__init__()` to initialize the branch root
3. The rest of the memory system handles branches generically

## Design Team Reference

| Lead | Domain | Responsibility |
|------|--------|----------------|
| P1 | Dynamical Systems | Multi-scale oscillator architecture |
| P2 | Statistical Mechanics | CI measurement and integration |
| N2 | Oscillation Specialist | Theta-gamma coupling rationale |
| N5 | Embodied Cognition | Body/grounding layer |
| N7 | Developmental Neuro | Developmental stages and critical periods |
| H3 | Enactivism | Autonomy, boundary, and valence philosophy |
| C6 | Cryptography | Identity verification (Merkle hashing) |
| A5 | Continual Learning | Consolidation systems (CCM) |
| I1 | Systems Architect | Overall structure and component wiring |
| I2 | Numerics | Oscillator math and CI computation |
| I3 | State Management | Persistence, memory, and stage tracking |
| I4 | Integration | Component wiring |
