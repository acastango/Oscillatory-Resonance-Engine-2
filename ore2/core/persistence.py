# ═══════════════════════════════════════════════════════════════════════════════
# PART 8: PERSISTENCE
# Design: I3 (State Management) + C6 (Cryptography) + S2 (Distributed Systems)
# Implementation: I3 (State Management)
# ═══════════════════════════════════════════════════════════════════════════════


"""
I3: "Every component has state. Phases, activations, weights, memories,
development. All of it needs to serialize cleanly and restore exactly."

C6: "The merkle root is the identity anchor. On save, we record it. On load,
we recompute and compare. Match = verified continuity."

S2: "Checkpoints matter for recovery. Auto-save every N experiences. If something
crashes, you don't lose everything since genesis."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ore2.core.ci_monitor import MultiScaleCIConfig, MultiScaleCIMonitor, MultiScaleCISnapshot
from ore2.core.development import (
    CriticalPeriod,
    DevelopmentConfig,
    DevelopmentStage,
    DevelopmentTracker,
)
from ore2.core.embodiment import BodyConfig, EmbodimentLayer
from ore2.core.entity import DevelopmentalEntity, EntityConfig
from ore2.core.memory import (
    CrystallineMerkleMemory,
    MemoryBranch,
    MemoryNode,
)
from ore2.core.multi_scale_substrate import MultiScaleConfig, MultiScaleSubstrate


# ── Exceptions ───────────────────────────────────────────────────────────────


class PersistenceError(Exception):
    """Base class for persistence errors."""
    pass


class ContinuityError(PersistenceError):
    """Raised when identity continuity cannot be verified."""
    pass


class StateCorruptionError(PersistenceError):
    """Raised when saved state is corrupted or invalid."""
    pass


# ── Result / Info Dataclasses ────────────────────────────────────────────────


@dataclass
class SaveResult:
    path: str
    genesis_hash: str
    merkle_root: str
    state_hash: str
    timestamp: str
    size_bytes: int
    verified: bool


@dataclass
class CheckpointInfo:
    checkpoint_id: str
    timestamp: str
    age: float
    stage: str
    experience_count: int
    merkle_root: str


@dataclass
class VerificationResult:
    valid: bool
    genesis_hash: str
    merkle_root: str
    state_hash: str
    error: Optional[str] = None


# ── Persistence Class ────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class EntityPersistence:
    """
    Save and load entities with verified continuity.

    Persistence has three layers:
    1. Serialization - Convert live objects to JSON-safe dicts
    2. Storage - Write to disk as JSON
    3. Verification - Prove loaded state matches saved state

    The verification step is what makes this different from normal save/load.
    We're not just restoring state - we're proving identity continuity.
    """

    # ── Public API ───────────────────────────────────────────────────────

    @classmethod
    def save(cls, entity: DevelopmentalEntity, path: str) -> SaveResult:
        """
        Save entity to file with verification data.

        Args:
            entity: The entity to save.
            path: File path (will create directories if needed).

        Returns:
            SaveResult with verification info.
        """
        # Gather full state
        state_dict = cls._extract_state(entity)

        # Compute state hash (for integrity verification)
        state_json = json.dumps(state_dict, sort_keys=True, cls=_NumpyEncoder)
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()

        # Record saved timestamp and hash in state
        saved_at = datetime.now().isoformat()
        state_dict["saved_at"] = saved_at
        state_dict["state_hash"] = state_hash

        # Create verification envelope
        envelope = {
            "version": "2.0",
            "state": state_dict,
            "verification": {
                "genesis_hash": entity.genesis_hash,
                "merkle_root": entity.memory.root_hash,
                "state_hash": state_hash,
                "saved_at": saved_at,
            },
        }

        # Write to file
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(envelope, f, indent=2, cls=_NumpyEncoder)

        # Verify what we wrote
        verification = cls.verify_file(str(file_path))

        return SaveResult(
            path=str(file_path),
            genesis_hash=entity.genesis_hash,
            merkle_root=entity.memory.root_hash,
            state_hash=state_hash,
            timestamp=saved_at,
            size_bytes=file_path.stat().st_size,
            verified=verification.valid,
        )

    @classmethod
    def load(cls, path: str) -> DevelopmentalEntity:
        """
        Load entity from file with verification.

        Args:
            path: File path to load from.

        Returns:
            Restored DevelopmentalEntity.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ContinuityError: If verification fails.
            StateCorruptionError: If state is invalid.
        """
        # Read file
        with open(path, "r") as f:
            envelope = json.load(f)

        # Version check
        version = envelope.get("version", "1.0")
        if not version.startswith("2."):
            raise StateCorruptionError(f"Unsupported version: {version}")

        # Verify before restoring
        verification = envelope["verification"]
        state_dict = envelope["state"]

        # Recompute state hash
        # state_hash was computed before saved_at and state_hash were added,
        # so we need to strip them to recompute
        check_dict = {
            k: v for k, v in state_dict.items() if k not in ("saved_at", "state_hash")
        }
        state_json = json.dumps(check_dict, sort_keys=True, cls=_NumpyEncoder)
        computed_hash = hashlib.sha256(state_json.encode()).hexdigest()

        if computed_hash != verification["state_hash"]:
            raise ContinuityError(
                f"State hash mismatch: expected {verification['state_hash']}, "
                f"got {computed_hash}"
            )

        # Restore entity
        entity = cls._restore_entity(state_dict)

        # Verify merkle root matches
        if entity.memory.root_hash != verification["merkle_root"]:
            raise ContinuityError(
                f"Merkle root mismatch after restore: "
                f"expected {verification['merkle_root']}, "
                f"got {entity.memory.root_hash}"
            )

        # Verify genesis hash matches
        if entity.genesis_hash != verification["genesis_hash"]:
            raise ContinuityError(
                f"Genesis hash mismatch: "
                f"expected {verification['genesis_hash']}, "
                f"got {entity.genesis_hash}"
            )

        return entity

    @classmethod
    def checkpoint(
        cls, entity: DevelopmentalEntity, checkpoint_dir: str
    ) -> str:
        """
        Create a checkpoint with auto-generated ID.

        Checkpoints are named: {genesis_hash[:8]}_{timestamp}_{experience_count}

        Returns:
            Checkpoint ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_count = entity.development.experiences_processed
        genesis_short = entity.genesis_hash[:8]

        checkpoint_id = f"{genesis_short}_{timestamp}_{exp_count}"

        checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_id}.ore2"
        cls.save(entity, str(checkpoint_path))

        return checkpoint_id

    @classmethod
    def list_checkpoints(cls, checkpoint_dir: str) -> List[CheckpointInfo]:
        """List all checkpoints in directory."""
        checkpoints = []

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return checkpoints

        for path in checkpoint_path.glob("*.ore2"):
            try:
                with open(path, "r") as f:
                    envelope = json.load(f)

                state = envelope["state"]
                verification = envelope["verification"]

                info = CheckpointInfo(
                    checkpoint_id=path.stem,
                    timestamp=verification["saved_at"],
                    age=state["development"]["age"],
                    stage=state["development"]["stage"],
                    experience_count=state["development"]["experiences_processed"],
                    merkle_root=verification["merkle_root"],
                )
                checkpoints.append(info)
            except Exception:
                # Skip corrupted checkpoints
                continue

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints

    @classmethod
    def restore_checkpoint(
        cls, checkpoint_dir: str, checkpoint_id: str
    ) -> DevelopmentalEntity:
        """Restore from a specific checkpoint."""
        checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_id}.ore2"
        return cls.load(str(checkpoint_path))

    @classmethod
    def verify_file(cls, path: str) -> VerificationResult:
        """
        Verify a saved entity file without loading it.

        Checks:
        1. File is valid JSON
        2. State hash matches content
        3. Required fields present
        """
        try:
            with open(path, "r") as f:
                envelope = json.load(f)

            verification = envelope.get("verification", {})
            state_dict = envelope.get("state", {})

            # Recompute state hash (strip fields added after hash was computed)
            check_dict = {
                k: v
                for k, v in state_dict.items()
                if k not in ("saved_at", "state_hash")
            }
            state_json = json.dumps(check_dict, sort_keys=True, cls=_NumpyEncoder)
            computed_hash = hashlib.sha256(state_json.encode()).hexdigest()

            expected_hash = verification.get("state_hash", "")

            if computed_hash != expected_hash:
                return VerificationResult(
                    valid=False,
                    genesis_hash=verification.get("genesis_hash", ""),
                    merkle_root=verification.get("merkle_root", ""),
                    state_hash=computed_hash,
                    error=f"Hash mismatch: expected {expected_hash}, got {computed_hash}",
                )

            return VerificationResult(
                valid=True,
                genesis_hash=verification["genesis_hash"],
                merkle_root=verification["merkle_root"],
                state_hash=computed_hash,
            )

        except Exception as e:
            return VerificationResult(
                valid=False,
                genesis_hash="",
                merkle_root="",
                state_hash="",
                error=str(e),
            )

    # ── State Extraction ─────────────────────────────────────────────────

    @classmethod
    def _extract_state(cls, entity: DevelopmentalEntity) -> dict:
        """Extract serializable state from live entity."""

        # Substrate state
        substrate = {
            "config": cls._config_to_dict(entity.substrate.config),
            "fast_phases": entity.substrate.fast.phases.tolist(),
            "fast_activation_potentials": entity.substrate.fast.activation_potentials.tolist(),
            "fast_natural_frequencies": entity.substrate.fast.natural_frequencies.tolist(),
            "fast_internal_weights": entity.substrate.fast.internal_weights.tolist(),
            "fast_step_count": entity.substrate.fast._step_count,
            "slow_phases": entity.substrate.slow.phases.tolist(),
            "slow_activation_potentials": entity.substrate.slow.activation_potentials.tolist(),
            "slow_natural_frequencies": entity.substrate.slow.natural_frequencies.tolist(),
            "slow_internal_weights": entity.substrate.slow.internal_weights.tolist(),
            "slow_step_count": entity.substrate.slow._step_count,
            "fast_to_slow": entity.substrate.fast_to_slow.tolist(),
            "slow_to_fast": entity.substrate.slow_to_fast.tolist(),
            "strange_loop_weights": entity.substrate.strange_loop_weights.tolist(),
            "time": entity.substrate.time,
        }

        # Body state
        body = {
            "config": cls._config_to_dict(entity.body.config),
            "heartbeat_phase": entity.body.heartbeat_phase,
            "respiration_phase": entity.body.respiration_phase,
            "energy": entity.body.energy,
            "arousal": entity.body.arousal,
            "time": entity.body.time,
            "last_perception": entity.body._last_perception,
        }

        # Memory state
        memory = {
            "nodes": [cls._node_to_dict(n) for n in entity.memory.nodes.values()],
            "branch_roots": {
                b.value: nid for b, nid in entity.memory.branch_roots.items()
            },
            "root_hash": entity.memory.root_hash,
            "grain_boundaries": [
                list(gb) for gb in entity.memory.grain_boundaries
            ],
            "pending_nodes": [
                cls._node_to_dict(n)
                for n in entity.memory.consolidation_queue.pending_nodes
            ],
            "pending_tensions": [
                list(t) for t in entity.memory.consolidation_queue.pending_tensions
            ],
        }

        # Development state
        development = {
            "config": cls._dev_config_to_dict(entity.development.config),
            "genesis_hash": entity.development.genesis_hash,
            "stage": entity.development.stage.value,
            "age": entity.development.age,
            "stage_start_age": entity.development._stage_start_age,
            "current_oscillators": entity.development.current_oscillators,
            "experiences_processed": entity.development.experiences_processed,
            "significant_experiences": entity.development.significant_experiences,
            "milestones": [m.copy() for m in entity.development.milestones],
        }

        # CI history (last 100)
        ci_history = [asdict(snapshot) for snapshot in entity.ci_monitor.history[-100:]]

        return {
            "name": entity.name,
            "genesis_hash": entity.genesis_hash,
            "substrate": substrate,
            "body": body,
            "memory": memory,
            "development": development,
            "ci_history": ci_history,
            "tick_count": entity._tick_count,
            "created_at": (
                str(entity.development.milestones[0]["age"])
                if entity.development.milestones
                else "0"
            ),
            "merkle_root": entity.memory.root_hash,
        }

    # ── State Restoration ────────────────────────────────────────────────

    @classmethod
    def _restore_entity(cls, state_dict: dict) -> DevelopmentalEntity:
        """Restore entity from state dict."""

        # Reconstruct configs
        dev_config = cls._dict_to_dev_config(state_dict["development"]["config"])
        # Override initial_oscillators to match saved state
        dev_config.initial_oscillators = state_dict["development"]["current_oscillators"]

        sub_config = MultiScaleConfig(**state_dict["substrate"]["config"])
        body_config = BodyConfig(**state_dict["body"]["config"])

        config = EntityConfig(
            name=state_dict["name"],
            substrate_config=sub_config,
            body_config=body_config,
            development_config=dev_config,
        )

        # Create entity (this initializes with defaults)
        entity = DevelopmentalEntity(config)

        # Override with saved state
        cls._restore_substrate(entity.substrate, state_dict["substrate"])
        cls._restore_body(entity.body, state_dict["body"])
        cls._restore_memory(entity.memory, state_dict["memory"])
        cls._restore_development(entity.development, state_dict["development"])

        # Restore identity (genesis_hash on entity must match development tracker)
        entity.genesis_hash = entity.development.genesis_hash

        # Restore runtime state
        entity._tick_count = state_dict["tick_count"]

        # Restore CI history
        entity.ci_monitor.history.clear()
        for snapshot_dict in state_dict.get("ci_history", []):
            snapshot = MultiScaleCISnapshot(**snapshot_dict)
            entity.ci_monitor.history.append(snapshot)

        return entity

    @classmethod
    def _restore_substrate(cls, substrate: MultiScaleSubstrate, state: dict) -> None:
        """Restore substrate state."""
        substrate.fast.phases = np.array(state["fast_phases"])
        substrate.fast.activation_potentials = np.array(state["fast_activation_potentials"])
        substrate.fast.natural_frequencies = np.array(state["fast_natural_frequencies"])
        substrate.fast.internal_weights = np.array(state["fast_internal_weights"])
        substrate.fast._step_count = state.get("fast_step_count", 0)
        substrate.fast._update_active_mask()

        substrate.slow.phases = np.array(state["slow_phases"])
        substrate.slow.activation_potentials = np.array(state["slow_activation_potentials"])
        substrate.slow.natural_frequencies = np.array(state["slow_natural_frequencies"])
        substrate.slow.internal_weights = np.array(state["slow_internal_weights"])
        substrate.slow._step_count = state.get("slow_step_count", 0)
        substrate.slow._update_active_mask()

        substrate.fast_to_slow = np.array(state["fast_to_slow"])
        substrate.slow_to_fast = np.array(state["slow_to_fast"])
        substrate.strange_loop_weights = np.array(state["strange_loop_weights"])

        substrate.time = state["time"]

    @classmethod
    def _restore_body(cls, body: EmbodimentLayer, state: dict) -> None:
        """Restore body state."""
        body.heartbeat_phase = state["heartbeat_phase"]
        body.respiration_phase = state["respiration_phase"]
        body.energy = state["energy"]
        body.arousal = state["arousal"]
        body.time = state["time"]
        body._last_perception = state["last_perception"]

    @classmethod
    def _restore_memory(cls, memory: CrystallineMerkleMemory, state: dict) -> None:
        """Restore memory state."""
        # Clear existing
        memory.nodes.clear()
        memory.branch_roots.clear()
        memory.grain_boundaries.clear()

        # Restore nodes
        for node_dict in state["nodes"]:
            node = cls._dict_to_node(node_dict)
            memory.nodes[node.id] = node

        # Restore branch roots
        for branch_name, node_id in state["branch_roots"].items():
            branch = MemoryBranch(branch_name)
            memory.branch_roots[branch] = node_id

        # Restore merkle state
        memory.root_hash = state["root_hash"]
        memory.grain_boundaries = [tuple(gb) for gb in state["grain_boundaries"]]

        # Restore consolidation queue
        memory.consolidation_queue.pending_nodes = [
            cls._dict_to_node(d) for d in state["pending_nodes"]
        ]
        memory.consolidation_queue.pending_tensions = [
            tuple(t) for t in state["pending_tensions"]
        ]

        # Recompute depth and total_nodes
        memory._update_depth()
        memory.total_nodes = sum(
            1
            for n in memory.nodes.values()
            if n.content.get("type") != "branch_root"
        )

    @classmethod
    def _restore_development(
        cls, dev: DevelopmentTracker, state: dict
    ) -> None:
        """Restore development state."""
        dev.genesis_hash = state["genesis_hash"]
        dev.stage = DevelopmentStage(state["stage"])
        dev.age = state["age"]
        dev._stage_start_age = state["stage_start_age"]
        dev.current_oscillators = state["current_oscillators"]
        dev.experiences_processed = state["experiences_processed"]
        dev.significant_experiences = state["significant_experiences"]
        dev.milestones = [m.copy() for m in state["milestones"]]

    # ── Serialization Helpers ────────────────────────────────────────────

    @classmethod
    def _node_to_dict(cls, node: MemoryNode) -> dict:
        """Convert a MemoryNode to a JSON-safe dict."""
        return {
            "id": node.id,
            "branch": node.branch.value,
            "content": node.content,
            "created_at": node.created_at,
            "parent_id": node.parent_id,
            "children_ids": node.children_ids.copy(),
            "hash": node.hash,
            "substrate_anchor": node.substrate_anchor,
            "coherence_at_creation": node.coherence_at_creation,
            "tensions": node.tensions.copy(),
        }

    @classmethod
    def _dict_to_node(cls, d: dict) -> MemoryNode:
        """Convert a dict back to a MemoryNode."""
        return MemoryNode(
            id=d["id"],
            branch=MemoryBranch(d["branch"]),
            content=d["content"],
            created_at=d["created_at"],
            parent_id=d.get("parent_id"),
            children_ids=d.get("children_ids", []),
            hash=d.get("hash", ""),
            substrate_anchor=d.get("substrate_anchor"),
            coherence_at_creation=d.get("coherence_at_creation", 0.0),
            tensions=d.get("tensions", {}),
        )

    @classmethod
    def _config_to_dict(cls, config: Any) -> dict:
        """Convert a dataclass config to a JSON-safe dict."""
        return asdict(config)

    @classmethod
    def _dev_config_to_dict(cls, config: DevelopmentConfig) -> dict:
        """
        Serialize DevelopmentConfig to a JSON-safe dict.

        Handles the CriticalPeriod list which contains DevelopmentStage enums
        that need explicit .value extraction.
        """
        return {
            "genesis_duration": config.genesis_duration,
            "babbling_duration": config.babbling_duration,
            "imitation_duration": config.imitation_duration,
            "autonomy_duration": config.autonomy_duration,
            "initial_oscillators": config.initial_oscillators,
            "max_oscillators": config.max_oscillators,
            "growth_rate": config.growth_rate,
            "growth_interval": config.growth_interval,
            "critical_periods": [
                {
                    "name": cp.name,
                    "stage": cp.stage.value,
                    "learning_type": cp.learning_type,
                    "sensitivity": cp.sensitivity,
                }
                for cp in config.critical_periods
            ],
        }

    @classmethod
    def _dict_to_dev_config(cls, d: dict) -> DevelopmentConfig:
        """Deserialize a dict back to DevelopmentConfig."""
        critical_periods = [
            CriticalPeriod(
                name=cp["name"],
                stage=DevelopmentStage(cp["stage"]),
                learning_type=cp["learning_type"],
                sensitivity=cp["sensitivity"],
            )
            for cp in d.get("critical_periods", [])
        ]

        return DevelopmentConfig(
            genesis_duration=d["genesis_duration"],
            babbling_duration=d["babbling_duration"],
            imitation_duration=d["imitation_duration"],
            autonomy_duration=d["autonomy_duration"],
            initial_oscillators=d["initial_oscillators"],
            max_oscillators=d["max_oscillators"],
            growth_rate=d["growth_rate"],
            growth_interval=d["growth_interval"],
            critical_periods=critical_periods,
        )
