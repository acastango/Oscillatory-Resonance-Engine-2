# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: CRYSTALLINE MERKLE MEMORY / CCM
# Design: C6 (Cryptography) + A5 (Continual Learning)
# Implementation: I3 (State Management)
# ═══════════════════════════════════════════════════════════════════════════════

"""
C6: "The merkle structure is cryptographically sound - hash chains prove
continuity. Keep it. But memory isn't just storage, it's living structure."

A5: "Real brains don't commit everything immediately. Updates queue for
consolidation during sleep. This prevents catastrophic forgetting."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MemoryBranch(Enum):
    """Memory branch types."""
    SELF = "self"                # Core identity claims
    RELATIONS = "relations"      # Connections to others
    INSIGHTS = "insights"        # Learned patterns
    EXPERIENCES = "experiences"  # Episodic memories


@dataclass
class MemoryNode:
    """
    A node in the Merkle memory tree.

    Extended from ORE1 with CCM tension tracking.
    """
    # Core
    id: str
    branch: MemoryBranch
    content: Dict[str, Any]
    created_at: str  # ISO format

    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Merkle hash
    hash: str = ""

    # Substrate anchoring
    substrate_anchor: Optional[Dict] = None
    coherence_at_creation: float = 0.0

    # CCM extension: tensions with other memories
    tensions: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConsolidationQueue:
    """Queued updates for sleep consolidation."""
    pending_nodes: List[MemoryNode] = field(default_factory=list)
    pending_tensions: List[Tuple[str, str, float]] = field(default_factory=list)

    def queue_node(self, node: MemoryNode) -> None:
        self.pending_nodes.append(node)

    def queue_tension(self, node_a: str, node_b: str, tension: float) -> None:
        self.pending_tensions.append((node_a, node_b, tension))

    def is_empty(self) -> bool:
        return len(self.pending_nodes) == 0 and len(self.pending_tensions) == 0

    def clear(self) -> None:
        self.pending_nodes.clear()
        self.pending_tensions.clear()


class CrystallineMerkleMemory:
    """
    Merkle tree memory with CCM (Crystalline Constraint Memory) dynamics.

    Memory as crystal growth:
    - New memory = stress on existing crystal
    - Compatible -> smooth incorporation
    - Incompatible -> grain boundary (tension)
    - Consolidation = annealing (resolve tensions probabilistically)

    The merkle tree provides cryptographic verification (identity continuity).
    The CCM dynamics provide living memory (not dead storage).
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, MemoryNode] = {}
        self.branch_roots: Dict[MemoryBranch, str] = {}
        self.root_hash: str = ""
        self.total_nodes: int = 0
        self.depth: int = 0

        # CCM
        self.consolidation_queue = ConsolidationQueue()
        self.grain_boundaries: List[Tuple[str, str, float]] = []

        # Initialize branch roots
        self._init_branches()

    def _init_branches(self) -> None:
        """Create root node for each branch (not counted in total_nodes)."""
        for branch in MemoryBranch:
            root_id = f"root_{branch.value}"
            root_node = MemoryNode(
                id=root_id,
                branch=branch,
                content={"type": "branch_root", "branch": branch.value},
                created_at=datetime.now().isoformat(),
            )
            root_node.hash = self._compute_node_hash(root_node, [])
            self.nodes[root_id] = root_node
            self.branch_roots[branch] = root_id

        self._update_root_hash()
        self._update_depth()

    # ── Public Methods ──────────────────────────────────────────────────────

    def add(
        self,
        branch: MemoryBranch,
        content: Dict[str, Any],
        substrate_state: Optional[Dict] = None,
        immediate: bool = True,
    ) -> MemoryNode:
        """
        Add a memory node.

        If immediate=True, commit to tree now.
        If immediate=False, queue for consolidation.
        """
        node_id = self._generate_id()
        parent_id = self.branch_roots[branch]

        node = MemoryNode(
            id=node_id,
            branch=branch,
            content=content,
            created_at=datetime.now().isoformat(),
            parent_id=parent_id,
        )

        # Substrate anchoring
        if substrate_state:
            node.substrate_anchor = {
                "coherence": substrate_state.get("global_coherence", 0),
                "cross_scale": substrate_state.get("cross_scale_coherence", 0),
                "time": substrate_state.get("time", 0),
            }
            node.coherence_at_creation = substrate_state.get("global_coherence", 0)

        if immediate:
            self._commit_node(node)
        else:
            self.consolidation_queue.queue_node(node)

        return node

    def verify(self) -> Tuple[bool, str]:
        """
        Verify hash integrity of entire tree.

        Returns (True, "All nodes verified") if valid,
        or (False, error_message) if invalid.
        """
        for node_id, node in self.nodes.items():
            children_hashes = [self.nodes[cid].hash for cid in node.children_ids]
            expected_hash = self._compute_node_hash(node, children_hashes)

            if node.hash != expected_hash:
                return False, f"Hash mismatch at {node_id}"

        return True, "All nodes verified"

    def consolidate(self, temperature: float = 1.0) -> dict:
        """
        Sleep consolidation: commit queued nodes and resolve tensions.

        Higher temperature = more likely to resolve tensions (annealing).
        """
        if self.consolidation_queue.is_empty() and not self.grain_boundaries:
            return {"consolidated": 0, "tensions_resolved": 0, "remaining_tensions": 0}

        consolidated = 0
        tensions_resolved = 0

        # Commit pending nodes
        for node in self.consolidation_queue.pending_nodes:
            self._commit_node(node)
            consolidated += 1

        # Process pending tensions
        for node_a, node_b, tension in self.consolidation_queue.pending_tensions:
            if node_a in self.nodes and node_b in self.nodes:
                self.nodes[node_a].tensions[node_b] = tension
                self.nodes[node_b].tensions[node_a] = tension
                self.grain_boundaries.append((node_a, node_b, tension))

        # Annealing: probabilistically resolve grain boundaries
        resolved_indices = []

        for i, (node_a, node_b, tension) in enumerate(self.grain_boundaries):
            # Skip if nodes were deleted
            if node_a not in self.nodes or node_b not in self.nodes:
                resolved_indices.append(i)
                continue

            # Resolution probability
            # At temp=1, tension=0.5: prob=0.25
            # At temp=2, tension=0.5: prob=0.5
            resolve_prob = temperature * (1 - tension) / 2

            if np.random.random() < resolve_prob:
                self.nodes[node_a].tensions.pop(node_b, None)
                self.nodes[node_b].tensions.pop(node_a, None)
                resolved_indices.append(i)
                tensions_resolved += 1

        # Remove resolved boundaries (reverse to preserve indices)
        for i in sorted(resolved_indices, reverse=True):
            self.grain_boundaries.pop(i)

        # Clear queue
        self.consolidation_queue.clear()

        return {
            "consolidated": consolidated,
            "tensions_resolved": tensions_resolved,
            "remaining_tensions": len(self.grain_boundaries),
        }

    def get_fractal_dimension(self) -> float:
        """
        Compute fractal dimension D = log(N) / log(depth).

        Used in CI calculation.
        """
        n = self.total_nodes
        d = max(self.depth, 1)

        if n <= 1 or d <= 1:
            return 1.0

        return float(np.log(n) / np.log(d))

    @property
    def fractal_dimension(self) -> float:
        """Property alias for get_fractal_dimension."""
        return self.get_fractal_dimension()

    def query(
        self,
        branch: Optional[MemoryBranch] = None,
        content_filter: Optional[Dict] = None,
    ) -> List[MemoryNode]:
        """
        Query memories by branch and/or content.

        content_filter: dict of key-value pairs that must match in content.
        """
        results = []

        for node in self.nodes.values():
            # Skip roots
            if node.content.get("type") == "branch_root":
                continue

            # Branch filter
            if branch and node.branch != branch:
                continue

            # Content filter
            if content_filter:
                match = all(
                    node.content.get(k) == v for k, v in content_filter.items()
                )
                if not match:
                    continue

            results.append(node)

        return results

    def get_state(self) -> dict:
        """Serialize current state."""
        return {
            "root_hash": self.root_hash,
            "total_nodes": self.total_nodes,
            "depth": self.depth,
            "fractal_dimension": self.fractal_dimension,
            "grain_boundaries": len(self.grain_boundaries),
            "pending_consolidation": len(self.consolidation_queue.pending_nodes),
            "verified": self.verify()[0],
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _commit_node(self, node: MemoryNode) -> None:
        """Add node to tree, detect tensions, update hashes."""
        # Add to nodes dict
        self.nodes[node.id] = node

        # Link to parent
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children_ids.append(node.id)

        # Detect tensions with existing nodes (CCM)
        self._detect_tensions(node)

        # Update merkle hashes up to root
        self._update_hashes_to_root(node.id)

        # Update depth
        self._update_depth()

        # Increment counter (not counting branch roots)
        if node.content.get("type") != "branch_root":
            self.total_nodes += 1

    def _detect_tensions(self, new_node: MemoryNode) -> None:
        """
        Detect tensions between new node and existing memories.

        Tension occurs when memories have medium word overlap (Jaccard 0.2-0.7).
        """
        new_content_str = json.dumps(new_node.content, sort_keys=True).lower()
        new_words = set(new_content_str.split())

        for node_id, existing in self.nodes.items():
            # Skip roots and self
            if existing.content.get("type") == "branch_root":
                continue
            if node_id == new_node.id:
                continue

            # Only check same branch
            if existing.branch != new_node.branch:
                continue

            existing_content_str = json.dumps(existing.content, sort_keys=True).lower()
            existing_words = set(existing_content_str.split())

            if not new_words or not existing_words:
                continue

            overlap = len(new_words & existing_words)
            union = len(new_words | existing_words)
            jaccard = overlap / union if union > 0 else 0

            # Tension = medium overlap (potentially contradictory)
            if 0.2 < jaccard < 0.7:
                tension = 1.0 - abs(jaccard - 0.5) * 2  # Peak at 0.5

                if tension > 0.2:
                    new_node.tensions[node_id] = tension
                    existing.tensions[new_node.id] = tension
                    self.grain_boundaries.append((new_node.id, node_id, tension))

    def _generate_id(self) -> str:
        """Generate unique node ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        counter = len(self.nodes)
        return f"node_{timestamp}_{counter}"

    def _compute_node_hash(self, node: MemoryNode, children_hashes: List[str]) -> str:
        """Compute merkle hash for a node."""
        data = {
            "id": node.id,
            "content": node.content,
            "children": sorted(children_hashes),
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _update_hashes_to_root(self, node_id: str) -> None:
        """Propagate hash updates from node to root."""
        node = self.nodes[node_id]

        # Compute this node's hash from children
        children_hashes = [self.nodes[cid].hash for cid in node.children_ids]
        node.hash = self._compute_node_hash(node, children_hashes)

        # Propagate to parent
        if node.parent_id and node.parent_id in self.nodes:
            self._update_hashes_to_root(node.parent_id)

        # If this is a branch root, update global root
        if node.id in self.branch_roots.values():
            self._update_root_hash()

    def _update_root_hash(self) -> None:
        """Compute global root hash from branch roots."""
        branch_hashes = sorted(
            [self.nodes[rid].hash for rid in self.branch_roots.values()]
        )
        data = {"branches": branch_hashes}
        self.root_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def _update_depth(self) -> None:
        """Recompute tree depth."""
        def node_depth(nid: str) -> int:
            node = self.nodes.get(nid)
            if not node or not node.children_ids:
                return 1
            return 1 + max(node_depth(cid) for cid in node.children_ids)

        if self.branch_roots:
            self.depth = max(node_depth(rid) for rid in self.branch_roots.values())
        else:
            self.depth = 0
