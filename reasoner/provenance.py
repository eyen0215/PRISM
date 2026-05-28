"""
Provenance tracking for derived physical quantities.

The three provenance rules:

    observable node  →  assumption_ids = frozenset()
    assumption node  →  assumption_ids = frozenset({node.id})
    derived node     →  assumption_ids = union of all premise assumption_ids

When a derivation edge fires, the conclusion inherits the union of its
premises' assumption sets. This means flagging a single assumption node
automatically propagates to every derived result that depends on it —
with no extra bookkeeping in the derivation rules themselves.

Public API
----------
ProvenanceRecord
    Immutable record binding a node ID to its set of ancestor assumptions.
    Classmethods for_observable / for_assumption / from_premises construct
    the three kinds. Query methods: depends_on(), is_flagged_by().

compute_provenance(graph) -> Dict[str, ProvenanceRecord]
    Walk the axiom graph in topological order and return a ProvenanceRecord
    for every node.  O(V + E).

flagged_nodes(provenance_map, flagged_assumptions) -> Dict[str, ProvenanceRecord]
    Filter a provenance map down to only those nodes whose provenance
    overlaps with the given set of flagged assumption IDs.

Fits into the system: reasoner/forward_chain.py calls compute_provenance()
so that every derived value carries its assumption footprint; experiments/
pilot.py calls flagged_nodes() to report which derived results are suspect
when validity predicates fire on the held-out regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Dict, FrozenSet, Iterable

from axiom_graph.graph import AxiomGraph


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable record of which assumption nodes a given node depends on.

    Parameters
    ----------
    node_id        : the ID of the node this record describes
    assumption_ids : frozenset of assumption node IDs in this node's ancestry
    """

    node_id: str
    assumption_ids: FrozenSet[str]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_observable(cls, node_id: str) -> ProvenanceRecord:
        """Observable nodes carry no assumption dependencies."""
        return cls(node_id=node_id, assumption_ids=frozenset())

    @classmethod
    def for_assumption(cls, node_id: str) -> ProvenanceRecord:
        """An assumption node is the sole member of its own provenance set."""
        return cls(node_id=node_id, assumption_ids=frozenset({node_id}))

    @classmethod
    def from_premises(
        cls,
        node_id: str,
        premise_records: Iterable[ProvenanceRecord],
    ) -> ProvenanceRecord:
        """Derived node: union of all premises' assumption_ids.

        If premise_records is empty the result has empty assumption_ids,
        equivalent to an observable — this is a valid degenerate case.
        """
        merged: FrozenSet[str] = frozenset()
        for rec in premise_records:
            merged = merged | rec.assumption_ids
        return cls(node_id=node_id, assumption_ids=merged)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def depends_on(self, assumption_id: str) -> bool:
        """True if `assumption_id` is in this node's ancestor assumption set."""
        return assumption_id in self.assumption_ids

    def is_flagged_by(self, flagged: AbstractSet[str]) -> bool:
        """True if any element of `flagged` appears in this node's assumption_ids.

        This is the propagation mechanism: a node is considered suspect
        whenever at least one assumption it depends on has been flagged as
        potentially violated.
        """
        return bool(self.assumption_ids & flagged)


# ---------------------------------------------------------------------------
# Forward chaining over the axiom graph
# ---------------------------------------------------------------------------

def compute_provenance(graph: AxiomGraph) -> Dict[str, ProvenanceRecord]:
    """Walk the axiom graph in topological order, computing ProvenanceRecords.

    Each node's record is constructed exactly once, from already-completed
    predecessor records, so the invariant always holds: a node's assumption_ids
    equal the union of its predecessors' assumption_ids.

    Returns
    -------
    Dict mapping every node ID in the graph to its ProvenanceRecord.
    """
    records: Dict[str, ProvenanceRecord] = {}

    for node_id in graph.topological_order():
        node = graph.nodes[node_id]

        if node.kind == "observable":
            records[node_id] = ProvenanceRecord.for_observable(node_id)

        elif node.kind == "assumption":
            records[node_id] = ProvenanceRecord.for_assumption(node_id)

        else:  # derived
            premise_records = [
                records[pred.id] for pred in graph.predecessors(node_id)
            ]
            records[node_id] = ProvenanceRecord.from_premises(node_id, premise_records)

    return records


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def flagged_nodes(
    provenance_map: Dict[str, ProvenanceRecord],
    flagged_assumptions: AbstractSet[str],
) -> Dict[str, ProvenanceRecord]:
    """Return the subset of provenance_map whose nodes depend on a flagged assumption.

    Parameters
    ----------
    provenance_map      : output of compute_provenance()
    flagged_assumptions : set of assumption node IDs that have been flagged
                          (e.g. by validity predicates firing below threshold)

    Returns
    -------
    Dict of {node_id: ProvenanceRecord} for every node whose provenance
    overlaps with flagged_assumptions. Nodes with empty assumption_ids
    (pure observables) are never included.
    """
    if not flagged_assumptions:
        return {}
    return {
        nid: rec
        for nid, rec in provenance_map.items()
        if rec.is_flagged_by(flagged_assumptions)
    }
