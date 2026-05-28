"""
Forward chaining engine over the axiom graph.

Queries each assumption node's validity predicate, then propagates flags
to every derived node whose provenance overlaps with a fired assumption.

The key insight: flag propagation is a pure set-intersection check on the
static provenance map — no re-traversal of the graph is required per state.
A batch of N states is handled by OR-ing per-assumption flag arrays.

Public API
----------
ForwardChainResult
    Dataclass holding, for a batch of N states:
        assumption_scores  — {assumption_id: float32 array (N,)}  scores in (0,1)
        assumption_flagged — {assumption_id: bool array (N,)}     score < threshold
        node_flagged       — {node_id: bool array (N,)}           any ancestor fired

run_forward_chain(graph, df, threshold) -> ForwardChainResult
    Evaluate all predicates and propagate flags for an entire DataFrame.

Fits into the system: called from experiments/pilot.py; uses
reasoner/provenance.py for the static assumption footprints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from axiom_graph.graph import AxiomGraph
from reasoner.provenance import compute_provenance


@dataclass
class ForwardChainResult:
    """Per-batch output of run_forward_chain().

    Attributes
    ----------
    n_states          : number of physical states evaluated
    assumption_scores : assumption_id -> float32 array (N,), sigmoid scores in (0,1)
    assumption_flagged: assumption_id -> bool array (N,), True when score < threshold
    node_flagged      : node_id -> bool array (N,), True when any ancestor assumption fired
    threshold         : the decision threshold used
    """

    n_states: int
    assumption_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    assumption_flagged: Dict[str, np.ndarray] = field(default_factory=dict)
    node_flagged: Dict[str, np.ndarray] = field(default_factory=dict)
    threshold: float = 0.5


def run_forward_chain(
    graph: AxiomGraph,
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> ForwardChainResult:
    """Evaluate predicates on a batch of states and propagate flags via provenance.

    Algorithm
    ---------
    1.  For each assumption node with a trained predicate, call predict() on the
        full feature matrix.  Score < threshold means the assumption is flagged.
    2.  Compute the static provenance map for the graph (O(V+E), done once).
    3.  For each node, OR together the flag arrays of every assumption in its
        provenance footprint.  Nodes with empty footprints (observables) are
        never flagged.

    Parameters
    ----------
    graph     : AxiomGraph with ValidityPredicates already attached to
                assumption nodes via node.attach_predicate()
    df        : DataFrame containing all feature columns needed by the attached predicates
    threshold : validity score below which a predicate is considered to have fired

    Returns
    -------
    ForwardChainResult with arrays of length len(df).
    """
    n = len(df)

    # ------------------------------------------------------------------
    # Step 1 — score every assumption node
    # ------------------------------------------------------------------
    assumption_scores: Dict[str, np.ndarray] = {}
    assumption_flagged: Dict[str, np.ndarray] = {}

    for node in graph.assumption_nodes():
        if node.validity_predicate is not None:
            feat_cols = node.validity_predicate.feature_cols
            features = df[feat_cols].values.astype(np.float32)
            scores = node.validity_predicate.predict(features)
            assumption_scores[node.id] = scores
            assumption_flagged[node.id] = scores < threshold
        else:
            # No predicate: treat assumption as always satisfied
            assumption_scores[node.id] = np.ones(n, dtype=np.float32)
            assumption_flagged[node.id] = np.zeros(n, dtype=bool)

    # ------------------------------------------------------------------
    # Step 2 — propagate flags to all nodes via provenance
    # ------------------------------------------------------------------
    prov_map = compute_provenance(graph)
    node_flagged: Dict[str, np.ndarray] = {}

    for node_id, prov_rec in prov_map.items():
        if not prov_rec.assumption_ids:
            node_flagged[node_id] = np.zeros(n, dtype=bool)
        else:
            flagged = np.zeros(n, dtype=bool)
            for aid in prov_rec.assumption_ids:
                if aid in assumption_flagged:
                    flagged |= assumption_flagged[aid]
            node_flagged[node_id] = flagged

    return ForwardChainResult(
        n_states=n,
        assumption_scores=assumption_scores,
        assumption_flagged=assumption_flagged,
        node_flagged=node_flagged,
        threshold=threshold,
    )
