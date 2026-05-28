"""
Axiom graph edge definitions for the ideal gas derivation.

An edge encodes a single derivation step: a directed hyper-edge from an
ordered list of premise node IDs (assumptions and/or derived quantities) to
a single conclusion node ID.  Each edge stores a human-readable rule_label
(short name) and an optional description of the physical reasoning step.

Fits into the system: graph.py wires edges between nodes to form the DAG;
forward_chain.py walks edges in topological order to simulate a derivation;
provenance.py uses premise_ids on each edge to compute transitive assumption
dependencies for each conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DerivationEdge:
    """A directed hyper-edge from a set of premises to one conclusion.

    Parameters
    ----------
    id             : unique identifier (e.g. 'E1')
    premise_ids    : ordered list of node IDs that must hold before the rule fires
    conclusion_id  : node ID produced by this derivation step
    rule_label     : short algebraic or physical name for the step
    description    : fuller explanation of why this step is valid
    """

    id: str
    premise_ids: List[str]
    conclusion_id: str
    rule_label: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.premise_ids:
            raise ValueError(f"Edge '{self.id}' must have at least one premise.")
        if self.conclusion_id in self.premise_ids:
            raise ValueError(
                f"Edge '{self.id}' has conclusion '{self.conclusion_id}' "
                f"listed as its own premise (self-loop not allowed)."
            )
