"""
Axiom graph node definitions for the ideal gas system.

Three node kinds:
  'observable'  — directly measured quantities (P, V, T, n); leaf inputs to the DAG.
  'assumption'  — physical axioms (A1–A4); each gets a validity_predicate slot.
  'derived'     — intermediate or final quantities produced by derivation rules.

Fits into the system: axiom_graph/graph.py assembles these into a DAG;
reasoner/provenance.py traverses the DAG to track which assumptions underpin
each derived result; validity_predicates/predicate.py attaches an MLP to each
assumption node via Node.attach_predicate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

NodeKind = Literal["observable", "assumption", "derived"]


@dataclass
class Node:
    """A single node in the axiom graph.

    Parameters
    ----------
    id          : unique identifier (e.g. 'A1_point_particles', 'obs_P')
    kind        : 'observable' | 'assumption' | 'derived'
    label       : short human-readable name
    description : physical meaning of this node
    """

    id: str
    kind: NodeKind
    label: str
    description: str = ""
    # Only assumption nodes use this slot; enforced by attach_predicate().
    validity_predicate: Optional[Any] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Predicate interface
    # ------------------------------------------------------------------

    def attach_predicate(self, predicate: Any) -> None:
        """Attach a trained validity predicate (any callable) to this node.

        Raises TypeError if called on an observable or derived node —
        predicates belong exclusively to assumption nodes.
        """
        if self.kind != "assumption":
            raise TypeError(
                f"attach_predicate() is only valid for assumption nodes; "
                f"node '{self.id}' has kind '{self.kind}'."
            )
        self.validity_predicate = predicate

    def score(self, features: Any) -> Optional[float]:
        """Call the predicate and return a validity score in [0, 1].

        Returns None if no predicate has been attached yet. The predicate
        callable receives `features` directly (e.g., a numpy array or torch
        tensor) and must return a float-compatible scalar.
        """
        if self.validity_predicate is None:
            return None
        return float(self.validity_predicate(features))

    def is_flagged(self, features: Any, threshold: float = 0.5) -> Optional[bool]:
        """Return True when the predicate score falls below `threshold`.

        A low score means the predicate believes this assumption is violated.
        Returns None if no predicate is attached.
        """
        s = self.score(features)
        if s is None:
            return None
        return s < threshold
