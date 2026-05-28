"""
Hardcoded ideal gas axiom graph.

The DAG encodes the kinetic-theory derivation of PV = nRT from four physical
assumptions.  Node IDs follow a prefix convention:

    obs_*  — observable (measured) quantities: P, V, T, n
    A*     — assumption nodes (get validity predicates):
                A1_point_particles       molecules have no volume
                A2_no_forces             no intermolecular forces
                A3_elastic_collisions    wall collisions are perfectly elastic
                A4_thermal_equilibrium   Maxwell-Boltzmann / equipartition holds
    D*     — derived intermediate and final results

Edge topology (premises → conclusion):

    E1: [A3]              → D1_momentum_transfer       Δp = 2mv_x per collision
    E2: [A1, A2, A3]      → D2_collision_frequency     f = v_x / (2L)
    E3: [A4]              → D3_mean_kinetic_energy      m⟨v_x²⟩ = kT
    E4: [D1, D2, D3]      → D4_single_particle_pressure P₁ = m⟨v_x²⟩/V
    E5: [D4, A1]          → D5_pressure_ideal           PV = NkT  (A1 again:
                                                          full V is free volume)
    E6: [D5]              → D6_ideal_gas_law             PV = nRT

Ancestor assumptions of D6: {A1, A2, A3, A4} — all four, as expected.

Fits into the system: reasoner/forward_chain.py walks this graph in
topological order; reasoner/provenance.py calls ancestor_assumptions();
experiments/pilot.py calls assumption_nodes() to enumerate which nodes need
a trained ValidityPredicate.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, List, Optional

from axiom_graph.edges import DerivationEdge
from axiom_graph.nodes import Node, NodeKind


class AxiomGraph:
    """A directed acyclic graph of physical assumptions and derivation steps."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[DerivationEdge] = []
        # conclusion_id → [premise_ids]  (all incoming premise IDs, possibly from
        # multiple edges, though in our graph each conclusion has exactly one edge)
        self._pred_map: Dict[str, List[str]] = {}
        # premise_id → [conclusion_ids]
        self._succ_map: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node '{node.id}' already exists in the graph.")
        self.nodes[node.id] = node
        self._pred_map.setdefault(node.id, [])
        self._succ_map.setdefault(node.id, [])

    def add_edge(self, edge: DerivationEdge) -> None:
        for pid in edge.premise_ids:
            if pid not in self.nodes:
                raise ValueError(
                    f"Edge '{edge.id}': premise node '{pid}' not in graph."
                )
        if edge.conclusion_id not in self.nodes:
            raise ValueError(
                f"Edge '{edge.id}': conclusion node '{edge.conclusion_id}' not in graph."
            )
        self.edges.append(edge)
        self._pred_map[edge.conclusion_id].extend(edge.premise_ids)
        for pid in edge.premise_ids:
            self._succ_map[pid].append(edge.conclusion_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def assumption_nodes(self) -> List[Node]:
        """Return all nodes with kind='assumption', in insertion order."""
        return [n for n in self.nodes.values() if n.kind == "assumption"]

    def predecessors(self, node_id: str) -> List[Node]:
        """Immediate predecessor nodes (deduplicated) across all incoming edges."""
        seen = set()
        result = []
        for pid in self._pred_map.get(node_id, []):
            if pid not in seen:
                seen.add(pid)
                result.append(self.nodes[pid])
        return result

    def successors(self, node_id: str) -> List[Node]:
        """Immediate successor nodes (deduplicated) across all outgoing edges."""
        seen = set()
        result = []
        for sid in self._succ_map.get(node_id, []):
            if sid not in seen:
                seen.add(sid)
                result.append(self.nodes[sid])
        return result

    def edges_into(self, node_id: str) -> List[DerivationEdge]:
        """All edges whose conclusion is `node_id`."""
        return [e for e in self.edges if e.conclusion_id == node_id]

    def edges_out_of(self, node_id: str) -> List[DerivationEdge]:
        """All edges that list `node_id` as a premise."""
        return [e for e in self.edges if node_id in e.premise_ids]

    def topological_order(self) -> List[str]:
        """Return node IDs in topological order (Kahn's algorithm).

        Raises ValueError if the graph contains a cycle.
        """
        # Count unique predecessor nodes (not edges) so that multi-premise
        # edges are handled correctly: a node is ready only after all of its
        # distinct premise nodes have been emitted, regardless of how many
        # separate edges they arrive through.
        in_degree: Dict[str, int] = {
            nid: len(set(self._pred_map.get(nid, [])))
            for nid in self.nodes
        }

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: List[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ_id in self._succ_map.get(nid, []):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        if len(order) != len(self.nodes):
            cycle_nodes = [nid for nid, deg in in_degree.items() if deg > 0]
            raise ValueError(
                f"Graph contains a cycle involving nodes: {cycle_nodes}"
            )
        return order

    def is_dag(self) -> bool:
        try:
            self.topological_order()
            return True
        except ValueError:
            return False

    def ancestor_assumptions(self, node_id: str) -> FrozenSet[str]:
        """Return the IDs of all assumption nodes that are ancestors of `node_id`.

        Performs a backward BFS from `node_id` through _pred_map, collecting
        every node with kind='assumption'. This is the provenance footprint:
        every assumption the given node transitively depends on.
        """
        visited: set[str] = set()
        queue: deque[str] = deque([node_id])

        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            for pid in self._pred_map.get(curr, []):
                if pid not in visited:
                    queue.append(pid)

        return frozenset(
            nid
            for nid in visited
            if nid != node_id and self.nodes[nid].kind == "assumption"
        )


# ---------------------------------------------------------------------------
# Hardcoded ideal gas graph
# ---------------------------------------------------------------------------

def build_ideal_gas_graph() -> AxiomGraph:
    """Construct and return the complete hardcoded ideal gas axiom DAG.

    All node and edge IDs are stable strings so that downstream code
    (predicate training, provenance tracking, experiment scripts) can
    reference them by name without importing constants.
    """
    g = AxiomGraph()

    # ---- Observable nodes ------------------------------------------------
    g.add_node(Node(
        id="obs_P", kind="observable",
        label="Pressure P",
        description="Gas pressure in atm, directly measurable.",
    ))
    g.add_node(Node(
        id="obs_V", kind="observable",
        label="Volume V",
        description="Container volume in litres, directly measurable.",
    ))
    g.add_node(Node(
        id="obs_T", kind="observable",
        label="Temperature T",
        description="Absolute temperature in Kelvin, directly measurable.",
    ))
    g.add_node(Node(
        id="obs_n", kind="observable",
        label="Moles n",
        description="Amount of gas in moles, directly measurable.",
    ))

    # ---- Assumption nodes ------------------------------------------------
    g.add_node(Node(
        id="A1_point_particles", kind="assumption",
        label="Point particles (A1)",
        description=(
            "Molecules occupy no volume. The excluded-volume correction "
            "in the van der Waals equation (b term) is negligible."
        ),
    ))
    g.add_node(Node(
        id="A2_no_forces", kind="assumption",
        label="No intermolecular forces (A2)",
        description=(
            "Molecules exert no attractive or repulsive forces on each "
            "other between collisions. The van der Waals 'a' term is negligible."
        ),
    ))
    g.add_node(Node(
        id="A3_elastic_collisions", kind="assumption",
        label="Elastic collisions (A3)",
        description=(
            "Collisions between molecules and container walls conserve "
            "kinetic energy; no energy is transferred to internal modes."
        ),
    ))
    g.add_node(Node(
        id="A4_thermal_equilibrium", kind="assumption",
        label="Thermal equilibrium (A4)",
        description=(
            "The gas is in thermal equilibrium, so molecular speeds follow "
            "the Maxwell-Boltzmann distribution and the equipartition theorem "
            "applies: (1/2)m⟨v_x²⟩ = (1/2)kT."
        ),
    ))

    # ---- Derived nodes ---------------------------------------------------
    g.add_node(Node(
        id="D1_momentum_transfer", kind="derived",
        label="Momentum transfer per collision",
        description=(
            "A molecule with velocity v_x hitting a wall elastically reverses "
            "its x-momentum: Δp = 2mv_x."
        ),
    ))
    g.add_node(Node(
        id="D2_collision_frequency", kind="derived",
        label="Wall collision frequency",
        description=(
            "A point particle in a force-free box of length L bounces "
            "back and forth with unchanged speed; collision rate = v_x / (2L)."
        ),
    ))
    g.add_node(Node(
        id="D3_mean_kinetic_energy", kind="derived",
        label="Mean translational kinetic energy",
        description=(
            "Equipartition theorem applied to one degree of freedom: "
            "m⟨v_x²⟩ = kT (one component of velocity)."
        ),
    ))
    g.add_node(Node(
        id="D4_single_particle_pressure", kind="derived",
        label="Single-particle pressure contribution",
        description=(
            "Force from one molecule on a wall = Δp × frequency = mv_x²/L; "
            "pressure P₁ = mv_x²/V.  After ensemble averaging: P₁ = m⟨v_x²⟩/V."
        ),
    ))
    g.add_node(Node(
        id="D5_pressure_ideal", kind="derived",
        label="N-particle ideal pressure (PV = NkT)",
        description=(
            "Summing N independent point-particle contributions (A1 ensures "
            "the full volume V is free): PV = Nm⟨v_x²⟩ = NkT."
        ),
    ))
    g.add_node(Node(
        id="D6_ideal_gas_law", kind="derived",
        label="Ideal gas law  PV = nRT",
        description=(
            "Converting from N molecules to n moles: N = nNₐ, k = R/Nₐ, "
            "so NkT = nRT.  This is the final result of the derivation."
        ),
    ))

    # ---- Edges -----------------------------------------------------------
    g.add_edge(DerivationEdge(
        id="E1",
        premise_ids=["A3_elastic_collisions"],
        conclusion_id="D1_momentum_transfer",
        rule_label="Elastic collision → Δp = 2mv_x",
        description=(
            "If the wall collision is elastic, the molecule's x-velocity "
            "reverses sign, giving momentum transfer Δp = 2mv_x to the wall."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="E2",
        premise_ids=["A1_point_particles", "A2_no_forces", "A3_elastic_collisions"],
        conclusion_id="D2_collision_frequency",
        rule_label="Free straight-line motion → f = v_x / (2L)",
        description=(
            "A1: the particle has no size, so it traverses the full box. "
            "A2: no forces, so it travels in a straight line at constant speed. "
            "A3: elastic bounce, so speed is preserved.  Round-trip time = 2L/v_x."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="E3",
        premise_ids=["A4_thermal_equilibrium"],
        conclusion_id="D3_mean_kinetic_energy",
        rule_label="Equipartition → m⟨v_x²⟩ = kT",
        description=(
            "Thermal equilibrium implies the Maxwell-Boltzmann distribution, "
            "from which the equipartition theorem gives (1/2)m⟨v_x²⟩ = (1/2)kT."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="E4",
        premise_ids=[
            "D1_momentum_transfer",
            "D2_collision_frequency",
            "D3_mean_kinetic_energy",
        ],
        conclusion_id="D4_single_particle_pressure",
        rule_label="Force = Δp × f, then average → P₁ = m⟨v_x²⟩/V",
        description=(
            "Time-averaged force from one molecule = 2mv_x × (v_x/2L) = mv_x²/L. "
            "Pressure on wall of area L² is force/area = mv_x²/L³ = mv_x²/V. "
            "Ensemble average gives m⟨v_x²⟩/V."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="E5",
        premise_ids=["D4_single_particle_pressure", "A1_point_particles"],
        conclusion_id="D5_pressure_ideal",
        rule_label="Sum N particles (point particles: no excluded volume) → PV = NkT",
        description=(
            "Total pressure = N × P₁ = Nm⟨v_x²⟩/V = NkT/V, so PV = NkT. "
            "A1 is required a second time: the summation assumes each particle "
            "has access to the full volume V (no excluded-volume correction)."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="E6",
        premise_ids=["D5_pressure_ideal"],
        conclusion_id="D6_ideal_gas_law",
        rule_label="N = nNₐ, k = R/Nₐ → PV = nRT",
        description=(
            "Substituting N = nNₐ and k = R/Nₐ into PV = NkT gives PV = nRT."
        ),
    ))

    return g
