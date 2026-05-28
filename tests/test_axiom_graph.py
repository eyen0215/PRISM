"""
Unit tests for axiom_graph/ — nodes, edges, and the hardcoded ideal gas DAG.

Test coverage:
    Node        — creation, predicate slot enforcement, score/is_flagged API
    DerivationEdge — creation, validation guards
    AxiomGraph  — structure queries, topological order, ancestor_assumptions
    build_ideal_gas_graph — physical correctness of the hardcoded DAG
"""

import pytest

from axiom_graph.edges import DerivationEdge
from axiom_graph.graph import AxiomGraph, build_ideal_gas_graph
from axiom_graph.nodes import Node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def g() -> AxiomGraph:
    return build_ideal_gas_graph()


@pytest.fixture
def assumption_node() -> Node:
    return Node(id="A_test", kind="assumption", label="Test assumption")


@pytest.fixture
def observable_node() -> Node:
    return Node(id="obs_test", kind="observable", label="Test observable")


@pytest.fixture
def derived_node() -> Node:
    return Node(id="D_test", kind="derived", label="Test derived")


# ---------------------------------------------------------------------------
# Node — creation and fields
# ---------------------------------------------------------------------------

class TestNodeCreation:
    def test_observable_kind(self, observable_node):
        assert observable_node.kind == "observable"

    def test_assumption_kind(self, assumption_node):
        assert assumption_node.kind == "assumption"

    def test_derived_kind(self, derived_node):
        assert derived_node.kind == "derived"

    def test_description_defaults_to_empty(self):
        n = Node(id="x", kind="derived", label="X")
        assert n.description == ""

    def test_validity_predicate_starts_none(self, assumption_node):
        assert assumption_node.validity_predicate is None


# ---------------------------------------------------------------------------
# Node — predicate slot
# ---------------------------------------------------------------------------

class TestPredicateSlot:
    def test_attach_to_assumption_succeeds(self, assumption_node):
        assumption_node.attach_predicate(lambda _: 0.9)
        assert assumption_node.validity_predicate is not None

    def test_attach_to_observable_raises(self, observable_node):
        with pytest.raises(TypeError, match="assumption nodes"):
            observable_node.attach_predicate(lambda _: 0.9)

    def test_attach_to_derived_raises(self, derived_node):
        with pytest.raises(TypeError, match="assumption nodes"):
            derived_node.attach_predicate(lambda _: 0.9)

    def test_score_none_without_predicate(self):
        n = Node(id="A_fresh", kind="assumption", label="Fresh")
        assert n.score(None) is None

    def test_score_calls_predicate(self):
        n = Node(id="A_s", kind="assumption", label="S")
        n.attach_predicate(lambda _: 0.8)
        assert abs(n.score("anything") - 0.8) < 1e-9

    def test_score_returns_float(self):
        n = Node(id="A_f", kind="assumption", label="F")
        n.attach_predicate(lambda _: 1)  # integer return
        result = n.score(None)
        assert isinstance(result, float)

    def test_is_flagged_none_without_predicate(self):
        n = Node(id="A_nf", kind="assumption", label="NF")
        assert n.is_flagged(None) is None

    def test_is_flagged_true_when_score_below_default_threshold(self):
        n = Node(id="A_low", kind="assumption", label="Low")
        n.attach_predicate(lambda _: 0.3)  # below 0.5
        assert n.is_flagged(None) is True

    def test_is_flagged_false_when_score_above_default_threshold(self):
        n = Node(id="A_high", kind="assumption", label="High")
        n.attach_predicate(lambda _: 0.7)  # above 0.5
        assert n.is_flagged(None) is False

    def test_is_flagged_custom_threshold(self):
        n = Node(id="A_ct", kind="assumption", label="CT")
        n.attach_predicate(lambda _: 0.6)
        assert n.is_flagged(None, threshold=0.8) is True
        assert n.is_flagged(None, threshold=0.4) is False

    def test_is_flagged_exactly_at_threshold_not_flagged(self):
        # score < threshold (strict), so exactly at threshold → not flagged
        n = Node(id="A_eq", kind="assumption", label="Eq")
        n.attach_predicate(lambda _: 0.5)
        assert n.is_flagged(None, threshold=0.5) is False


# ---------------------------------------------------------------------------
# DerivationEdge — creation and guards
# ---------------------------------------------------------------------------

class TestDerivationEdge:
    def test_basic_creation(self):
        e = DerivationEdge(
            id="E_t", premise_ids=["A", "B"], conclusion_id="C",
            rule_label="Test rule",
        )
        assert e.conclusion_id == "C"
        assert e.premise_ids == ["A", "B"]

    def test_empty_premises_raises(self):
        with pytest.raises(ValueError, match="at least one premise"):
            DerivationEdge(id="E_bad", premise_ids=[], conclusion_id="C",
                           rule_label="bad")

    def test_self_loop_raises(self):
        with pytest.raises(ValueError, match="self-loop"):
            DerivationEdge(id="E_loop", premise_ids=["C"], conclusion_id="C",
                           rule_label="loop")

    def test_description_optional(self):
        e = DerivationEdge(id="E_nd", premise_ids=["A"], conclusion_id="B",
                           rule_label="R")
        assert e.description == ""


# ---------------------------------------------------------------------------
# AxiomGraph — node/edge counts and membership
# ---------------------------------------------------------------------------

class TestGraphCounts:
    def test_node_count(self, g):
        # 4 observables + 4 assumptions + 6 derived = 14
        assert len(g.nodes) == 14

    def test_edge_count(self, g):
        assert len(g.edges) == 6

    def test_four_assumption_nodes(self, g):
        assert len(g.assumption_nodes()) == 4

    def test_assumption_ids(self, g):
        ids = {n.id for n in g.assumption_nodes()}
        assert ids == {
            "A1_point_particles",
            "A2_no_forces",
            "A3_elastic_collisions",
            "A4_thermal_equilibrium",
        }

    def test_observable_ids(self, g):
        obs = {nid for nid, n in g.nodes.items() if n.kind == "observable"}
        assert obs == {"obs_P", "obs_V", "obs_T", "obs_n"}

    def test_derived_ids(self, g):
        derived = {nid for nid, n in g.nodes.items() if n.kind == "derived"}
        assert derived == {
            "D1_momentum_transfer",
            "D2_collision_frequency",
            "D3_mean_kinetic_energy",
            "D4_single_particle_pressure",
            "D5_pressure_ideal",
            "D6_ideal_gas_law",
        }

    def test_duplicate_node_raises(self, g):
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(Node(id="obs_P", kind="observable", label="dup"))

    def test_edge_with_unknown_premise_raises(self):
        g2 = AxiomGraph()
        g2.add_node(Node(id="C", kind="derived", label="C"))
        with pytest.raises(ValueError, match="premise node 'UNKNOWN' not in graph"):
            g2.add_edge(DerivationEdge(id="E", premise_ids=["UNKNOWN"],
                                       conclusion_id="C", rule_label="r"))

    def test_edge_with_unknown_conclusion_raises(self):
        g2 = AxiomGraph()
        g2.add_node(Node(id="A", kind="assumption", label="A"))
        with pytest.raises(ValueError, match="conclusion node 'UNKNOWN' not in graph"):
            g2.add_edge(DerivationEdge(id="E", premise_ids=["A"],
                                       conclusion_id="UNKNOWN", rule_label="r"))


# ---------------------------------------------------------------------------
# AxiomGraph — structural correctness
# ---------------------------------------------------------------------------

class TestGraphStructure:
    def test_is_dag(self, g):
        assert g.is_dag() is True

    def test_topological_order_covers_all_nodes(self, g):
        order = g.topological_order()
        assert len(order) == len(g.nodes)
        assert set(order) == set(g.nodes.keys())

    def test_topological_order_premises_before_conclusion(self, g):
        order = g.topological_order()
        pos = {nid: i for i, nid in enumerate(order)}
        for edge in g.edges:
            for pid in edge.premise_ids:
                assert pos[pid] < pos[edge.conclusion_id], (
                    f"Edge {edge.id}: premise '{pid}' appears after "
                    f"conclusion '{edge.conclusion_id}' in topological order."
                )

    def test_observable_nodes_have_no_predecessors(self, g):
        for nid, node in g.nodes.items():
            if node.kind == "observable":
                assert g.predecessors(nid) == [], (
                    f"Observable '{nid}' should have no predecessors."
                )

    def test_assumption_nodes_have_no_predecessors(self, g):
        for node in g.assumption_nodes():
            assert g.predecessors(node.id) == [], (
                f"Assumption '{node.id}' should have no predecessors."
            )

    def test_ideal_gas_law_has_no_successors(self, g):
        assert g.successors("D6_ideal_gas_law") == []

    def test_is_dag_detects_cycle(self):
        g2 = AxiomGraph()
        g2.add_node(Node(id="X", kind="derived", label="X"))
        g2.add_node(Node(id="Y", kind="derived", label="Y"))
        g2.add_edge(DerivationEdge(id="E1", premise_ids=["X"],
                                   conclusion_id="Y", rule_label="X→Y"))
        g2.add_edge(DerivationEdge(id="E2", premise_ids=["Y"],
                                   conclusion_id="X", rule_label="Y→X"))
        assert g2.is_dag() is False


# ---------------------------------------------------------------------------
# AxiomGraph — predecessors / successors
# ---------------------------------------------------------------------------

class TestPredecessorsSuccessors:
    def test_d1_predecessor_is_a3(self, g):
        preds = {n.id for n in g.predecessors("D1_momentum_transfer")}
        assert preds == {"A3_elastic_collisions"}

    def test_d2_predecessors(self, g):
        preds = {n.id for n in g.predecessors("D2_collision_frequency")}
        assert preds == {"A1_point_particles", "A2_no_forces", "A3_elastic_collisions"}

    def test_d3_predecessor_is_a4(self, g):
        preds = {n.id for n in g.predecessors("D3_mean_kinetic_energy")}
        assert preds == {"A4_thermal_equilibrium"}

    def test_d5_predecessors(self, g):
        preds = {n.id for n in g.predecessors("D5_pressure_ideal")}
        assert preds == {"D4_single_particle_pressure", "A1_point_particles"}

    def test_a1_has_two_successors(self, g):
        succs = {n.id for n in g.successors("A1_point_particles")}
        # A1 appears as premise in E2 (→ D2) and E5 (→ D5)
        assert succs == {"D2_collision_frequency", "D5_pressure_ideal"}

    def test_a3_has_two_successors(self, g):
        succs = {n.id for n in g.successors("A3_elastic_collisions")}
        assert succs == {"D1_momentum_transfer", "D2_collision_frequency"}

    def test_d6_has_no_successors(self, g):
        assert g.successors("D6_ideal_gas_law") == []


# ---------------------------------------------------------------------------
# Provenance — ancestor_assumptions
# ---------------------------------------------------------------------------

class TestAncestorAssumptions:
    def test_ideal_gas_law_depends_on_all_four_assumptions(self, g):
        anc = g.ancestor_assumptions("D6_ideal_gas_law")
        assert anc == {
            "A1_point_particles",
            "A2_no_forces",
            "A3_elastic_collisions",
            "A4_thermal_equilibrium",
        }

    def test_momentum_transfer_depends_only_on_a3(self, g):
        anc = g.ancestor_assumptions("D1_momentum_transfer")
        assert anc == {"A3_elastic_collisions"}

    def test_collision_frequency_does_not_depend_on_a4(self, g):
        anc = g.ancestor_assumptions("D2_collision_frequency")
        assert "A4_thermal_equilibrium" not in anc

    def test_mean_kinetic_energy_depends_only_on_a4(self, g):
        anc = g.ancestor_assumptions("D3_mean_kinetic_energy")
        assert anc == {"A4_thermal_equilibrium"}

    def test_collision_frequency_ancestors(self, g):
        anc = g.ancestor_assumptions("D2_collision_frequency")
        assert anc == {"A1_point_particles", "A2_no_forces", "A3_elastic_collisions"}

    def test_single_particle_pressure_has_all_four(self, g):
        # D4 = f(D1, D2, D3), which pulls in A1, A2, A3 (via D1/D2) and A4 (via D3)
        anc = g.ancestor_assumptions("D4_single_particle_pressure")
        assert anc == {
            "A1_point_particles",
            "A2_no_forces",
            "A3_elastic_collisions",
            "A4_thermal_equilibrium",
        }

    def test_observable_has_no_ancestor_assumptions(self, g):
        for obs_id in ("obs_P", "obs_V", "obs_T", "obs_n"):
            assert g.ancestor_assumptions(obs_id) == frozenset()

    def test_assumption_node_is_not_its_own_ancestor(self, g):
        # The assumption itself is in visited, but the method excludes node_id itself
        for node in g.assumption_nodes():
            anc = g.ancestor_assumptions(node.id)
            assert node.id not in anc

    def test_assumption_node_has_empty_ancestors(self, g):
        # Assumption nodes are roots — no predecessors
        for node in g.assumption_nodes():
            assert g.ancestor_assumptions(node.id) == frozenset()

    def test_no_forces_isolated_from_momentum_transfer(self, g):
        # A2 contributes to D2 (free motion), but not to D1 (elastic collision only)
        anc = g.ancestor_assumptions("D1_momentum_transfer")
        assert "A2_no_forces" not in anc


# ---------------------------------------------------------------------------
# All assumption nodes start with no predicate
# ---------------------------------------------------------------------------

class TestPredicateSlotsOnGraph:
    def test_all_assumption_nodes_start_without_predicates(self, g):
        for node in g.assumption_nodes():
            assert node.validity_predicate is None, (
                f"Assumption '{node.id}' should start with no predicate."
            )

    def test_predicate_attached_to_graph_node(self, g):
        node = g.nodes["A4_thermal_equilibrium"]
        node.attach_predicate(lambda _: 0.95)
        assert node.validity_predicate is not None
        # Clean up so other tests see no predicate
        node.validity_predicate = None
