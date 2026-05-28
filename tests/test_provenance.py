"""
Exhaustive unit tests for reasoner/provenance.py.

Structure
---------
TestProvenanceRecord     — the dataclass itself (construction, query methods)
TestMinimalGraphs        — tiny hand-built 2–4 node graphs; every result is
                           computed by hand before the assertion
TestIdealGasProvenance   — integration test on build_ideal_gas_graph(); each
                           node's expected value is pre-computed in the module
                           docstring of this file and verified independently
TestFlaggedNodes         — flagged_nodes() filtering function

Hand-computed expected values for the ideal gas graph
-----------------------------------------------------
obs_P, obs_V, obs_T, obs_n : {}
A1_point_particles          : {A1}
A2_no_forces                : {A2}
A3_elastic_collisions       : {A3}
A4_thermal_equilibrium      : {A4}
D1_momentum_transfer        : {A3}               (only E1: A3→D1)
D2_collision_frequency      : {A1,A2,A3}         (E2: A1,A2,A3→D2)
D3_mean_kinetic_energy      : {A4}               (E3: A4→D3)
D4_single_particle_pressure : {A1,A2,A3,A4}      ({A3}∪{A1,A2,A3}∪{A4})
D5_pressure_ideal           : {A1,A2,A3,A4}      ({A1,A2,A3,A4}∪{A1})
D6_ideal_gas_law            : {A1,A2,A3,A4}      ({A1,A2,A3,A4})
"""

import pytest

from axiom_graph.graph import AxiomGraph, build_ideal_gas_graph
from axiom_graph.edges import DerivationEdge
from axiom_graph.nodes import Node
from reasoner.provenance import (
    ProvenanceRecord,
    compute_provenance,
    flagged_nodes,
)

# Short aliases used in expected-value literals throughout this file
A1 = "A1_point_particles"
A2 = "A2_no_forces"
A3 = "A3_elastic_collisions"
A4 = "A4_thermal_equilibrium"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(*specs) -> AxiomGraph:
    """Build a small AxiomGraph from a compact spec list.

    Each element of specs is either:
      ('obs',  id)               — observable node
      ('asm',  id)               — assumption node
      ('der',  id)               — derived node (no edges yet)
      ('edge', id, [prem...], conclusion)  — DerivationEdge

    Nodes must be declared before edges that reference them.
    """
    g = AxiomGraph()
    edge_counter = [0]
    for spec in specs:
        kind = spec[0]
        if kind == "obs":
            g.add_node(Node(id=spec[1], kind="observable", label=spec[1]))
        elif kind == "asm":
            g.add_node(Node(id=spec[1], kind="assumption", label=spec[1]))
        elif kind == "der":
            g.add_node(Node(id=spec[1], kind="derived", label=spec[1]))
        elif kind == "edge":
            _, eid, prems, conc = spec
            edge_counter[0] += 1
            g.add_edge(DerivationEdge(
                id=eid, premise_ids=prems, conclusion_id=conc,
                rule_label=f"rule_{edge_counter[0]}",
            ))
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ideal_graph() -> AxiomGraph:
    return build_ideal_gas_graph()


@pytest.fixture(scope="module")
def ideal_prov(ideal_graph) -> dict:
    return compute_provenance(ideal_graph)


# ===========================================================================
# TestProvenanceRecord — the dataclass itself
# ===========================================================================

class TestProvenanceRecord:

    # ---- constructors ----

    def test_for_observable_empty_ids(self):
        r = ProvenanceRecord.for_observable("obs_P")
        assert r.assumption_ids == frozenset()

    def test_for_observable_node_id(self):
        r = ProvenanceRecord.for_observable("obs_V")
        assert r.node_id == "obs_V"

    def test_for_assumption_singleton(self):
        r = ProvenanceRecord.for_assumption("A1")
        assert r.assumption_ids == frozenset({"A1"})

    def test_for_assumption_node_id(self):
        r = ProvenanceRecord.for_assumption("A1")
        assert r.node_id == "A1"

    def test_for_assumption_id_equals_node_id(self):
        # The assumption's own ID is the only element of its provenance
        r = ProvenanceRecord.for_assumption("A99")
        assert r.assumption_ids == frozenset({"A99"})

    def test_from_premises_single_assumption(self):
        prem = ProvenanceRecord.for_assumption("A1")
        r = ProvenanceRecord.from_premises("D1", [prem])
        assert r.assumption_ids == frozenset({"A1"})

    def test_from_premises_two_assumptions_merged(self):
        p1 = ProvenanceRecord.for_assumption("A")
        p2 = ProvenanceRecord.for_assumption("B")
        r = ProvenanceRecord.from_premises("D", [p1, p2])
        assert r.assumption_ids == frozenset({"A", "B"})

    def test_from_premises_no_duplicates_on_shared_ancestor(self):
        # Both premises share assumption A — union should not double-count
        p1 = ProvenanceRecord.from_premises("D1", [ProvenanceRecord.for_assumption("A")])
        p2 = ProvenanceRecord.from_premises("D2", [ProvenanceRecord.for_assumption("A")])
        r = ProvenanceRecord.from_premises("D3", [p1, p2])
        assert r.assumption_ids == frozenset({"A"})

    def test_from_premises_observable_contributes_nothing(self):
        obs = ProvenanceRecord.for_observable("obs_T")
        asm = ProvenanceRecord.for_assumption("A1")
        r = ProvenanceRecord.from_premises("D", [obs, asm])
        assert r.assumption_ids == frozenset({"A1"})

    def test_from_premises_empty_list_gives_empty(self):
        r = ProvenanceRecord.from_premises("D", [])
        assert r.assumption_ids == frozenset()

    def test_from_premises_all_observables_gives_empty(self):
        o1 = ProvenanceRecord.for_observable("obs_P")
        o2 = ProvenanceRecord.for_observable("obs_T")
        r = ProvenanceRecord.from_premises("D", [o1, o2])
        assert r.assumption_ids == frozenset()

    def test_frozen_cannot_set_assumption_ids(self):
        r = ProvenanceRecord.for_assumption("A1")
        with pytest.raises((AttributeError, TypeError)):
            r.assumption_ids = frozenset()  # type: ignore[misc]

    def test_frozen_cannot_set_node_id(self):
        r = ProvenanceRecord.for_observable("obs_P")
        with pytest.raises((AttributeError, TypeError)):
            r.node_id = "other"  # type: ignore[misc]

    def test_equality_same_content(self):
        r1 = ProvenanceRecord(node_id="D", assumption_ids=frozenset({"A", "B"}))
        r2 = ProvenanceRecord(node_id="D", assumption_ids=frozenset({"B", "A"}))
        assert r1 == r2

    def test_equality_different_node_id(self):
        r1 = ProvenanceRecord(node_id="D1", assumption_ids=frozenset({"A"}))
        r2 = ProvenanceRecord(node_id="D2", assumption_ids=frozenset({"A"}))
        assert r1 != r2

    # ---- depends_on ----

    def test_depends_on_true(self):
        r = ProvenanceRecord.from_premises("D", [ProvenanceRecord.for_assumption("A1")])
        assert r.depends_on("A1") is True

    def test_depends_on_false(self):
        r = ProvenanceRecord.from_premises("D", [ProvenanceRecord.for_assumption("A1")])
        assert r.depends_on("A2") is False

    def test_depends_on_empty_provenance(self):
        r = ProvenanceRecord.for_observable("obs")
        assert r.depends_on("A1") is False

    def test_depends_on_assumption_itself(self):
        r = ProvenanceRecord.for_assumption("A1")
        assert r.depends_on("A1") is True

    # ---- is_flagged_by ----

    def test_is_flagged_by_matching_assumption(self):
        r = ProvenanceRecord.for_assumption("A1")
        assert r.is_flagged_by({"A1"}) is True

    def test_is_flagged_by_non_matching(self):
        r = ProvenanceRecord.for_assumption("A1")
        assert r.is_flagged_by({"A2"}) is False

    def test_is_flagged_by_empty_flagged_set(self):
        r = ProvenanceRecord.for_assumption("A1")
        assert r.is_flagged_by(set()) is False

    def test_is_flagged_by_empty_provenance(self):
        r = ProvenanceRecord.for_observable("obs")
        assert r.is_flagged_by({"A1", "A2"}) is False

    def test_is_flagged_by_partial_overlap(self):
        r = ProvenanceRecord(node_id="D", assumption_ids=frozenset({"A1", "A2"}))
        # Flagging A3 and A2: A2 overlaps → flagged
        assert r.is_flagged_by({"A2", "A3"}) is True

    def test_is_flagged_by_superset_of_assumptions(self):
        # Flagging more assumptions than the node depends on still works
        r = ProvenanceRecord(node_id="D", assumption_ids=frozenset({"A1"}))
        assert r.is_flagged_by({"A1", "A2", "A3", "A4"}) is True


# ===========================================================================
# TestMinimalGraphs — tiny hand-built graphs, each property tested in isolation
# ===========================================================================

class TestMinimalGraphs:

    # ---- A → D  (single assumption, single derivation step) ----

    def test_single_step_derived_inherits_assumption(self):
        g = _make_graph(
            ("asm", "A"),
            ("der", "D"),
            ("edge", "E1", ["A"], "D"),
        )
        prov = compute_provenance(g)
        assert prov["D"].assumption_ids == frozenset({"A"})

    def test_single_step_assumption_provenance(self):
        g = _make_graph(("asm", "A"), ("der", "D"), ("edge", "E1", ["A"], "D"))
        prov = compute_provenance(g)
        assert prov["A"].assumption_ids == frozenset({"A"})

    # ---- A → D1 → D2  (chain: provenance propagates transitively) ----

    def test_chain_intermediate_inherits_assumption(self):
        g = _make_graph(
            ("asm", "A"),
            ("der", "D1"),
            ("der", "D2"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["D1"], "D2"),
        )
        prov = compute_provenance(g)
        assert prov["D1"].assumption_ids == frozenset({"A"})

    def test_chain_end_inherits_through_intermediate(self):
        g = _make_graph(
            ("asm", "A"),
            ("der", "D1"),
            ("der", "D2"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["D1"], "D2"),
        )
        prov = compute_provenance(g)
        assert prov["D2"].assumption_ids == frozenset({"A"})

    # ---- A, B → D  (multi-premise single edge: merge) ----

    def test_multi_premise_edge_merges_assumptions(self):
        g = _make_graph(
            ("asm", "A"),
            ("asm", "B"),
            ("der", "D"),
            ("edge", "E1", ["A", "B"], "D"),
        )
        prov = compute_provenance(g)
        assert prov["D"].assumption_ids == frozenset({"A", "B"})

    # ---- A → D1, B → D2, [D1, D2] → D3  (two-premise merge through derived) ----

    def test_two_independent_chains_merge_at_final(self):
        g = _make_graph(
            ("asm", "A"),
            ("asm", "B"),
            ("der", "D1"),
            ("der", "D2"),
            ("der", "D3"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["B"], "D2"),
            ("edge", "E3", ["D1", "D2"], "D3"),
        )
        prov = compute_provenance(g)
        assert prov["D3"].assumption_ids == frozenset({"A", "B"})

    def test_two_independent_chains_intermediate_isolation(self):
        g = _make_graph(
            ("asm", "A"),
            ("asm", "B"),
            ("der", "D1"),
            ("der", "D2"),
            ("der", "D3"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["B"], "D2"),
            ("edge", "E3", ["D1", "D2"], "D3"),
        )
        prov = compute_provenance(g)
        # D1 only depends on A — not B
        assert prov["D1"].assumption_ids == frozenset({"A"})
        assert prov["D2"].assumption_ids == frozenset({"B"})

    # ---- Diamond: A → D1, A → D2, [D1, D2] → D3 (no double-counting) ----

    def test_diamond_no_duplicate_assumption(self):
        g = _make_graph(
            ("asm", "A"),
            ("der", "D1"),
            ("der", "D2"),
            ("der", "D3"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["A"], "D2"),
            ("edge", "E3", ["D1", "D2"], "D3"),
        )
        prov = compute_provenance(g)
        assert prov["D3"].assumption_ids == frozenset({"A"})

    # ---- Observable flows through: obs → D ----

    def test_observable_premise_contributes_no_assumptions(self):
        g = _make_graph(
            ("obs", "obs_T"),
            ("der", "D"),
            ("edge", "E1", ["obs_T"], "D"),
        )
        prov = compute_provenance(g)
        assert prov["D"].assumption_ids == frozenset()

    # ---- Mixed: [obs, asm] → D ----

    def test_mixed_observable_and_assumption_premise(self):
        g = _make_graph(
            ("obs", "obs_T"),
            ("asm", "A"),
            ("der", "D"),
            ("edge", "E1", ["obs_T", "A"], "D"),
        )
        prov = compute_provenance(g)
        assert prov["D"].assumption_ids == frozenset({"A"})

    # ---- Standalone observable (no edges) ----

    def test_standalone_observable_empty_provenance(self):
        g = _make_graph(("obs", "obs_P"))
        prov = compute_provenance(g)
        assert prov["obs_P"].assumption_ids == frozenset()

    # ---- Standalone assumption ----

    def test_standalone_assumption_singleton_provenance(self):
        g = _make_graph(("asm", "A1"))
        prov = compute_provenance(g)
        assert prov["A1"].assumption_ids == frozenset({"A1"})

    # ---- Three-level chain: A → D1 → D2 → D3 ----

    def test_three_level_chain_full_propagation(self):
        g = _make_graph(
            ("asm", "A"),
            ("der", "D1"),
            ("der", "D2"),
            ("der", "D3"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["D1"], "D2"),
            ("edge", "E3", ["D2"], "D3"),
        )
        prov = compute_provenance(g)
        for nid in ("D1", "D2", "D3"):
            assert prov[nid].assumption_ids == frozenset({"A"}), \
                f"Expected {{A}} for {nid}, got {prov[nid].assumption_ids}"

    # ---- Two assumptions, only one path reaches final node ----

    def test_unreachable_assumption_not_in_provenance(self):
        # A → D1 → D3
        # B is standalone — not connected to D3
        g = _make_graph(
            ("asm", "A"),
            ("asm", "B"),
            ("der", "D1"),
            ("der", "D3"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["D1"], "D3"),
        )
        prov = compute_provenance(g)
        assert "B" not in prov["D3"].assumption_ids

    # ---- node_id in record matches the graph node ----

    def test_record_node_id_matches_graph_node(self):
        g = _make_graph(("asm", "A"), ("der", "D"), ("edge", "E1", ["A"], "D"))
        prov = compute_provenance(g)
        for nid, rec in prov.items():
            assert rec.node_id == nid


# ===========================================================================
# TestIdealGasProvenance — integration on the full hardcoded graph
# Hand-computed expected values are in this file's module docstring.
# ===========================================================================

class TestIdealGasProvenance:

    # ---- All nodes are present ----

    def test_all_nodes_have_a_record(self, ideal_graph, ideal_prov):
        assert set(ideal_prov.keys()) == set(ideal_graph.nodes.keys())

    # ---- Observable nodes: empty assumption_ids ----

    @pytest.mark.parametrize("obs_id", ["obs_P", "obs_V", "obs_T", "obs_n"])
    def test_observable_has_empty_provenance(self, ideal_prov, obs_id):
        assert ideal_prov[obs_id].assumption_ids == frozenset()

    # ---- Assumption nodes: singleton {self} ----

    @pytest.mark.parametrize("asm_id", [A1, A2, A3, A4])
    def test_assumption_has_singleton_provenance(self, ideal_prov, asm_id):
        assert ideal_prov[asm_id].assumption_ids == frozenset({asm_id})

    # ---- Derived nodes: hand-computed expected values ----

    def test_d1_provenance(self, ideal_prov):
        # E1: [A3] → D1
        assert ideal_prov["D1_momentum_transfer"].assumption_ids == frozenset({A3})

    def test_d2_provenance(self, ideal_prov):
        # E2: [A1, A2, A3] → D2
        assert ideal_prov["D2_collision_frequency"].assumption_ids == frozenset({A1, A2, A3})

    def test_d3_provenance(self, ideal_prov):
        # E3: [A4] → D3
        assert ideal_prov["D3_mean_kinetic_energy"].assumption_ids == frozenset({A4})

    def test_d4_provenance(self, ideal_prov):
        # D4 ← D1({A3}), D2({A1,A2,A3}), D3({A4}) → {A1,A2,A3,A4}
        assert ideal_prov["D4_single_particle_pressure"].assumption_ids == frozenset(
            {A1, A2, A3, A4}
        )

    def test_d5_provenance(self, ideal_prov):
        # D5 ← D4({A1,A2,A3,A4}), A1({A1}) → {A1,A2,A3,A4}
        assert ideal_prov["D5_pressure_ideal"].assumption_ids == frozenset(
            {A1, A2, A3, A4}
        )

    def test_d6_provenance(self, ideal_prov):
        # D6 ← D5({A1,A2,A3,A4}) → {A1,A2,A3,A4}
        assert ideal_prov["D6_ideal_gas_law"].assumption_ids == frozenset(
            {A1, A2, A3, A4}
        )

    # ---- Isolation: specific assumptions absent from specific nodes ----

    def test_a4_not_in_d1_provenance(self, ideal_prov):
        assert not ideal_prov["D1_momentum_transfer"].depends_on(A4)

    def test_a4_not_in_d2_provenance(self, ideal_prov):
        assert not ideal_prov["D2_collision_frequency"].depends_on(A4)

    def test_a2_not_in_d1_provenance(self, ideal_prov):
        assert not ideal_prov["D1_momentum_transfer"].depends_on(A2)

    def test_a1_not_in_d3_provenance(self, ideal_prov):
        assert not ideal_prov["D3_mean_kinetic_energy"].depends_on(A1)

    def test_a2_not_in_d3_provenance(self, ideal_prov):
        assert not ideal_prov["D3_mean_kinetic_energy"].depends_on(A2)

    def test_a3_not_in_d3_provenance(self, ideal_prov):
        assert not ideal_prov["D3_mean_kinetic_energy"].depends_on(A3)

    def test_only_a3_in_d1_provenance(self, ideal_prov):
        assert ideal_prov["D1_momentum_transfer"].assumption_ids == frozenset({A3})

    def test_only_a4_in_d3_provenance(self, ideal_prov):
        assert ideal_prov["D3_mean_kinetic_energy"].assumption_ids == frozenset({A4})

    # ---- is_flagged_by on real nodes ----

    def test_d6_flagged_by_any_single_assumption(self, ideal_prov):
        for asm in (A1, A2, A3, A4):
            assert ideal_prov["D6_ideal_gas_law"].is_flagged_by({asm}), \
                f"D6 should be flagged by {asm}"

    def test_d1_not_flagged_by_a2(self, ideal_prov):
        assert not ideal_prov["D1_momentum_transfer"].is_flagged_by({A2})

    def test_d3_flagged_by_a4(self, ideal_prov):
        assert ideal_prov["D3_mean_kinetic_energy"].is_flagged_by({A4})

    def test_d3_not_flagged_by_a1_a2_a3(self, ideal_prov):
        assert not ideal_prov["D3_mean_kinetic_energy"].is_flagged_by({A1, A2, A3})

    def test_obs_never_flagged(self, ideal_prov):
        for obs_id in ("obs_P", "obs_V", "obs_T", "obs_n"):
            assert not ideal_prov[obs_id].is_flagged_by({A1, A2, A3, A4})


# ===========================================================================
# TestFlaggedNodes — the filtering function
# ===========================================================================

class TestFlaggedNodes:

    # ---- Basic behaviour ----

    def test_empty_flagged_set_returns_empty_dict(self, ideal_prov):
        result = flagged_nodes(ideal_prov, set())
        assert result == {}

    def test_flagging_nonexistent_id_returns_empty(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {"DOES_NOT_EXIST"})
        assert result == {}

    def test_returns_dict_of_provenance_records(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A1})
        assert all(isinstance(v, ProvenanceRecord) for v in result.values())

    # ---- Flagging A4 should affect D3, D4, D5, D6, and A4 itself ----

    def test_flag_a4_includes_d3(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        assert "D3_mean_kinetic_energy" in result

    def test_flag_a4_includes_d4(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        assert "D4_single_particle_pressure" in result

    def test_flag_a4_includes_d5(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        assert "D5_pressure_ideal" in result

    def test_flag_a4_includes_d6(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        assert "D6_ideal_gas_law" in result

    def test_flag_a4_includes_a4_itself(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        assert A4 in result

    def test_flag_a4_excludes_d1(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        assert "D1_momentum_transfer" not in result

    def test_flag_a4_excludes_d2(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        assert "D2_collision_frequency" not in result

    def test_flag_a4_excludes_observables(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A4})
        for obs_id in ("obs_P", "obs_V", "obs_T", "obs_n"):
            assert obs_id not in result

    # ---- Flagging A3 should NOT affect D3 (thermal equilibrium branch) ----

    def test_flag_a3_excludes_d3(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A3})
        assert "D3_mean_kinetic_energy" not in result

    def test_flag_a3_includes_d1(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A3})
        assert "D1_momentum_transfer" in result

    def test_flag_a3_includes_d2(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A3})
        assert "D2_collision_frequency" in result

    # ---- Flagging A1: affects D2 (free motion), D4, D5, D6, A1 itself ----
    #      but NOT D1 (momentum transfer; A1 not a premise) or D3 ----

    def test_flag_a1_excludes_d1(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A1})
        assert "D1_momentum_transfer" not in result

    def test_flag_a1_excludes_d3(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A1})
        assert "D3_mean_kinetic_energy" not in result

    def test_flag_a1_includes_d2(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A1})
        assert "D2_collision_frequency" in result

    def test_flag_a1_includes_d5(self, ideal_prov):
        # A1 appears a second time as premise of E5 → D5
        result = flagged_nodes(ideal_prov, {A1})
        assert "D5_pressure_ideal" in result

    # ---- All four flagged: every non-observable node is affected ----

    def test_flag_all_assumptions_hits_all_derived(self, ideal_prov):
        all_asm = {A1, A2, A3, A4}
        result = flagged_nodes(ideal_prov, all_asm)
        derived = {"D1_momentum_transfer", "D2_collision_frequency",
                   "D3_mean_kinetic_energy", "D4_single_particle_pressure",
                   "D5_pressure_ideal", "D6_ideal_gas_law"}
        assert derived.issubset(result.keys())

    def test_flag_all_assumptions_excludes_observables(self, ideal_prov):
        result = flagged_nodes(ideal_prov, {A1, A2, A3, A4})
        for obs_id in ("obs_P", "obs_V", "obs_T", "obs_n"):
            assert obs_id not in result

    # ---- Works on tiny custom graph too ----

    def test_flagged_nodes_on_minimal_graph(self):
        g = _make_graph(
            ("asm", "A"),
            ("asm", "B"),
            ("der", "D1"),
            ("der", "D2"),
            ("der", "D3"),
            ("edge", "E1", ["A"], "D1"),
            ("edge", "E2", ["B"], "D2"),
            ("edge", "E3", ["D1", "D2"], "D3"),
        )
        prov = compute_provenance(g)
        # Flag A only: D2 (B-branch) should not appear; D1 and D3 should
        result = flagged_nodes(prov, {"A"})
        assert "D1" in result
        assert "D3" in result
        assert "D2" not in result
        assert "B" not in result

    def test_flagged_nodes_returns_records_not_just_ids(self):
        g = _make_graph(("asm", "A"), ("der", "D"), ("edge", "E1", ["A"], "D"))
        prov = compute_provenance(g)
        result = flagged_nodes(prov, {"A"})
        assert isinstance(result["D"], ProvenanceRecord)
        assert isinstance(result["A"], ProvenanceRecord)
