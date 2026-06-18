"""Linear elasticity axiom graph.

Five physical assumptions underpin continuum linear elasticity:

    A1_small_strain   — max(ε) < 0.01
    A2_linearity      — σ_vm tracks E·ε within 10%
    A3_isotropic      — directional moduli agree within 5%
    A4_homogeneous    — spatial E variation (CV) < 10%
    A5_quasi_static   — inertia / stress-divergence < 0.01

Four derived quantities with different provenance footprints:

    D1 stress_field   ← A1, A2, A3, A4
    D2 displacement   ← A1, A2, A3, A4
    D3 strain_energy  ← A1, A2          (no A3/A4/A5)
    D4 frequencies    ← A1, A2, A3, A4, A5

D3's footprint excludes A5 — the key result for Scenario C, where A5 fires
but D3 remains trusted.
"""

from __future__ import annotations

from axiom_graph.edges import DerivationEdge
from axiom_graph.graph import AxiomGraph
from axiom_graph.nodes import Node


def build_le_graph() -> AxiomGraph:
    """Construct and return the linear elasticity axiom DAG."""
    g = AxiomGraph()

    # ---- Assumption nodes ------------------------------------------------
    g.add_node(Node(
        id="A1_small_strain", kind="assumption",
        label="Small strain (A1)",
        description=(
            "Equivalent strain ε_eq remains below 0.01. "
            "Violated in large-deformation regimes where geometric nonlinearity "
            "and finite-strain kinematics must be used."
        ),
    ))
    g.add_node(Node(
        id="A2_linearity", kind="assumption",
        label="Linearity (A2)",
        description=(
            "Von Mises stress σ_vm stays within 10% of E·ε_eq. "
            "Violated past the yield point where elasto-plastic or "
            "nonlinear constitutive laws apply."
        ),
    ))
    g.add_node(Node(
        id="A3_isotropic", kind="assumption",
        label="Isotropic (A3)",
        description=(
            "Directional Young's moduli E_11 and E_22 agree within 5%. "
            "Violated for fibre-reinforced composites and single crystals "
            "with strong crystallographic texture."
        ),
    ))
    g.add_node(Node(
        id="A4_homogeneous", kind="assumption",
        label="Homogeneous (A4)",
        description=(
            "Coefficient of variation of spatial E is below 10%. "
            "Violated in graded or porous materials where property "
            "gradients invalidate a single-modulus constitutive law."
        ),
    ))
    g.add_node(Node(
        id="A5_quasi_static", kind="assumption",
        label="Quasi-static (A5)",
        description=(
            "Inertia-to-stress-divergence ratio (ρ·(2πf)²·L²/E) is below 0.01. "
            "Violated at high loading frequencies where wave propagation and "
            "dynamic amplification effects dominate."
        ),
    ))

    # ---- Derived nodes ---------------------------------------------------
    g.add_node(Node(
        id="D1_stress_field", kind="derived",
        label="Stress field sigma(x)",
        description=(
            "Full 3-D stress tensor field. Requires small-strain kinematics (A1), "
            "linear constitutive law (A2), isotropic material (A3), and "
            "homogeneous properties (A4)."
        ),
    ))
    g.add_node(Node(
        id="D2_displacement", kind="derived",
        label="Displacement field u(x)",
        description=(
            "Full 3-D displacement field obtained by integrating the strain-"
            "displacement equations. Same dependence as D1: A1, A2, A3, A4."
        ),
    ))
    g.add_node(Node(
        id="D3_strain_energy", kind="derived",
        label="Strain energy U",
        description=(
            "Total elastic strain energy U = ½ σ:ε integrated over the volume. "
            "Depends only on A1 (small strain) and A2 (linear constitutive law); "
            "isotropy and homogeneity are not required for a scalar energy bound."
        ),
    ))
    g.add_node(Node(
        id="D4_frequencies", kind="derived",
        label="Natural frequencies omega_n",
        description=(
            "Eigenfrequencies of the structure from the linear elastodynamic "
            "eigenvalue problem. Requires all five assumptions: A1–A4 for the "
            "stiffness matrix, A5 for the quasi-static (low-frequency) regime "
            "in which the linearised equations of motion are valid."
        ),
    ))

    # ---- Edges -----------------------------------------------------------
    g.add_edge(DerivationEdge(
        id="LE1",
        premise_ids=["A1_small_strain", "A2_linearity", "A3_isotropic", "A4_homogeneous"],
        conclusion_id="D1_stress_field",
        rule_label="A1+A2+A3+A4 → σ(x)",
        description=(
            "Small-strain kinematics (A1) gives ε = ½(∇u + ∇uᵀ). "
            "Linear constitutive law (A2), isotropy (A3), and homogeneity (A4) "
            "together give σ = C:ε with a constant isotropic stiffness tensor C."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="LE2",
        premise_ids=["A1_small_strain", "A2_linearity", "A3_isotropic", "A4_homogeneous"],
        conclusion_id="D2_displacement",
        rule_label="A1+A2+A3+A4 → u(x)",
        description=(
            "The displacement field is obtained from the equilibrium BVP "
            "∇·σ + b = 0 with σ = C:ε. Same assumptions as D1 are required."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="LE3",
        premise_ids=["A1_small_strain", "A2_linearity"],
        conclusion_id="D3_strain_energy",
        rule_label="A1+A2 → U = ½σ:ε",
        description=(
            "Strain energy U = ½ ∫ σ:ε dV requires only that strain be small "
            "(A1) and the stress-strain relation be linear (A2). "
            "Isotropy and homogeneity are not needed for the scalar integral."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="LE4",
        premise_ids=[
            "A1_small_strain", "A2_linearity", "A3_isotropic",
            "A4_homogeneous", "A5_quasi_static",
        ],
        conclusion_id="D4_frequencies",
        rule_label="A1+A2+A3+A4+A5 → ω_n",
        description=(
            "Natural frequencies come from det(K - ω²M) = 0. "
            "K is valid under A1–A4 (linear elastic stiffness matrix). "
            "A5 (quasi-static) ensures the linearised equations of motion "
            "hold and inertia terms do not dominate the static response."
        ),
    ))

    return g


if __name__ == "__main__":
    g = build_le_graph()

    # 1. Print every node with its kind
    print("=== Nodes ===")
    for node in g.nodes.values():
        print(f"  [{node.kind:10s}]  {node.id}  --  {node.label}")

    # 2. Print provenance footprints for derived nodes
    print("\n=== Provenance footprints (DerivedNodes) ===")
    derived_nodes = [n for n in g.nodes.values() if n.kind == "derived"]
    footprints: dict[str, frozenset[str]] = {}
    for node in derived_nodes:
        fp = g.ancestor_assumptions(node.id)
        footprints[node.id] = fp
        names = sorted(fp)
        print(f"  {node.id}: {names}")

    # 3. Assertions
    expected_d3 = frozenset({"A1_small_strain", "A2_linearity"})
    expected_d4 = frozenset({
        "A1_small_strain", "A2_linearity", "A3_isotropic",
        "A4_homogeneous", "A5_quasi_static",
    })

    assert footprints["D3_strain_energy"] == expected_d3, (
        f"D3 footprint mismatch!\n"
        f"  expected: {sorted(expected_d3)}\n"
        f"  got:      {sorted(footprints['D3_strain_energy'])}"
    )
    assert footprints["D4_frequencies"] == expected_d4, (
        f"D4 footprint mismatch!\n"
        f"  expected: {sorted(expected_d4)}\n"
        f"  got:      {sorted(footprints['D4_frequencies'])}"
    )

    print("\nAll assertions passed.")
