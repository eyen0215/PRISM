"""
Regime boundary definitions and state classification utilities.

Defines the pressure boundaries separating the training regime (ideal gas
valid) from the held-out regime (van der Waals corrections dominate), and
provides helpers to:
    - classify a given (P, V, T, n) state into its regime
    - compute the fractional van der Waals correction to pressure
      ΔP/P_ideal = [nRT/(V-nb) - an²/V² - nRT/V] / (nRT/V)
      which quantifies how wrong PV=nRT is at a given state

These utilities are used at both data-generation time (data/generate.py)
and inference time (experiments/pilot.py) to route states to the correct
evaluation split without leaking held-out information into training.
"""
