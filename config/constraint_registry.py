"""
Constraint Registry for RoutingAGI
----------------------------------

This file defines ALL Amazon Middle-Mile constraint families:
capacity, time windows, lane legality, hub throughput, SLA,
flow conservation, mode constraints, regional rules,
load balancing, trailer availability.

Each entry defines:
- loss_type: mse | bce | ce
- output_dim: number of regression/class outputs
- activation: activation inside GenericExpert trunk
- target_fn_name: name of target generator in milp_targets.py
"""

from data.milp_targets import (
    capacity_target_fn,
    time_window_target_fn,
    lane_legality_target_fn,
    hub_throughput_target_fn,
    sla_risk_target_fn,
    flow_conservation_target_fn,
    mode_constraints_target_fn,
    region_rules_target_fn,
    load_balancing_target_fn,
    trailer_availability_target_fn,
)

CONSTRAINTS = {

    # ------------------------------------------------------------
    # 1. Trailer Capacity Constraint
    # ------------------------------------------------------------
    "capacity": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "softplus",        # hinge-like
        "target_fn": capacity_target_fn,
        "description": "Trailer capacity violation: max(0, load - cap)."
    },

    # ------------------------------------------------------------
    # 2. Time Window / Cutoff / Departure Time Constraint
    # ------------------------------------------------------------
    "time_window": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "softplus",
        "target_fn": time_window_target_fn,
        "description": "Time-window lateness penalty."
    },

    # ------------------------------------------------------------
    # 3. Lane Legality (DS→FC forbidden edges etc.)
    # ------------------------------------------------------------
    "lane_legality": {
        "loss_type": "bce",
        "output_dim": 1,                 # BCE logits
        "activation": "linear",          # no squash
        "target_fn": lane_legality_target_fn,
        "description": "Binary legality for edges (illegal edge → 1)."
    },

    # ------------------------------------------------------------
    # 4. Hub Throughput / Sort Center Congestion
    # ------------------------------------------------------------
    "hub_throughput": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "tanh",            # saturating bottleneck
        "target_fn": hub_throughput_target_fn,
        "description": "Hub capacity usage / congestion ratio."
    },

    # ------------------------------------------------------------
    # 5. SLA / Delivery Deadline Risk
    # ------------------------------------------------------------
    "sla_risk": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "softplus",        # exponential-ish
        "target_fn": sla_risk_target_fn,
        "description": "SLA risk when near or past deadline."
    },

    # ------------------------------------------------------------
    # 6. Flow Conservation (inflow=outflow)
    # ------------------------------------------------------------
    "flow_conservation": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "linear",          # pure linear eq violation
        "target_fn": flow_conservation_target_fn,
        "description": "Flow conservation absolute violation."
    },

    # ------------------------------------------------------------
    # 7. Mode Constraints (Air vs Ground constraints)
    # ------------------------------------------------------------
    "mode_constraints": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "softplus",        # hinge-like penalty
        "target_fn": mode_constraints_target_fn,
        "description": "Mode-specific constraints (air legality, penalties)."
    },

    # ------------------------------------------------------------
    # 8. Regional Routing Rules (multiclass)
    # ------------------------------------------------------------
    "region_rules": {
        "loss_type": "ce",
        "output_dim": 8,                 # region classes e.g. 8 zones
        "activation": "linear",
        "target_fn": region_rules_target_fn,
        "description": "Multi-class region transition feasibility."
    },

    # ------------------------------------------------------------
    # 9. Sortation Load Balancing / Hub Load Leveling
    # ------------------------------------------------------------
    "load_balancing": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "softplus",
        "target_fn": load_balancing_target_fn,
        "description": "Convex load imbalance penalty."
    },

    # ------------------------------------------------------------
    # 10. Trailer Availability / Shortage Constraints
    # ------------------------------------------------------------
    "trailer_availability": {
        "loss_type": "mse",
        "output_dim": 1,
        "activation": "relu",
        "target_fn": trailer_availability_target_fn,
        "description": "Trailer shortage penalty: max(0, used - max)."
    },
}


def list_constraints():
    """Utility for debugging / printing all constraints."""
    return list(CONSTRAINTS.keys())
