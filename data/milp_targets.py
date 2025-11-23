"""
MILP → Differentiable Surrogate Target Functions
------------------------------------------------

These functions convert classical Middle-Mile MILP constraints
into regression or classification targets that can be learned
by the Constraint-MoE experts.

Every function returns a tensor with shape [1] or [K] for
K-class classification.

You will map these into [B, T, ...] shapes in your Dataset.
"""

import torch
import torch.nn.functional as F


# ============================================================
# 1. TRAILER CAPACITY
# load_a = Σ_o w_o * x_{o,a}
# violation = max(0, load_a - Cap_a) / Cap_a
# ============================================================

def capacity_target_fn(weights, x_oa, cap_a):
    """
    weights: [O]          item weights
    x_oa:    [O]          0/1 assignment of item→trailer
    cap_a:   scalar       trailer capacity

    Returns normalized capacity violation [1].
    """
    load = (weights * x_oa).sum()
    violation = torch.clamp(load - cap_a, min=0.0)
    return (violation / cap_a).unsqueeze(0)   # [1]


# ============================================================
# 2. TIME WINDOW / CUTOFF LATE PENALTY
# lateness = max(0, t_i - L_i) / (L_i - E_i)
# ============================================================

def time_window_target_fn(t_i, L_i, E_i):
    """
    t_i : arrival/dispatch time
    L_i : latest allowed
    E_i : earliest allowed

    Returns normalized lateness [1].
    """
    late = torch.clamp(t_i - L_i, min=0.0)
    denom = torch.clamp(L_i - E_i, min=1e-6)
    return (late / denom).unsqueeze(0)        # [1]


# ============================================================
# 3. LANE LEGALITY (binary)
# illegal = 1 if edge chosen AND edge illegal
# ============================================================

def lane_legality_target_fn(x_edge, allowed_flag):
    """
    x_edge:       0/1 whether this edge was chosen
    allowed_flag: 1 if legal, 0 if illegal

    Returns [1] = 1 if (x_edge=1 and illegal), else 0.
    """
    v = 1.0 if (x_edge == 1 and allowed_flag == 0) else 0.0
    return torch.tensor([v], dtype=torch.float32)


# ============================================================
# 4. HUB THROUGHPUT / CONGESTION
# congestion = flow_in / cap_h
# ============================================================

def hub_throughput_target_fn(flow_in, cap_h):
    """
    flow_in: [K] inbound flows
    cap_h : scalar hub capacity

    Returns congestion ratio [1].
    """
    total = flow_in.sum()
    return torch.tensor([total / cap_h], dtype=torch.float32)


# ============================================================
# 5. SLA RISK (convex)
# slack = deadline - delivery_time
# risk = softplus(-slack / max_slack)
# ============================================================

def sla_risk_target_fn(t_delivery, deadline, max_slack):
    """
    t_delivery: actual delivery/arrival time
    deadline:   SLA cutoff
    max_slack:  normalizing constant

    Returns SLA lateness risk [1].
    """
    slack = (deadline - t_delivery) / max_slack
    risk = F.softplus(-slack)          # higher when slack < 0
    return risk.unsqueeze(0)           # [1]


# ============================================================
# 6. FLOW CONSERVATION
# violation = |Σ_in - Σ_out|
# ============================================================

def flow_conservation_target_fn(flow_in, flow_out):
    """
    flow_in:  [M]
    flow_out: [N]

    Returns absolute flow conservation violation [1].
    """
    v = torch.abs(flow_in.sum() - flow_out.sum())
    return v.unsqueeze(0)


# ============================================================
# 7. MODE CONSTRAINTS (Air vs Ground)
# if air chosen & not allowed → penalty
# ============================================================

def mode_constraints_target_fn(air_allowed, chosen_mode, air_penalty=1.0):
    """
    air_allowed: 0/1
    chosen_mode: "air" or "ground"
    air_penalty: scalar penalty for illegal air usage

    Returns [1] penalty value.
    """
    if chosen_mode == "air" and air_allowed == 0:
        v = air_penalty
    else:
        v = 0.0
    return torch.tensor([v], dtype=torch.float32)


# ============================================================
# 8. REGIONAL ROUTING RULES (multiclass)
# We return a class index. You choose:
#   0 = illegal
#   1 = allowed_path_1
#   2 = allowed_path_2
#   ...
# ============================================================

def region_rules_target_fn(region_from, region_to, legal_matrix):
    """
    region_from:  index 0..R-1
    region_to:    index 0..R-1
    legal_matrix: [R, R] of allowed transitions

    Output:
      A class index for cross-entropy loss.

    Example:
      if legal_matrix[from][to] == 1 → class 1
      else → class 0 (illegal)
    """
    is_legal = legal_matrix[region_from, region_to].item()
    cls = 1 if is_legal == 1 else 0      # you can expand to >2 classes
    return torch.tensor([cls], dtype=torch.long)


# ============================================================
# 9. LOAD BALANCING (convex)
# imbalance = mean((load_i / mean_load - 1)^2)
# ============================================================

def load_balancing_target_fn(hub_loads, target_ratio=1.0):
    """
    hub_loads: [H]
    target_ratio: desired mean (usually 1)

    Returns convex penalty [1].
    """
    mean_load = torch.clamp(hub_loads.mean(), min=1e-6)
    ratios = hub_loads / mean_load
    penalty = ((ratios - target_ratio) ** 2).mean()
    return penalty.unsqueeze(0)


# ============================================================
# 10. TRAILER AVAILABILITY SHORTAGE
# shortage = max(0, used_trailers - max_trailers)
# ============================================================

def trailer_availability_target_fn(used_trailers, max_trailers):
    """
    used_trailers: scalar
    max_trailers: scalar

    Returns shortage [1].
    """
    diff = torch.clamp(used_trailers - max_trailers, min=0.0)
    return diff.unsqueeze(0)
