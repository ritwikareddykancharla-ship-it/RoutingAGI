# RoutingAGI: Differentiable Middle-Mile Optimization  
A Hybrid Graphormer + Mamba + Constraint-MoE Architecture for Neural Routing

RoutingAGI is a research framework exploring neural surrogates for large-scale middle-mile routing optimization, inspired by Amazon Transportation (SCOT, ATS, Middle Mile).  
It blends graph neural encoders, selective state-space models, expert-level constraint modules, and world-modeling techniques to approximate the behavior of MILP-based routing systems.

---

## 🔍 Overview

Traditional middle-mile routing relies heavily on mixed-integer linear programming (MILP).  
While exact solvers are powerful, they become slow or brittle under:

- large dynamic networks  
- real-time constraints  
- multi-objective cost surfaces  
- operational uncertainty  
- multi-region interactions  

RoutingAGI investigates whether modern deep learning architectures can:

- learn MILP constraint structure,  
- forecast routing feasibility,  
- compress combinatorial search spaces,  
- evaluate candidate decisions faster than solving full MILPs.

This project is **research-oriented**, not production routing.

---

## 🧠 Architecture Summary

Graphormer Encoder → Mamba Block → Constraint-MoE → World Model → Decoder


### **1. Graphormer Encoder**
Learns the spatial structure of Amazon middle-mile networks:

- node types (FC, SC, DS, air hubs)  
- lane types & legality  
- shortest-path distances  
- congestion & throughput features  
- centrality and structural position  

### **2. Mamba Block (Selective State-Space Model)**
Models long-range dependencies:

- time windows  
- SLA propagation  
- congestion ripple effects  
- temporal stability  

Mamba provides linear-time sequence modeling with dynamic gating.

### **3. Constraint-MoE Layer**
Key innovation: each MILP constraint family maps to an expert with matching activation geometry.

| Constraint | Expert Activation | Rationale |
|-----------|-------------------|-----------|
| Capacity | ReLU / Softplus | hinge-shaped overload penalty |
| Time Windows | ReLU | lateness hinge |
| Lane Legality | Sigmoid | binary feasibility |
| Throughput | Tanh / Sigmoid | saturating bottlenecks |
| SLA Risk | Softplus | convex penalty curve |
| Flow Conservation | Linear | equality constraint |
| Region Rules | Softmax | categorical transitions |
| Load Balancing | Softplus | convex overload |
| Trailer Availability | ReLU | piecewise linear |

This injects **mathematical constraint shape** directly into the model, enabling differentiable approximations of MILP structure.

### **4. World Model**
Predicts multi-step evolution of system state:

- congestion  
- trailer shortages  
- SLA slack drift  
- flow stability  

### **5. Decoder**
Produces:

- feasibility scores  
- constraint violation estimations  
- route embeddings  
- policy logits for downstream search

---

## 📦 Repository Structure

