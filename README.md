# 🚚 RoutingAGI: Neural Optimization for Middle-Mile Logistics  
### 🧠 Graphormer + Mamba + Constraint-Aware MoE for Large-Scale Routing

RoutingAGI is a research framework that investigates **neural surrogates** for large-scale middle-mile routing optimization, inspired by real operational challenges in Amazon Transportation (SCOT, ATS, Middle Mile Science).  
It integrates **graph-aware encoders**, **state-space sequence models**, **constraint-aligned experts**, and **world-model forecasting** to approximate MILP-like routing behavior.

💡 The goal is to explore whether deep learning can capture the **structure, constraints, feasibility patterns, and multi-objective trade-offs** present in real logistics networks.

> ⚠️ RoutingAGI is a **research prototype**, intended for experimentation and hybrid optimization—not production deployment.

---

## 🔍 Problem Motivation

Middle-mile routing involves shipping between FC → SC → DS → Hubs while respecting:

- 🚛 trailer & container capacities  
- ⏱️ time windows, cutoffs, SLAs  
- 🛣️ lane legality & mode constraints (air/ground)  
- ⚙️ sort-center throughput & bottlenecks  
- 🌍 region-level routing rules  
- ⚖️ load balancing  
- 🚚 equipment & trailer availability  
- 💰 multi-objective operational costs  

Traditional MILPs struggle under:

- large, dynamic networks  
- real-time decision needs  
- stochastic demand  
- multi-step forecasting  
- non-linear constraints  

RoutingAGI explores whether neural networks can act as **fast, differentiable surrogates** that:

- estimate feasibility and constraint violations  
- compress combinatorial decision spaces  
- support routing search or RL  
- simulate future network states  
- learn operational patterns from data  

---

## 🧠 Architecture Overview

## Graphormer Encoder → Mamba Block → Constraint-Aware MoE → World Model → Decoder


---

### **1. 🗺️ Graphormer Encoder**
Learns structural and spatial routing features:

- facility type embeddings (FC, SC, DS, Hubs)  
- lane types & legality  
- shortest-path and distance encodings  
- centrality, connectivity, and congestion signals  
- multi-hop relational context  

---

### **2. ⚡ Mamba Block (Selective State-Space Model)**
Provides temporal reasoning with linear-time scaling:

- SLA propagation  
- congestion ripple effects  
- equipment availability drift  
- scheduling dependencies  

Mamba’s **selective gating + dynamic filters** make it powerful for operational sequences.

---

### **3. 🧩 Constraint-Aware Mixture-of-Experts**
Each expert models a MILP constraint through activation geometry.

| Constraint | Activation | Why |
|-----------|------------|-----|
| 📦 Capacity | ReLU / Softplus | hinge-shaped overload |
| ⏱️ Time Windows | ReLU | lateness hinge |
| ❌ Lane Legality | Sigmoid | binary feasibility |
| ⚙️ Throughput | Tanh / Sigmoid | saturating bottlenecks |
| 🚨 SLA Risk | Softplus | convex penalty |
| 🔁 Flow Conservation | Linear | equality constraint |
| 🌍 Region Rules | Softmax | categorical transitions |
| ⚖️ Load Balancing | Softplus | convex overload |
| 🚚 Trailer Availability | ReLU | piecewise linear |

This layer injects **mathematical constraint structure** directly into the network.

---

### **4. 🔮 World Model**
Forecasts routing state evolution:

- congestion & queue buildup  
- SLA slack drift  
- trailer shortages  
- sort-center saturation  
- cross-dock propagation  

Enables multi-step simulation & planning.

---

### **5. 🎛️ Decoder**
Outputs routing-relevant predictions:

- constraint violation likelihood  
- feasibility scores  
- route-embedding vectors  
- logits for downstream decision modules  

---

## 📦 Repository Structure
routing_agi/
│
├── modules/
│ ├── graph_encoder.py
│ ├── mamba_block.py
│ ├── constraint_moe.py
│ ├── world_model.py
│ ├── decoder_block.py
│ └── routing_agi_model.py
│
├── data/
│ ├── dataset_builder.py
│ ├── collator.py
│ └── milp_targets.py
│
├── training/
│ ├── train_loop.py
│ ├── optimizer.py
│ └── evaluation.py
│
├── config/
│ ├── model_config.py
│ └── constraint_registry.py
│
├── RoutingAGI_Training.ipynb
└── README.md


---

## 🚀 Quickstart (Google Colab)

Use the included notebook:

RoutingAGI_Training.ipynb


It provides:

- GitHub cloning  
- dataset + dataloader creation  
- model construction  
- training & evaluation  
- HuggingFace upload support  

---

## 🎯 Research Questions

RoutingAGI enables exploration of questions like:

- Can neural models approximate MILP structure through activations?  
- Do constraint-specific experts improve feasibility prediction?  
- Can world models capture multi-step operational drift?  
- How do Graphormer + Mamba hybrids perform in routing environments?  
- Can differentiable models accelerate search or RL planning?  
- Can this help build AGI-grade routing intelligence?

---

## 🧩 Dependencies

- PyTorch  
- NetworkX  
- tqdm  
- huggingface_hub  
- lion-pytorch (optional)

---

## 📄 License

MIT License.

---

## ✨ Author

**Ritwika Kancharla**  
Applied Scientist — Neural Optimization & Routing Models 🚛🧠✨

