# 🚚 RoutingAGI: Neural Optimization for Middle-Mile Logistics  
### Graphormer + Mamba + Constraint-Aware MoE for Large-Scale Routing

RoutingAGI is a research framework exploring **neural surrogates** for middle-mile routing optimization.  
Inspired by real operational challenges in Amazon Transportation (SCOT, ATS, Middle Mile Science), the system blends:

- graph-aware spatial encoders  
- selective state-space sequence models  
- constraint-aligned expert modules  
- learned world-model forecasting  

to approximate MILP-like routing behavior in a differentiable, scalable way.

💡 The goal is to understand whether deep learning can capture the **structure, constraints, and multi-objective trade-offs** fundamental to real logistics networks.

> This project is **research-oriented** — intended for experimentation, analysis, and hybrid optimization workflows.

---

## 🔍 Problem Motivation

Middle-mile routing (FC → SC → DS → Hubs) must respect:

- trailer & container capacity  
- time windows, cutoffs, SLAs  
- lane legality & mode constraints  
- sort-center throughput limits  
- region-level routing policies  
- load balancing  
- trailer & equipment availability  
- multi-objective cost structures  

Traditional MILP optimizers struggle with:

- highly dynamic networks  
- real-time routing decisions  
- multi-step forecasting  
- non-linear congestion effects  
- multi-region combinatorial complexity  

RoutingAGI investigates whether neural models can become **fast differentiable surrogates** capable of:

- estimating feasibility or constraint violations  
- compressing combinatorial decision spaces  
- forecasting routing state evolution  
- supporting RL, heuristics, or hybrid search  
- learning real routing patterns from data  

---

## 🧠 Architecture Overview

```
Graphormer Encoder → Mamba Block → Constraint-Aware MoE → World Model → Decoder
```

---

### **1. Graphormer Encoder**
Learns spatial and structural routing information, including:

- facility type embeddings (FC, SC, DS, Hubs)  
- lane legality and mode attributes  
- shortest-path encoding  
- structural centrality and connectivity  
- congestion and throughput signals  

---

### **2. Mamba Block (Selective State-Space Model)**
Handles temporal interactions such as:

- SLA propagation  
- congestion ripple effects  
- equipment availability drift  
- scheduling dependencies  

Mamba’s **dynamic filters + selective gating** allow efficient long-range operational reasoning.

---

### **3. Constraint-Aware MoE Layer**
Each expert corresponds to a MILP constraint, using activation functions aligned with its mathematical shape:

| Constraint | Activation | Rationale |
|-----------|------------|-----------|
| Capacity | ReLU / Softplus | hinge-like overload penalty |
| Time Windows | ReLU | lateness hinge |
| Lane Legality | Sigmoid | binary feasibility |
| Throughput | Tanh / Sigmoid | saturating bottlenecks |
| SLA Risk | Softplus | convex exponential penalty |
| Flow Conservation | Linear | equality constraint |
| Region Rules | Softmax | categorical transitions |
| Load Balancing | Softplus | convex overload |
| Trailer Availability | ReLU | piecewise shortage |

This injects **constraint geometry** directly into the model.

---

## 📦 Repository Structure

```
routing_agi/
│
├── modules/
│   ├── graph_encoder.py
│   ├── mamba_block.py
│   ├── constraint_moe.py
│   ├── world_model.py
│   ├── decoder_block.py
│   └── routing_agi_model.py
│
├── data/
│   ├── dataset_builder.py
│   ├── collator.py
│   └── milp_targets.py
│
├── training/
│   ├── train_loop.py
│   ├── optimizer.py
│   └── evaluation.py
│
├── config/
│   ├── model_config.py
│   └── constraint_registry.py
│
├── RoutingAGI_Training.ipynb
└── README.md
```

---

## 🚀 Quickstart (Google Colab)

Use the included notebook:

```
RoutingAGI_Training.ipynb
```

It provides:

- GitHub cloning  
- dataset & dataloader creation  
- model assembly  
- training + evaluation  
- optional HuggingFace upload  

---

## 🎯 Research Questions

RoutingAGI explores:

- Can neural networks approximate MILP constraint geometry?  
- Do constraint-aligned experts improve feasibility prediction?  
- Can world models capture multi-step operational drift?  
- Are Graphormer + Mamba hybrids effective for routing?  
- Can differentiable surrogates accelerate routing search or RL?  
- What does “neural routing intelligence” look like at scale?  

---

## 🧩 Dependencies

- PyTorch  
- NetworkX  
- tqdm  
- huggingface_hub  
- lion-pytorch *(optional)*  

---

## 📄 License

MIT License.

---

## ✨ Author

**Ritwika Kancharla**  
Applied Scientist — Neural Optimization & Routing Models
