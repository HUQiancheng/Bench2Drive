# What is Bench2Drive?

## Core Identity

**Bench2Drive = Closed-Loop Benchmark + Standardized Training Dataset**

It is designed for **fair algorithm-level comparison** of end-to-end autonomous driving methods.

## The Two Components

| Component | What It Is | Purpose |
|-----------|------------|---------|
| **Benchmark** | 220 routes with 44 interactive scenarios in CARLA | Closed-loop evaluation |
| **Dataset** | 10000+ clips from Think2Drive RL expert | Standardized training for fair comparison |

## Why the Dataset Exists

The key insight from the paper:

> "Bench2Drive consists of a standardized training set... ensuring that all AD systems are trained under **abundant yet similar conditions**, which is crucial for **fair algorithm-level comparisons**."

The dataset is NOT meant to produce SOTA models from scratch. It exists so that:
- **All methods train on the same data** → Fair comparison
- **Results are reproducible** → Scientific rigor
- **Algorithm differences are isolated** → Not confounded by data differences

## Dataset Sizes

| Split | Clips | Size | Purpose |
|-------|-------|------|---------|
| Mini | 10 | 4 GB | Quick sanity check |
| Base | 1000 | 400 GB | Development/ablation |
| Full | 13638 | 4 TB | Full training for fair comparison |

## Comparison with Other Benchmarks

| Benchmark | Has Dataset? | Evaluation Type | Fair Comparison? |
|-----------|--------------|-----------------|------------------|
| CARLA Leaderboard | No | Closed-loop | No (train anywhere) |
| nuScenes | Yes | Open-loop | Yes (but open-loop) |
| **Bench2Drive** | **Yes** | **Closed-loop** | **Yes** |

## The Training Reality

For **fair Bench2Drive comparison**:
- Train on Bench2Drive dataset (Base or Full)
- Evaluate on 220 routes
- Report standardized metrics

For **pushing SOTA** (separate goal):
- Pre-train on large external datasets (nuScenes, internal data)
- Fine-tune or adapt to CARLA
- May not be directly comparable to Bench2Drive-trained models

## Summary

Bench2Drive solves the problem: *"How do we fairly compare E2E driving algorithms when everyone trains on different data?"*

Answer: Provide standardized training data + closed-loop evaluation = fair algorithm comparison.
