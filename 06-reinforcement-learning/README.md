# Reinforcement Learning

**Agent-based learning** — choose actions to maximise cumulative reward over time. Covers the multi-armed bandit problem.

## Algorithms

| Algorithm | Key Idea | File |
|-----------|----------|------|
| Upper Confidence Bound (UCB) | Balance exploration vs exploitation with confidence intervals | ucb.py |
| Thompson Sampling | Bayesian approach — sample from probability distributions | thompson.py |

## Problem: The Multi-Armed Bandit

Imagine 10 different ad versions. Each has an unknown CTR (click-through rate). Which ad should we show to maximise total clicks with as few experiments as possible?

- **Random selection** — wastes impressions exploring bad options
- **UCB / Thompson** — quickly converge to the best ad

## Lab

```bash
cd 06-reinforcement-learning
python ucb.py
python thompson.py
```

## Cross-reference

- [study-plan/README.md](../study-plan/README.md) — Phase 4
