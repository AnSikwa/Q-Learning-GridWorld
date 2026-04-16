# Q-Learning in Grid World

**CS 57200: Heuristic Problem Solving — Assignment 6**

## Overview

A tabular Q-Learning agent that learns to navigate a stochastic 3×4 Grid World, reaching a goal while avoiding a pit. The environment uses an 80/10/10 transition model — each action succeeds 80% of the time, with a 10% chance of slipping perpendicular in either direction.

```
| →  →  →  G |
| ↑  #  ↑  P |
| ↑  →  ↑  ↓ |
```

## Key Concepts

- **Epsilon-Greedy Action Selection** — balances exploration (random actions) with exploitation (best-known action) using ε = 0.1
- **Q-Learning Update Rule** — `Q(s,a) ← Q(s,a) + α·(r + γ·max Q(s',a') − Q(s,a))`
- **Stochastic Transitions** — 80% intended direction, 10% slip left, 10% slip right
- **Environment Validation** — defensive checks on grid configuration before training

## Grid Layout

| Cell | Description |
|------|-------------|
| (0,0) | Start state |
| (0,3) | Goal (+1.0 reward, terminal) |
| (1,3) | Pit (−1.0 reward, terminal) |
| (1,1) | Blocked obstacle |
| All others | −0.04 step cost |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 0.1 | Learning rate |
| γ (gamma) | 0.9 | Discount factor |
| ε (epsilon) | 0.1 | Exploration rate |
| Episodes | 5,000 | Training episodes |

## Files

- `GridWorld_Starter.py` — Complete Q-learning implementation (GridWorld environment + QLearner agent)
- `answers.md` — Convergence analysis: policy evaluation, Q-value analysis at state (0,2), and exploration vs. exploitation experiment

## Running

```bash
python GridWorld_Starter.py
```

Output includes the validation check, learned policy grid, and Q-values for state (0,2).
