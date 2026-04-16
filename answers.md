# CS 57200 Assignment 6 — Part 3: Convergence Analysis

## Question 1 — Does the Learned Policy Avoid the Pit? (10 pts)

After training for 5,000 episodes with epsilon = 0.1, the learned policy grid is:

```
| →  →  →  G |
| ↑  #  ↑  P |
| ↑  →  ↑  ↓ |
```

Tracing the path from start (0,0) following the policy arrows: (0,0)→RIGHT→(0,1)→RIGHT→(0,2)→RIGHT→(0,3)=Goal. The agent successfully reaches the goal without entering the pit. Critically, the cells adjacent to the pit at (1,3) — namely (1,2) and (2,3) — choose actions that move away from the pit (UP at (1,2) and DOWN at (2,3)), which demonstrates that the agent has learned to avoid the dangerous region. The policy routes the agent along the top row, which is the safest path because it keeps maximum distance from the pit while heading directly toward the goal.

## Question 2 — Q-Values at State (0, 2) (15 pts)

The Q-values at state (0, 2) after training with epsilon = 0.1 for 5,000 episodes:

```
Q((0,2), UP)    = 0.7936
Q((0,2), DOWN)  = 0.6265
Q((0,2), LEFT)  = 0.6670
Q((0,2), RIGHT) = 0.9339
```

**Which action has the highest Q-value? Does this match the optimal action?**
RIGHT has the highest Q-value (0.9339), which matches the optimal action. State (0,2) is one step left of the goal at (0,3), so moving RIGHT leads directly to the +1.0 goal reward. This is exactly what we would expect from a well-trained agent.

**Compare the Q-values for RIGHT (toward goal) and DOWN (toward pit). Is there a large gap? Explain why.**
There is a substantial gap: RIGHT = 0.9339 vs. DOWN = 0.6265, a difference of about 0.31. RIGHT leads toward the goal (+1.0 reward), while DOWN moves the agent to state (1,2), which is directly adjacent to the pit at (1,3). From (1,2), the agent faces increased risk of accidentally falling into the pit due to stochastic slips. The lower Q-value for DOWN reflects the danger of being near the pit: even if the agent eventually reaches the goal from (1,2), the path is longer (incurring more -0.04 step penalties) and riskier (higher chance of slipping into the pit for -1.0). The gap captures both the additional step cost and the expected negative reward from pit proximity.

**How does the stochastic transition model (80/10/10) affect your answers?**
The 80/10/10 stochastic model means that even the optimal action RIGHT carries a 10% chance of slipping DOWN into (1,2) (near the pit) and a 10% chance of slipping UP (bouncing off the wall back to (0,2)). This is why RIGHT's Q-value is 0.9339 rather than a value closer to 1.0 — the 20% chance of unintended movement reduces the expected return. Similarly, DOWN's Q-value (0.6265) is not catastrophically low because the agent only has a 10% chance of slipping RIGHT into the pit from (1,2); 80% of the time it actually moves down to (2,2), which is safer. The stochastic transitions smooth out the Q-values, making the differences between actions less extreme than they would be in a deterministic world.

## Question 3 — The Role of Exploration (15 pts)

After retraining with epsilon = 0.0 (purely greedy, no exploration) for 5,000 episodes, a typical result:

```
| →  →  →  G |
| ↑  #  ←  P |
| ↑  ←  ←  ← |
```

Q-values at (0,2) with epsilon = 0.0:
```
Q((0,2), UP)    = 0.0000
Q((0,2), DOWN)  = 0.0000
Q((0,2), LEFT)  = 0.0000
Q((0,2), RIGHT) = 0.9271
```

State-action pairs visited: ~33–36 out of 48 total.

**Does the agent still learn the optimal policy? Why or why not?**
The agent does not reliably learn the optimal policy across all states. While it may find a path to the goal along the top row (because stochastic slips provide some accidental exploration), the policy in cells far from the main path is often suboptimal or arbitrary. For example, cells like (2,1), (2,2), and (2,3) frequently show incorrect actions (e.g., LEFT pointing away from the goal) because the greedy agent gets locked into the first action it tried from those states and never explores alternatives. The agent only updates Q-values for actions it actually takes, so if it stumbles into an early suboptimal action that happens to yield a slightly positive Q-value, it will exploit that action forever and never discover that a different action would be better.

**What happens to the Q-values of actions the agent never tries?**
Actions the agent never tries remain at their default value of 0.0 indefinitely. As seen in the Q-values at (0,2), UP, DOWN, and LEFT all stayed at 0.0 or near 0.0 because the greedy agent found RIGHT early (due to stochastic slips carrying it to the goal) and then always exploited RIGHT without ever deliberately trying the other three actions. These untried Q-values are essentially "frozen" — the agent has no information about them and cannot improve its estimates without actually experiencing those actions.

**In one sentence: what is the lesson about exploration in reinforcement learning?**
Without exploration, a reinforcement learning agent can become permanently trapped exploiting suboptimal actions because it never gathers the experience needed to discover better alternatives — exploration is essential for convergence to the true optimal policy.
