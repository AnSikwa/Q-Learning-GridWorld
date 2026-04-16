# CS 57200 Assignment 6 — Part 3: Convergence Analysis

## Question 1 — Does the Learned Policy Avoid the Pit? (10 pts)

After training for 5,000 episodes with epsilon = 0.1, the learned policy grid is:

```
| →  →  →  G |
| ↑  #  ↑  P |
| ↑  →  ↑  ↓ |
```

Starting at (0,0), the agent follows the policy arrows: (0,0) to the right to (0,1), then right to (0,2), and right again to (0,3), reaching the goal. The agent avoids the pit entirely. The cells next to the pit at (1,3), specifically (1,2) and (2,3), take actions that move away from the pit: UP at (1,2) and DOWN at (2,3). This shows the agent has learned to avoid the dangerous area. The policy keeps the agent on the top row, which is the safest route because it stays as far from the pit as possible while heading straight to the goal

## Question 2 — Q-Values at State (0, 2) (15 pts)

The Q-values at state (0, 2) after training with epsilon = 0.1 for 5,000 episodes:

```
Q((0,2), UP)    = 0.7936
Q((0,2), DOWN)  = 0.6265
Q((0,2), LEFT)  = 0.6670
Q((0,2), RIGHT) = 0.9339
```

**Which action has the highest Q-value? Does this match the optimal action?**
RIGHT has the highest Q-value (0.9339), which matches the optimal action. The state (0,2) is one step left of the goal at (0,3), so moving RIGHT takes the agent directly to the +1.0 goal reward. This is what we would expect from a well-trained agent.

**Compare the Q-values for RIGHT (toward goal) and DOWN (toward pit). Is there a large gap? Explain why.**
There is a clear gap: RIGHT has a Q-value of 0.9339, while DOWN is 0.6265, so the difference is about 0.31. RIGHT moves the agent toward the goal and the +1.0 reward. DOWN takes the agent to (1,2), which is next to the pit at (1,3). From (1,2), the agent is more likely to fall into the pit because of random slips. The lower Q-value for DOWN shows the risk of being close to the pit. Even if the agent eventually reaches the goal from (1,2), the path is longer, so there are more -0.04 step penalties, and it is riskier because there is a higher chance of slipping into the pit and getting -1.0. The gap in Q-values shows both the extra step cost and the risk of a negative reward from being near the pit.

**How does the stochastic transition model (80/10/10) affect your answers?**
The 80/10/10 stochastic model means that even the best action, RIGHT, has a 10% chance of slipping DOWN into (1,2) near the pit and a 10% chance of slipping UP, which just sends the agent back to (0,2). This is why RIGHT’s Q-value is 0.9339 instead of closer to 1.0. The 20% chance of moving in the wrong direction lowers the expected return. Similarly, DOWN’s Q-value (0.6265) is not extremely low because the agent only has a 10% chance of slipping RIGHT into the pit from (1,2). Most of the time (80%), it moves down to (2,2), which is safer. These random transitions make the Q-values less extreme, so the differences between actions are smaller than they would be if the world were deterministic.

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

***Does the agent still learn the optimal policy? Why or why not?***
The agent does not always learn the best policy in every state. It might find a way to the goal along the top row, since random slips sometimes help it explore by accident. However, in cells far from this main path, its choices are often not the best or seem random. For instance, in cells like (2,1), (2,2), and (2,3), the agent often picks the wrong action, such as LEFT, which moves it away from the goal. This happens because the greedy agent sticks with the first action it tries in those states and never tries other options. Since it only updates Q-values for actions it actually takes, if it finds a suboptimal action early that gives a slightly positive Q-value, it will keep choosing that action and never find out if another action is better.

**What happens to the Q-values of actions the agent never tries?**__
Actions the agent never tries keep their default value of 0.0 forever. For example, at (0,2), the Q-values for UP, DOWN, and LEFT stayed at 0.0 or close to it because the greedy agent found RIGHT early, thanks to random slips that took it to the goal. After that, it always chose RIGHT and never tried the other actions. These untried Q-values are basically frozen, since the agent has no information about them and cannot improve its estimates without actually trying those actions.

**In one sentence: what is the lesson about exploration in reinforcement learning?**
If a reinforcement learning agent does not explore, it can get stuck, always choosing actions that are not the best, because it never gets the experience needed to find better options. Exploration is necessary for the agent to eventually learn the true optimal policy.


