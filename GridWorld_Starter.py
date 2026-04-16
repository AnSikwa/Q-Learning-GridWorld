"""GridWorld_Starter.py
CS 57200: Heuristic Problem Solving

A skeleton implementation of Q-Learning in a stochastic Grid World.
Your task is to implement the Q-Learning update rule and the epsilon-greedy action selection.
"""

import random
import numpy as np


class GridWorld:
    def __init__(self, rows=3, cols=4):
        self.rows = rows
        self.cols = cols
        self.start_state = (0, 0)
        self.goal_state = (0, 3)
        self.pit_state = (1, 3)
        self.blocks = [(1, 1)]
        
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.move_dirs = {
            'UP': (-1, 0), 'DOWN': (1, 0), 
            'LEFT': (0, -1), 'RIGHT': (0, 1)
        }

    def validate_configuration(self):
        """
        Sanity checks for environment setup before training.
        Returns (is_valid, message).
        """
        states_to_check = {
            "start_state": self.start_state,
            "goal_state": self.goal_state,
            "pit_state": self.pit_state,
        }

        for name, state in states_to_check.items():
            if not (isinstance(state, tuple) and len(state) == 2):
                return False, f"{name} must be a (row, col) tuple."
            r, c = state
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return False, f"{name}={state} is outside grid bounds {self.rows}x{self.cols}."

        if self.start_state in self.blocks:
            return False, "start_state cannot be inside a blocked cell."
        if self.goal_state in self.blocks:
            return False, "goal_state cannot be inside a blocked cell."
        if self.pit_state in self.blocks:
            return False, "pit_state cannot be inside a blocked cell."
        if self.start_state == self.goal_state:
            return False, "start_state and goal_state must be different."
        if self.start_state == self.pit_state:
            return False, "start_state and pit_state must be different."
        if self.goal_state == self.pit_state:
            return False, "goal_state and pit_state must be different."

        for action in self.actions:
            if action not in self.move_dirs:
                return False, f"Action '{action}' is missing from move_dirs."

        return True, "GridWorld configuration is valid."

    def get_reward(self, state):
        if state == self.goal_state: return 1.0
        if state == self.pit_state: return -1.0
        return -0.04  # Small step cost to encourage efficiency

    def is_terminal(self, state):
        return state == self.goal_state or state == self.pit_state

    def step(self, state, action):
        """Stochastic transition: 0.8 success, 0.1 for each side."""
        r, c = state
        roll = random.random()
        
        if roll < 0.8:
            actual_action = action
        elif roll < 0.9:
            actual_action = self._perp_left(action)
        else:
            actual_action = self._perp_right(action)
            
        dr, dc = self.move_dirs[actual_action]
        nr, nc = r + dr, c + dc
        
        # Check boundaries and blocks
        if (0 <= nr < self.rows and 0 <= nc < self.cols and 
            (nr, nc) not in self.blocks):
            next_state = (nr, nc)
        else:
            next_state = (r, c)
            
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def _perp_left(self, action):
        return {'UP': 'LEFT', 'DOWN': 'RIGHT', 'LEFT': 'DOWN', 'RIGHT': 'UP'}[action]

    def _perp_right(self, action):
        return {'UP': 'RIGHT', 'DOWN': 'LEFT', 'LEFT': 'UP', 'RIGHT': 'DOWN'}[action]


class QLearner:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.q_table = {}

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """
        Epsilon-Greedy Action Selection.
        With probability epsilon, choose a random action (explore).
        Otherwise, choose the action with the highest Q-value (exploit).
        Ties among best actions are broken randomly.
        """
        # Step 1: Decide whether to explore or exploit
        if random.random() < self.epsilon:
            # EXPLORE: pick any action uniformly at random
            return random.choice(self.actions)

        # Step 2: EXPLOIT — find the action with the highest Q-value
        q_values = [self.q_table.get((state, a), 0.0) for a in self.actions]
        max_q = max(q_values)

        # Break ties randomly to avoid bias from action ordering
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, s, a, r, s_prime):
        """
        Q-Learning Update Rule.
        Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a'(Q(s', a')) - Q(s, a))
        """
        # Step 1: Look up the current Q-value for (s, a); default 0.0 for unseen pairs
        current_q = self.q_table.get((s, a), 0.0)

        # Step 2: Compute the maximum Q-value for the next state (s_prime)
        next_q_values = [self.q_table.get((s_prime, a_prime), 0.0)
                         for a_prime in self.actions]
        max_next_q = max(next_q_values)

        # Step 3: Compute the target value
        target = r + self.gamma * max_next_q

        # Step 4: Compute the temporal-difference (TD) error
        td_error = target - current_q

        # Step 5: Update Q(s, a) by blending old estimate toward target
        new_q = current_q + self.alpha * td_error

        # Step 6: Store the updated value back into the Q-table
        self.q_table[(s, a)] = new_q


def explain_environment_validity(env):
    ok, msg = env.validate_configuration()
    header = "GridWorld configuration check: PASS" if ok else "GridWorld configuration check: FAIL"
    return f"{header}\n{msg}"


def train(episodes=10000):
    env = GridWorld()
    ok, msg = env.validate_configuration()
    if not ok:
        raise ValueError(f"Invalid GridWorld configuration: {msg}")
    learner = QLearner(env.actions)
    
    for _ in range(episodes):
        state = env.start_state
        done = False
        while not done:
            action = learner.choose_action(state)
            next_state, reward, done = env.step(state, action)
            learner.learn(state, action, reward, next_state)
            state = next_state
            
    return env, learner


def print_policy(env, learner):
    print("Learned Policy (↑ ↓ ← →):")
    for r in range(env.rows):
        line = "|"
        for c in range(env.cols):
            if (r, c) == env.goal_state: char = " G "
            elif (r, c) == env.pit_state: char = " P "
            elif (r, c) in env.blocks: char = " # "
            else:
                try:
                    # Best action is max Q
                    q_vals = [learner.get_q((r,c), a) for a in env.actions]
                    max_q = max(q_vals)
                    best_a = env.actions[q_vals.index(max_q)]
                    char = {'UP': ' ↑ ', 'DOWN': ' ↓ ', 'LEFT': ' ← ', 'RIGHT': ' → '}[best_a]
                except:
                    char = " ? "
            line += char
        print(line + "|")
    print("-" * 14)


if __name__ == "__main__":
    print("--- Training Q-Learner on 4x3 Grid World ---")
    preview_env = GridWorld()
    print(explain_environment_validity(preview_env))
    environment, agent = train(5000)
    print("Training Complete.")
    print_policy(environment, agent)
    
    # Show some sample Q-values
    example_state = (0, 2)
    print(f"\nQ-values for state {example_state}:")
    for a in environment.actions:
        print(f"  {a}: {agent.get_q(example_state, a):.4f}")