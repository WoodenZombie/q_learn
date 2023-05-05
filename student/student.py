from blockworld import BlockWorldEnv
from collections import defaultdict
import random
import numpy as np


class QLearning():
    def __init__(self, env):
        self.env = env
        self.q_values = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1

    def act(self, s):
        state, _ = s # extract the current state from the tuple
        if self.epsilon > random.uniform(0, 1):
            # Choose a random action
            a = state.get_actions()
            if not a:
                # If no actions are available, return None
                return None
            return random.choice(a)
        else:
            # Choose the best action based on the Q-values
            q_values = self.q_values[str(state)]
            if not q_values:
                # If no Q-values are available, return a random action
                a = state.get_actions()
                if not a:
                    # If no actions are available, return None
                    return None
                return random.choice(a)
            else:
                max_q_value = max(q_values.values())
                actions_with_max_q_value = [a for a, q in q_values.items() if q == max_q_value]
                return random.choice(actions_with_max_q_value)

    def dist(self,s,goal):
        bad_place_blocks = 0
        self_state = sorted(list(s.get_state()))
        goal_state = sorted(list(goal.get_state()))

        for stack_idx, stack in enumerate(self_state):
            for block_idx, block in enumerate(stack):
                if stack_idx < len(goal_state) and block_idx < len(goal_state[stack_idx]):
                    if block != goal_state[stack_idx][block_idx]:
                        bad_place_blocks += 1
                    else:
                        bad_place_blocks += 1

        if bad_place_blocks == 0:
            return 0
        else:
            return 1/bad_place_blocks

    def reward(self, s):
        """
        Returns the reward for transitioning to state s given goal g.
        """
        # Calculate the distance between the current state and the goal state
        state = s[0]
        goal = s[1]
        dist_to_goal = self.dist(state, goal)

        # If the distance to the goal is 0, the agent has reached the goal and gets a positive reward
        if dist_to_goal == 0:
            return 1.0

        # Otherwise, the agent gets a negative reward proportional to the distance to the goal
        return dist_to_goal

    def train(self):
        # Initialize Q-values
        num_episodes=10000
        gamma=0.9
        alpha=0.5
        self.q_values = defaultdict(lambda: defaultdict(float))

        for episode in range(num_episodes):
            # Reset the environment
            reset = self.env.reset()
            s, g = reset 

            while True:
                # Choose an action
                a = self.act(reset)

                if a is None:
                    # If no actions are available, break out of the loop
                    break

                # Perform the action and observe the next state and reward

                new_s, _, done = self.env.step(a)

                # Update the Q-value for the current state and action
                old_q_value = self.q_values[str(s)][a]
                if done:
                    # If the episode is done, there is no next Q-value
                    next_q_value = 0
                else:
                    # Choose the best action for the next state
                    next_a = self.act(new_s)
                    next_q_value = self.q_values[str(new_s)][next_a]

                new_q_value = (1 - alpha) * old_q_value + alpha * (self.reward(new_s) + gamma * next_q_value)
                self.q_values[str(s)][a] = new_q_value

                # Update the state
                s = new_s[0]

            # Update the epsilon value for the next episode
            self.epsilon *= self.epsilon_decay





if __name__ == '__main__':
	# Here you can test your algorithm. Stick with N <= 4
	N = 4

	env = BlockWorldEnv(N)
	qlearning = QLearning(env)

	# Train
	qlearning.train()

	# Evaluate
	test_env = BlockWorldEnv(N)

	test_problems = 10
	solved = 0
	avg_steps = []

	for test_id in range(test_problems):
		s = test_env.reset()
		done = False

		print(f"\nProblem {test_id}:")
		print(f"{s[0]} -> {s[1]}")

		for step in range(50): 	# max 50 steps per problem
			a = qlearning.act(s)
			s_, r, done = test_env.step(a)

			print(f"{a}: {s[0]}")

			s = s_

			if done:
				solved += 1
				avg_steps.append(step + 1)
				break

	avg_steps = sum(avg_steps) / len(avg_steps)
	print(f"Solved {solved}/{test_problems} problems, with average number of steps {avg_steps}.")