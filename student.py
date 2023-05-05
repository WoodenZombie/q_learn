from blockworld import BlockWorldEnv
import numpy as np
import random
 
class QLearning():
    # don't modify the methods' signatures!
    def __init__(self, env: BlockWorldEnv):
        self.env = env
        self.epsilon = 0.9
        self.alpha_init = 0.5
        self.gamma = 0.9
        self.goalDict = dict()
        self.episode = 1
 
    def get_alpha(self):
        return self.alpha_init
 
    def act(self, s):
        action_dict = self.state_dict[s[0]]
        # epsilon greedy
        if np.random.rand() > self.epsilon:
            a = max(action_dict, key=action_dict.get)
        else:
            a = random.choice(list(action_dict.keys()))
 
        return a
 
    def train(self):
    # Use BlockWorldEnv to simulate the environment with reset() and step() methods.
        for ig in range(72):
            state, goal = self.env.reset()
            while goal in self.goalDict:
                state, goal = self.env.reset()
            self.state_dict = {state: {a: 0 for a in state.get_actions()}}
 
            for episode in range(50):
                done = False
                while not done:
                    a = self.act((state, goal))
                    s_next, reward, done = self.env.step(a)
                    if s_next[0] in self.state_dict:
                        q_max = max(self.state_dict[s_next[0]].values())
                    else:
                        action_dict = {ak: 0 for ak in s_next[0].get_actions()}
                        q_max = 0
                        self.state_dict[s_next[0]] = action_dict
 
                    val = self.alpha_init * (reward + self.gamma * q_max)
                    self.state_dict[state][a] += val - self.alpha_init * self.state_dict[state][a]
                    s = s_next
 
            self.goalDict[goal] = self.state_dict
        # s = self.env.reset()
        # s_, r, done = self.env.step
 
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