import gym
import d4rl

env = gym.make('walker2d-medium-replay-v2')
score = env.get_normalized_score(1300) * 100

print(score)