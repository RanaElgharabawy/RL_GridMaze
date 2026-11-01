import numpy as np
from grid_maze_env import GridMazeEnv
from policy_iteration import policy_iteration, print_policy

env = GridMazeEnv()
n_states, n_actions = env.size * env.size, env.action_space.n

# --- one maze, shown in detail ---
obs, info = env.reset(seed=42)
P = env.get_transition_model()
policy, V, iterations, sweeps = policy_iteration(P, n_states, n_actions)
print(f"converged in {iterations} policy iterations ({sweeps} evaluation sweeps)\n")
print_policy(env, policy, V)

total, steps = 0, 0
while True:
    action = policy[env.pos_to_state(env.agent)]
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    steps += 1
    if terminated or truncated:
        break
print(f"\ngreedy episode: return {total} in {steps} steps")

# --- many random mazes, for Q3 ---
its, wins = [], 0
for seed in range(200):
    env.reset(seed=seed)
    P = env.get_transition_model()
    policy, V, it, _ = policy_iteration(P, n_states, n_actions)
    its.append(it)
    while True:
        obs, reward, terminated, truncated, info = env.step(policy[env.pos_to_state(env.agent)])
        if terminated or truncated:
            break
    wins += reward == 1.0
print(f"iterations: mean {np.mean(its):.2f}, min {min(its)}, max {max(its)}; reached goal {wins}/200")