from gymnasium.utils.env_checker import check_env
from grid_maze_env import GridMazeEnv

env = GridMazeEnv()
check_env(env)
print("check_env passed")

obs, info = env.reset(seed=42)
print("start obs:", obs)
total = 0
for t in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    print(f"step {t+1}: {env.ACTION_NAMES[action]:5s} -> agent ({obs[0]},{obs[1]}) reward {reward}")
    if terminated or truncated:
        break
print("episode return:", total)

P = env.get_transition_model()
print("states in model:", len(P), " example P[0]:", P[0])