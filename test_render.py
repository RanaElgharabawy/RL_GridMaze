from grid_maze_env import GridMazeEnv

env = GridMazeEnv(render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(30):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        print("episode ended, reward:", reward)
        break

env.close()