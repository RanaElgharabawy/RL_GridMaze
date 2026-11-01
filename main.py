from gymnasium.wrappers import RecordVideo
from grid_maze_env import GridMazeEnv
from policy_iteration import policy_iteration, print_policy

## RecordVideo that repeats each frame, so low-fps videos play correctly in any player. ##
class SmoothRecordVideo(RecordVideo):
    """RecordVideo that repeats each frame, so low-fps videos play correctly in any player."""

    def __init__(self, env, repeat=5, **kwargs):
        super().__init__(env, **kwargs)
        self.repeat = repeat

    def _capture_frame(self):
        super()._capture_frame()  # renders and stores one frame, like normal
        if self.recording and self.recorded_frames:
            self.recorded_frames.extend([self.recorded_frames[-1]] * (self.repeat - 1))


N_EPISODES = 10

env = GridMazeEnv(render_mode="rgb_array")
env = SmoothRecordVideo(env, repeat=5, video_folder="videos", name_prefix="grid_maze", episode_trigger=lambda episode_id: True, fps=10)
maze=env.unwrapped
n_states, n_actions = maze.size * maze.size, maze.action_space.n

for episode in range(N_EPISODES):
    obs, info=env.reset(seed=episode)
    P = maze.get_transition_model()
    policy, V, iterations, sweeps = policy_iteration(P, n_states, n_actions)
    print(f"\n=== Episode {episode} | converged in {iterations} iterations ({sweeps} sweeps) ===")
    print_policy(maze, policy, V)

    total_reward, steps = 0.0, 0
    while True:
        state=maze.pos_to_state((obs[0],obs[1]))
        obs, reward, terminated, truncated, info = env.step(policy[state])
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    print(f"return {total_reward} in {steps} steps")

env.close()
print("\nvideos saved in ./videos")