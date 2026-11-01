# Grid Maze — Policy Iteration (CMPS458 Assignment 2)

A custom [Gymnasium](https://gymnasium.farama.org/) environment for a 5×5 Grid Maze, solved with
**Policy Iteration** (dynamic programming), rendered with PyGame and recorded with `RecordVideo`.

## The environment
- **Grid:** 5×5. Each episode randomly places the agent start, one goal **G** and two bad cells **X** on 4 distinct cells.
- **Observation:** 8 integers `[agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]`
  (x = column, y = row, (0,0) is the top-left cell).
- **Actions:** `0 = right, 1 = up, 2 = left, 3 = down` (deterministic; moving into a wall leaves the agent in place).
- **Rewards:** +1 on reaching G, −1 on reaching X (both end the episode), 0 otherwise.
- **Truncation:** an episode is cut off after 100 steps.

## Policy Iteration
After each `reset()` the goal and bad cells are fixed, so the agent's position (25 states) is the planning state.
`get_transition_model()` builds the model `P[s][a] = (next_state, reward, terminated)` and `policy_iteration()`
alternates **policy evaluation** (in-place Bellman updates, γ = 0.99, θ = 1e-6) and **policy improvement**
(greedy, switching only on a strict improvement) until the policy is stable.

## Files
| File | Contents |
|---|---|
| `grid_maze_env.py` | The `GridMazeEnv` Gymnasium environment, PyGame rendering and transition model |
| `policy_iteration.py` | Policy evaluation, policy improvement, policy iteration and a policy printer |
| `main.py` | Solves 10 random mazes and records the agent with `RecordVideo` into `videos/` |
| `experiments.py` | Multiple-goal example (Q4) and runtime scaling up to 100×100 (Q5) |
| `test_env.py`, `test_render.py`, `test_pi.py` | Tests: Gymnasium `check_env`, rendering, policy iteration over 200 mazes |
| `videos/` | Recorded episodes of the trained agent |
| `report.pdf` | Outcome summary and answers to the assignment questions |

## Results
- Policy iteration converges in **6.07 iterations on average** (min 3, max 9) over 200 random 5×5 mazes.
- The agent reaches the goal in **200/200** test mazes and all **10/10** recorded episodes.
- A 100×100 maze (10,000 states) is solved in about 96 s, with the work growing roughly as O(n⁴).

## How to run
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

python main.py                 # solve + record videos into videos/
python experiments.py          # Q4 / Q5 experiments (the 100x100 case takes ~2 min)
python test_pi.py              # policy iteration over 200 random mazes
```

## Author
Rana Ahmed Mohamed Samir Elgharabawy — CMPS458 Reinforcement Learning, Fall 2026
