import numpy as np

## Policy Evaluation ##
def policy_evaluation(P, policy, n_states, gamma=0.99, theta=1e-6):
    V=np.zeros(n_states)
    sweeps=0
    while True:
        delta=0.0
        for s in range(n_states):
            next_state, reward, done=P[s][policy[s]]
            v_new=reward+(gamma*V[next_state]*(not done))
            delta=max(delta,abs(v_new-V[s]))
            V[s]=v_new
        sweeps+=1

        if delta<theta:
            return V,sweeps

## Policy Improvement ##
def policy_improvement(P, V, policy, n_states, n_actions, gamma=0.99):
    stable=True
    for s in range(n_states):
        q=np.zeros(n_actions)
        for a in range(n_actions):
            next_state, reward, done=P[s][a]
            q[a]=reward+(gamma*V[next_state]*(not done))
        best=int(np.argmax(q))

        if q[best]>q[policy[s]]+1e-9:
            policy[s]=best
            stable=False
    return policy, stable

## Policy Iteration ##
def policy_iteration(P, n_states, n_actions, gamma=0.99, theta=1e-6):
    policy=np.zeros(n_states,dtype=int)
    iterations=0
    total_sweeps=0
    while True:
        V,sweeps=policy_evaluation(P, policy, n_states, gamma, theta)
        total_sweeps+=sweeps
        iterations+=1
        policy, stable=policy_improvement(P, V, policy, n_states, n_actions, gamma)
        if stable:
            return policy, V, iterations, total_sweeps


## Print Policy ##
def print_policy(env, policy, V):
    arrows = [">", "^", "<", "v"]
    goal_s = env.pos_to_state(env.goal)
    bad_s = {env.pos_to_state(bad) for bad in env.bad_cells}
    for y in range(env.size):
        row_p, row_v=[], []
        for x in range(env.size):
            s=env.pos_to_state((x,y))
            row_p.append("G" if s==goal_s else "X" if s in bad_s else arrows[policy[s]])
            row_v.append(f"{V[s]:6.3f}")
        print(" ".join(row_p)+"   "+" ".join(row_v))