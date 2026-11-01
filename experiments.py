import time
from policy_iteration import policy_iteration

MOVES = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # right, up, left, down (same order as the env)
ARROWS = [">", "^", "<", "v"]

def build_model(size, goals, bads):
    goal_s={y*size+x for x,y in goals}
    bad_s={y*size+x for x,y in bads}
    P={}
    for s in range(size*size):
        x,y=s%size,s//size
        P[s]={}
        for a, (dx,dy) in enumerate(MOVES):
            if s in goal_s or s in bad_s:
                P[s][a]=(s,0.0,True)
                continue
            nx,ny= min(max(x+dx,0), size-1), min(max(y+dy,0), size-1)
            next_state=ny*size+nx
            if next_state in goal_s:
                P[s][a]=(next_state,1.0,True)
            elif next_state in bad_s:
                P[s][a]=(next_state,-1.0,True)
            else:
                P[s][a]=(next_state,0.0,False)
    return P, goal_s, bad_s

def show(size, policy, V, goal_s, bad_s):
    for y in range(size):
        cells, vals=[], []
        for x in range(size):
            s=y*size+x
            cells.append("G" if s in goal_s else "X" if s in bad_s else ARROWS[policy[s]])
            vals.append(f"{V[s]:6.3f}")
        print(" ".join(cells)+"   "+" ".join(vals))



## Q4: Multiple Goal Cells ##
print("=== Q4: two goals at (0,0) and (4,4), bad cells at (2,1) and (2,3) ===")
size=5
P, goal_s, bad_s= build_model(size, goals=[(0,0),(4,4)], bads=[(2,1),(2,3)])
policy, V, iterations, _=policy_iteration(P, size*size, 4)
print(f"converged in {iterations} iterations")
show(size, policy, V, goal_s, bad_s)

## Q5: How does it Scale with Maze Size? ##
print("\n=== Q5: scaling (goal in one corner, 2 bad cells, agent anywhere) ===")
print(f"{'size':>8} {'states':>8} {'iterations':>11} {'sweeps':>8} {'time (s)':>9}")
for size in [5, 10, 20, 30, 50, 100]:
    P, _, _=build_model(size, goals=[(size-1,size-1)], bads=[(1,1),(size//2,size//2)])
    start=time.perf_counter()
    policy, V, iterations, sweeps=policy_iteration(P, size*size, 4)
    elapsed=time.perf_counter()-start
    print(f"{size:>4}x{size:<3} {size * size:>8} {iterations:>11} {sweeps:>8} {elapsed:>9.2f}")