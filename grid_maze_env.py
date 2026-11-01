import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces

class GridMazeEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    ACTIONS={
            0: np.array([1,0]), #right
            1: np.array([0,-1]), #up
            2: np.array([-1,0]), #left
            3: np.array([0,1]), #down
        }
    ACTION_NAMES=["right", "up", "left", "down"]

    def __init__(self, size=5, max_steps=100, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        # [agent_x, agent_y, goal_x, goal_y, bad1_x, bad1_y, bad2_x, bad2_y]
        self.observation_space = spaces.MultiDiscrete([size] * 8)

        self.agent = None
        self.goal = None
        self.bad_cells = None
        self.steps = 0

        self.cell_size=100
        self.window_size=self.size*self.cell_size
        self.window=None
        self.clock=None
        self.font=None


    ## HELPERS ##
    def _get_obs(self):
        return np.concatenate([self.agent, self.goal, *self.bad_cells]).astype(np.int64)

    def _get_info(self):
        return {"distance_to_goal": int(np.abs(self.agent-self.goal).sum())}

    def pos_to_state(self,pos):
        return int((pos[1]*self.size)+pos[0])

    def state_to_pos(self,state):
        return np.array([state%self.size, state//self.size])

    ## Core API ##
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cells=self.np_random.choice(self.size*self.size, size=4, replace=False)
        self.agent,self.goal, bad1, bad2= [self.state_to_pos(cell) for cell in cells]
        self.bad_cells=[bad1,bad2]
        self.steps=0

        if self.render_mode=="human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.agent = np.clip(self.agent+self.ACTIONS[int(action)], 0, self.size-1)
        self.steps+=1

        if np.array_equal(self.agent,self.goal):
            reward,terminated=1.0,True
        elif any(np.array_equal(self.agent,bad) for bad in self.bad_cells):
            reward,terminated=-1.0,True
        else:
            reward,terminated=0.0,False

        truncated=self.steps>=self.max_steps

        if self.render_mode=="human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()


    ## Model For Dynamic Programming ##
    def get_transition_model(self):
        goal_s=self.pos_to_state(self.goal)
        bad_s={self.pos_to_state(bad) for bad in self.bad_cells}
        P={}
        for s in range(self.size*self.size):
            P[s]={}
            for a,move in self.ACTIONS.items():
                if s==goal_s or s in bad_s:
                    P[s][a]=(s,0.0,True)
                    continue
                next_pos=np.clip(self.state_to_pos(s)+move,0,self.size-1)
                next_state=self.pos_to_state(next_pos)
                if next_state==goal_s:
                    P[s][a]=(next_state,1.0,True)
                elif next_state in bad_s:
                    P[s][a]=(next_state,-1.0,True)
                else:
                    P[s][a]=(next_state,0.0,False)
        return P

    ## Render ##
    def render(self):
        if self.render_mode=="rgb_array":
            return self._render_frame()
        elif self.render_mode=="human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode=="human":
            pygame.init()
            pygame.display.init()
            self.window=pygame.display.set_mode((self.window_size,self.window_size))
            pygame.display.set_caption("Grid Maze")
            self.clock=pygame.time.Clock()

        if self.font is None:
            pygame.font.init()
            self.font=pygame.font.SysFont("arial", self.cell_size // 2, bold=True)

        c=self.cell_size
        canvas=pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))

        self._draw_cell(canvas, self.goal, (0,170,80), "G") 
        for bad in self.bad_cells:
            self._draw_cell(canvas, bad, (220,40,40), "X")

        center = (self.agent + 0.5) * c
        pygame.draw.circle(canvas, (255, 210, 0), center, c // 3)
        pygame.draw.circle(canvas, (60, 60, 60), center, c // 3, 3)

        for i in range(self.size+1):
            pygame.draw.line(canvas, (0, 0, 0), (0, c * i), (self.window_size, c * i), 2)
            pygame.draw.line(canvas, (0, 0, 0), (c * i, 0), (c * i, self.window_size), 2)

        if self.render_mode=="human":
            self.window.blit(canvas,(0,0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))

    def _draw_cell(self, canvas, pos, color, label):
        c=self.cell_size
        rect=pygame.Rect(int(pos[0])*c, int(pos[1])*c, c, c)
        pygame.draw.rect(canvas, color, rect)
        text=self.font.render(label, True, (255,255,255))
        canvas.blit(text, text.get_rect(center=rect.center))

    ## Close the Environment ##
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.font = None