import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from vallado import vallado
import matplotlib.pyplot as plt
'''
from propagation import vallado as vallado_fast
from special import stumpff_c2 as c2, stumpff_c3 as c3
'''

class HohmannTransferEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    # tof calculated to be 3232.0131611 seconds
    def __init__(self, render_mode=None):
        super().__init__()

        # Define action space: [-540, 540] for each of the three directions
        self.action_space = spaces.Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

        # Define observation space for the current position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Define initial position and target position
        self.initial_pos = np.array([0.0, 0.0, 7000.0])
        self.initial_velocity = np.array([-7.546066877, 0, 0])  # e=0 velocity y direction
        self.target_pos = np.array([0.0, 0.0, -8000.0])

        # Define maximum steps and current step count
        self.max_steps = 1
        self.current_step = 0
        self.grav_const = 398600
        self.tof = 3232.0131611
        """
        x_c = self.initial_pos[0]
        y_c = self.initial_pos[1]
        z_c = self.initial_pos[2]
        radius_r = math.sqrt(x_c**2+y_c**2+z_c**2)
        grav_const = 6.67408*10**(-11)
        mass_e = 5.972 * 10**24
        grav_chang = grav_const*mass_e / radius_r
        a_g = [[-x_c*grav_chang, -y_c*grav_chang, -z_c*grav_chang]]
        velocity_1 = self.initial_velocity + self.action_space +a_g
        """
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize current position
        # self.current_pos = self.initial_pos.copy()

    def step(self, action):
        # self.initial_velocity += self.action_space.sample()
        self.initial_velocity += action
        # self.current_pos += self.initial_velocity
        try:
            [self.current_pos, self.final_velocity] = vallado(k=self.grav_const, r0=self.initial_pos,
                                                                v0=self.initial_velocity,
                                                                tof=self.tof, numiter=10)
            distance_to_target = np.linalg.norm(self.current_pos - self.target_pos)
            change_in_actions = np.sum(np.abs(action))
            reward = 1 / (1 + distance_to_target) + 1 / (1 + change_in_actions)
            print(reward)
            done = self.current_step >= self.max_steps
        except:
            reward = 0
            self.current_pos = self.initial_pos
            done = 1

        self.current_step += 1
        if self.render_mode == "human":
                self._render_frame()
        return self.current_pos, reward, done, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.initial_pos = np.array([0.0, 0.0, 7000.0])
        self.initial_velocity = np.array([-7.546066877, 0, 0])
        return [0,0]

    def render(self):
            if self.render_mode == "rgb_array":
                return self._render_frame()

    def _render_frame(self):

        if self.render_mode == "human":
            if plt.fignum_exists(1):
                self.ax.scatter(self.current_pos[0], self.current_pos[2], color = 'blue', label='Final State')
            else:
                fig, self.ax = plt.subplots()
                self.ax.scatter(self.initial_pos[0], self.initial_pos[2], color = 'red', label='Initial State')
                self.ax.scatter(self.current_pos[0], self.current_pos[2], color = 'blue', label='Final State')
                self.ax.scatter(self.target_pos[0], self.target_pos[2], color = 'green', label='Target State')
                self.ax.legend()
            # Add title and axes labels
            # Make the axes equal

            plt.show()