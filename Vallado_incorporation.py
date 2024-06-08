import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from vallado import vallado
from propagation import vallado as vallado_fast
from special import stumpff_c2 as c2, stumpff_c3 as c3

class CustomEnvironment(gym.Env):

    def __init__(self):
        super(CustomEnvironment, self).__init__()
        [r[i], v[i]] = vallado(k=gravitation_constant, r0=init_r, v0=init_v, tof=temp_tof, numiter=iterations)
        #tof calculated to be 3232.0131611 seconds
        #unsure what the notation is for init_r, so i created two variables for it :,)

        # Define action space: [-540, 540] for each of the three directions
        self.action_space = spaces.Box(low=-9.0, high=9.0, shape=(3,), dtype=np.float32)

        # Define observation space for the current position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Define initial position and target position
        self.init_radius = np.array([0.0, 0.0, 7000.0])
        self.vallado.init_v = np.array([-7.546066877, 0, 0]) #e=0 velocity y direction
        self.target_pos = np.array([0.0, 0.0, -8000.0])

        # Define maximum steps and current step count
        self.max_steps = 1
        self.current_step = 0
        x_c = self.init_radius[0]
        y_c = self.init_radius[1]
        z_c = self.init_radius[2]
        vallado.init_r = math.sqrt(x_c**2+y_c**2+z_c**2)
        grav_const = 6.67408*10**(-11)
        mass_e = 5.972 * 10**24
        grav_chang = grav_const*mass_e / radius_r
        a_g = [[-x_c*grav_chang, -y_c*grav_chang, -z_c*grav_chang]]
        velocity_1 = self.vallado.init_v + self.action_space +a_g



        # Initialize current position
        self.current_pos = self.vallado.init_r.copy()



    def step(self, action):

        self.init_v += self.action_space.sample()
        self.current_pos += self.init_v
        distance_to_target = np.linalg.norm(self.current_pos - self.target_pos)
        change_in_actions = np.sum(np.abs(action))
        reward = 1/(1 + distance_to_target) + 1/(1 + change_in_actions)
        print(reward)
        done = self.current_step >= self.max_steps

        self.current_step += 1

        return self.current_pos, reward, done, {}



    def reset(self):
        self.current_step = 0
        self.init_radius = np.array([0.0, 0.0, 7000.0])
        self.vallado.init_v = np.array([-7.546066877, 0, 0])
        self.current_pos
        return self.current_pos

#excuting the code is tripping me up :.)
#in the example code they used
    while True:
        __init__(self)
        step(self, action)
        reset(self)
