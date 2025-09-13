import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PortRoutingEnv(gym.Env):
    """
    Gymnasium-compatible environment for port-to-port routing.

    - States: (current_port_id, vessel_type_id, season_id, optional env features)
    - Actions: index of next port to visit (discrete)
    - Reward: always 0 (IRL provides reward signal)
    - Episode ends when destination is reached or max_steps is exceeded
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 ports,
                 adjacency_dict,
                 vessel_types,
                 seasons,
                 max_steps=20,
                 render_mode=None):
        super().__init__()

        # Basic environment setup
        self.ports = ports
        self.num_ports = len(ports)
        self.adjacency_dict = adjacency_dict
        self.vessel_types = vessel_types
        self.seasons = seasons
        self.max_steps = max_steps

        # Spaces
        low = np.array([0, 0, 0], dtype=np.int32)   # min for each feature
        high = np.array([
            self.num_ports - 1, 
            len(self.vessel_types) - 1,
            len(self.seasons) - 1
        ], dtype=np.int32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.action_space = spaces.Discrete(self.num_ports)

        # Internal state
        self.current_port = None
        self.destination_port = None
        self.vessel_type = None
        self.season = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Sample vessel type, season, start/destination
        self.vessel_type = self.np_random.integers(len(self.vessel_types))
        self.season = self.np_random.integers(len(self.seasons))
        self.current_port = self.np_random.integers(self.num_ports)

        # Ensure destination != start
        self.destination_port = self.current_port
        while self.destination_port == self.current_port:
            self.destination_port = self.np_random.integers(self.num_ports)

        self.steps = 0

        obs = self._get_obs()
        info = {"destination": self.destination_port}
        return obs, info

    def step(self, action):
        self.steps += 1

        # Check if action is valid
        valid_next_ports = self.adjacency_dict.get(self.current_port, [])
        if action not in valid_next_ports:
            next_port = self.current_port
        else:
            next_port = action

        self.current_port = next_port

        # Reward is always 0 (for IRL)
        reward = 0.0

        # Done if destination reached or max steps exceeded
        terminated = self.current_port == self.destination_port
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {"destination": self.destination_port}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([
            int(self.current_port),
            int(self.vessel_type),
            int(self.season)
        ], dtype=np.int32)

    def render(self):
        print(f"Step {self.steps}: Port {self.current_port} "
              f"(Dest={self.destination_port}, Vessel={self.vessel_type}, Season={self.season})")

    def close(self):
        pass
