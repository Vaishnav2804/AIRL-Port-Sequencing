import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PortRoutingEnv(gym.Env):
    """
    Gymnasium-compatible environment for port-to-port routing.

    - Observation: (current_port_id, vessel_type_id, season_id), dtype int32
    - Action: next port to visit (Discrete(num_ports))
    - Reward: 0.0 (AIRL provides learned reward)
    - Episode ends when destination reached or max_steps exceeded
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, ports, adjacency_dict, vessel_types, seasons, max_steps=20):
        super().__init__()
        self.ports = ports
        self.num_ports = len(ports)
        self.adjacency_dict = adjacency_dict
        self.vessel_types = vessel_types
        self.seasons = seasons
        self.max_steps = max_steps

        low = np.array([0, 0, 0], dtype=np.int32)
        high = np.array(
            [self.num_ports - 1, len(self.vessel_types) - 1, len(self.seasons) - 1],
            dtype=np.int32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_ports)

        self.current_port = None
        self.destination_port = None
        self.vessel_type = None
        self.season = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.vessel_type = self.np_random.integers(len(self.vessel_types))
        self.season = self.np_random.integers(len(self.seasons))
        self.current_port = self.np_random.integers(self.num_ports)

        self.destination_port = self.current_port
        while self.destination_port == self.current_port:
            self.destination_port = self.np_random.integers(self.num_ports)

        self.steps = 0
        return self._get_obs(), {"destination": self.destination_port}

    def step(self, action):
        self.steps += 1
        valid_next_ports = self.adjacency_dict.get(self.current_port, [])
        next_port = action if action in valid_next_ports else self.current_port
        self.current_port = next_port

        reward = 0.0
        terminated = self.current_port == self.destination_port
        truncated = self.steps >= self.max_steps

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {"destination": self.destination_port},
        )

    def _get_obs(self):
        return np.array(
            [self.current_port, self.vessel_type, self.season], dtype=np.int32
        )

    def render(self):
        print(
            f"Step {self.steps}: Port {self.current_port} "
            f"(Dest={self.destination_port}, Vessel={self.vessel_type}, Season={self.season})"
        )

    def close(self):
        pass
