import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple


def _normalize_obs(obs: Any) -> np.ndarray:
    """
    Convert incoming observation to a 2D numpy array (n_envs, obs_dim).
    Handles cases where obs may be (obs, info), list, dict, or 1D array.
    """
    # Unwrap (obs, info) or (obs,) tuples
    if isinstance(obs, tuple):
        if len(obs) >= 1:
            obs = obs
    # If dict, take the first value (not expected here, but safer)
    if isinstance(obs, dict):
        obs = next(iter(obs.values()))
    # Convert lists to arrays
    if not isinstance(obs, np.ndarray):
        obs = np.asarray(obs)
    # Ensure at least 2D: (obs_dim,) -> (1, obs_dim)
    if obs.ndim == 1:
        obs = obs[None, :]
    return obs


class ExpertPortPolicy:
    """
    Expert policy from CSV of sequences.
    Supports PORT_Sequence (names) or H3_Sequence (IDs).
    Exposes:
      - predict(obs, deterministic=True) -> (actions, None)
      - __call__(obs, state=None, dones=None) -> (actions, None)
    """

    def __init__(
        self,
        csv_path: str,
        port_to_index: Dict[str, int],
        action_space: Optional[Any] = None,
        observation_space: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.extra_kwargs = kwargs

        df = pd.read_csv(csv_path)
        use_col = "PORT_Sequence" if "PORT_Sequence" in df.columns else "H3_Sequence"
        if use_col not in df.columns:
            raise ValueError(
                f"CSV must contain PORT_Sequence or H3_Sequence, found: {list(df.columns)}"
            )

        self.seq_dict: Dict[int, list] = {}
        for _, row in df.iterrows():
            mmsi = int(row["mmsi"])
            tokens = [x.strip() for x in str(row[use_col]).split(",") if x.strip()]
            idx_seq = [
                port_to_index[t] for t in tokens if t in port_to_index
            ]  # filter unknown ports
            if not idx_seq:
                continue  # skip this MMSI entirely if no valid ports in env
            self.seq_dict[mmsi] = idx_seq

        if not self.seq_dict:
            raise ValueError(
                "No usable sequences from CSV with provided port_to_index. "
                "Ensure mapping keys match the sequence tokens."
            )

        self.port_to_index = port_to_index
        self.current_idx: Dict[int, int] = {m: 0 for m in self.seq_dict}
        self._mmsi_list = list(self.seq_dict.keys())
        self._round_robin_ptr = 0

    def _pick_mmsi(self) -> int:
        mmsi = self._mmsi_list[self._round_robin_ptr % len(self._mmsi_list)]
        self._round_robin_ptr += 1
        return mmsi

    def predict(self, obs: Any, deterministic: bool = True):
        obs = _normalize_obs(obs)
        n_envs = obs.shape[0]  # FIXED
        actions = np.zeros(n_envs, dtype=np.int32)

        for i in range(n_envs):
            mmsi = self._pick_mmsi()
            seq = self.seq_dict.get(mmsi, [])
            idx = self.current_idx.get(mmsi, 0)
            if not seq:
                cur_port_idx = int(obs[i, 0]) if obs.shape[1] >= 1 else 0
                actions[i] = cur_port_idx
            else:
                if idx >= len(seq):
                    idx = len(seq) - 1
                actions[i] = seq[idx]
                self.current_idx[mmsi] = idx + 1

        if n_envs == 1:
            return np.int32(actions[0]), None  # Return scalar for single env
        return actions, None

    def __call__(
        self,
        obs: Any,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        dones: Optional[np.ndarray] = None,
    ):
        obs = _normalize_obs(obs)
        actions, _ = self.predict(obs, deterministic=True)
        return actions, None

    def reset(self):
        self.current_idx = {m: 0 for m in self.seq_dict}
        self._round_robin_ptr = 0
