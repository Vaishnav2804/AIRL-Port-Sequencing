import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

from PortRoutingEnv import PortRoutingEnv

SEED = 42

def make_env():
    ports = [0, 1, 2, 3, 4]
    adjacency = {0: [1, 3], 1: [2, 3], 2: [23], 3: [24], 4: []}
    vessel_types = ["cargo", "tanker"]
    seasons = ["summer", "winter"]
    env = PortRoutingEnv(ports, adjacency, vessel_types, seasons)
    env = RolloutInfoWrapper(env)
    return env

# 1) Thin evaluation wrapper that swaps env reward with AIRL reward
class RewardedEvalVecEnv(VecEnv):
    def __init__(self, venv: VecEnv, reward_fn):
        self.venv = venv
        self.reward_fn = reward_fn
        super().__init__(venv.num_envs, venv.observation_space, venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, env_rew, terminated, truncated, infos = self.venv.step_wait()
        # Predict imitation reward; ensure numpy arrays and no stat updates
        # obs_t, act_t, next_obs_t, done_t must be batched
        # Here, infos may not have next_obs; use obs returned by env as next_obs of previous step.
        # For simplicity, call reward on current obs and actions with dummy next_obs; AIRL uses shaping, but predict_processed expects next_obs and done.
        # We can store last obs to compute next_obs. A simpler alternative is to call the wrapper approach from docs.
        return obs, env_rew, terminated, truncated, infos  # placeholder; prefer manual evaluator below

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def close(self):
        return self.venv.close()

    def render(self, mode="human"):
        return self.venv.render(mode)

    def get_attr(self, name, indices=None):
        return self.venv.get_attr(name, indices)

    def set_attr(self, name, values, indices=None):
        return self.venv.set_attr(name, values, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def seed(self, seed=None):
        return self.venv.seed(seed)

# 2) Safer: manual evaluation loop using reward_net.predict_processed
def evaluate_with_airl(policy: PPO, venv: VecEnv, reward_net, n_episodes=10, rng=None):
    rng = np.random.default_rng(SEED if rng is None else rng)

    # VecEnv reset -> obs only
    obs = venv.reset()  # shape: (n_envs, 3)

    ep_returns, ep_steps, ep_success, ep_invalid = [], [], [], []
    ep_ret = np.zeros(venv.num_envs, dtype=np.float32)
    ep_len = np.zeros(venv.num_envs, dtype=np.int32)
    ep_inv = np.zeros(venv.num_envs, dtype=np.int32)
    episodes_done = 0

    try:
        adjacency_list = venv.get_attr("adjacency_dict")
    except Exception:
        adjacency_list = [None] * venv.num_envs

    while episodes_done < n_episodes:
        actions, _ = policy.predict(obs, deterministic=True)

        next_obs, _, dones, infos = venv.step(actions)

        # Use terminal_observation for done envs to compute the correct last reward
        next_obs_used = next_obs.copy()
        for i in range(venv.num_envs):
            if dones[i]:
                term_obs = infos[i].get("terminal_observation", None)
                if term_obs is not None:
                    next_obs_used[i] = term_obs

        # Compute imitation reward without updating running stats
        r_im = reward_net.predict_processed(obs, actions, next_obs_used, dones, update_stats=False)
        ep_ret += r_im.astype(np.float32)
        ep_len += 1

        # Invalid action counting: obs[i] is [port, vessel, season]
        for i in range(venv.num_envs):
            if adjacency_list[i] is not None:
                cur_port = int(obs[i, 0])  # FIX: index first feature
                act = int(actions[i])
                valid = act in adjacency_list[i].get(cur_port, [])
                if not valid and act != cur_port:
                    ep_inv[i] += 1

        # Handle episode ends
        for i in range(venv.num_envs):
            if dones[i]:
                time_limit_trunc = bool(infos[i].get("TimeLimit.truncated", False))
                terminated_flag = not time_limit_trunc  # success if not truncated in this toy env
                ep_success.append(terminated_flag)

                ep_returns.append(float(ep_ret[i]))
                ep_steps.append(int(ep_len[i]))
                ep_invalid.append(int(ep_inv[i]))
                episodes_done += 1

                if episodes_done < n_episodes:
                    # Reset one sub-env: env_method returns a list
                    reset_result = venv.env_method("reset", indices=[i])
                    # reset_result is a list with one element
                    # The element should be (obs, info) tuple from your environment's reset method
                    if isinstance(reset_result[0], tuple) and len(reset_result[0]) == 2:
                        # Modern gymnasium format: (obs, info)
                        reset_obs, reset_info = reset_result[0]
                    else:
                        # Older format or different return: just obs
                        reset_obs = reset_result[0]
                    
                    next_obs[i] = reset_obs
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                    ep_inv[i] = 0

        obs = next_obs

    metrics = {
        "imitation_return_mean": float(np.mean(ep_returns)),
        "imitation_return_std": float(np.std(ep_returns)),
        "success_rate": float(np.mean(ep_success)),
        "steps_mean": float(np.mean(ep_steps)),
        "invalid_actions_mean": float(np.mean(ep_invalid)),
    }
    return metrics, {
        "ep_returns": ep_returns,
        "ep_steps": ep_steps,
        "ep_success": ep_success,
        "ep_invalid": ep_invalid,
    }




def main():
    venv = DummyVecEnv([make_env for _ in range(4)])

    # Expert demos
    expert_policy = PPO(MlpPolicy, venv, seed=SEED, verbose=0)
    expert_rollouts = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=30),
        rng=np.random.default_rng(SEED),
    )

    # Learner
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=5,
        seed=SEED,
        tensorboard_log="./ppo_learner_tensorboard/",
    )

    # Reward net
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # AIRL trainer
    airl_trainer = AIRL(
        demonstrations=expert_rollouts,
        demo_batch_size=128,
        gen_replay_buffer_capacity=1024,
        n_disc_updates_per_round=16,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Train
    airl_trainer.train(total_timesteps=50_000)

    # Evaluate with learned reward (manual loop)
    metrics, _ = evaluate_with_airl(learner, venv, reward_net, n_episodes=20)
    print("AIRL evaluation metrics:", metrics)

    # If desired: evaluate expert on the same learned reward to sanity-check
    expert_metrics, _ = evaluate_with_airl(expert_policy, venv, reward_net, n_episodes=20)
    print("Expert (imitation-reward) metrics:", expert_metrics)

if __name__ == "__main__":
    main()
