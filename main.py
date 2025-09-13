import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

# Import your custom environment
from PortRoutingEnv import PortRoutingEnv

SEED = 42


def make_env():
    """Factory for PortRoutingEnv so it can be vectorized."""
    ports = [0, 1, 2, 3, 4]  # Example port IDs
    adjacency = {
        0: [1, 3],  # Halifax -> {PEI, Ontario}
        1: [2, 3],
        2: [3],
        3: [4],
        4: []
    }
    vessel_types = ["cargo", "tanker"]
    seasons = ["summer", "winter"]

    env = PortRoutingEnv(ports, adjacency, vessel_types, seasons)
    env = RolloutInfoWrapper(env)  # needed for imitation rollouts
    return env


def main():
    # --- 1. Create vectorized environment ---
    venv = DummyVecEnv([make_env for _ in range(4)])

    # --- 2. Generate expert demonstrations ---
    expert_policy = PPO(MlpPolicy, venv, seed=SEED, verbose=0)
    expert_rollouts = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=30),
        rng=np.random.default_rng(SEED),
    )

    # --- 3. Define PPO learner (generator policy) ---
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
        tensorboard_log="./ppo_learner_tensorboard/"
    )

    # --- 4. Define reward network ---
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # --- 5. AIRL trainer ---
    airl_trainer = AIRL(
        demonstrations=expert_rollouts,
        demo_batch_size=128,
        gen_replay_buffer_capacity=1024,
        n_disc_updates_per_round=16,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,   # <--- add this
    )


    # --- 6. Evaluate before training ---
    rewards_before, _ = evaluate_policy(
        learner, venv, n_eval_episodes=10, return_episode_rewards=True
    )
    print("Mean reward BEFORE training:", np.mean(rewards_before))

    # --- 7. Train ---
    airl_trainer.train(total_timesteps=50_000)  # increase for real runs

    # --- 8. Evaluate after training ---
    rewards_after, _ = evaluate_policy(
        learner, venv, n_eval_episodes=10, return_episode_rewards=True
    )
    print("Mean reward AFTER training:", np.mean(rewards_after))


if __name__ == "__main__":
    main()
