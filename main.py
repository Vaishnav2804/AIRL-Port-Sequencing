import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.airl import AIRL

from ExpertPolicy import ExpertPortPolicy
from PortRoutingEnv import PortRoutingEnv
from helper import build_port_to_index

SEED = 42


def build_adjacency_from_csv(csv_path, port_to_index):
    """
    Build adjacency dictionary of reachable ports from CSV dataset.
    (Currently a placeholder: actual logic depends on dataset structure,
    e.g., adding edges between consecutive port visits.)
    """
    adjacency = {idx: [] for idx in port_to_index.values()}
    return adjacency


def make_env_from_csv(csv_path, allowed_ports):
    """
    Create a PortRoutingEnv environment configured from CSV data.
    Includes port mapping, adjacency, vessel types, and seasons.
    """
    port_to_index = build_port_to_index(csv_path, allowed_ports=allowed_ports)
    ports = list(port_to_index.values())
    adjacency = build_adjacency_from_csv(csv_path, port_to_index)
    vessel_types = ["cargo", "tanker"]
    seasons = ["summer", "winter"]

    env = PortRoutingEnv(ports, adjacency, vessel_types, seasons)
    env = RolloutInfoWrapper(env)  # adds episode info tracking
    return env


def evaluate_with_airl(policy, venv: VecEnv, reward_net, n_episodes=10):
    """
    Evaluate a policy (learner or expert) using the AIRL learned reward.
    Runs episodes, collects AIRL-based returns, and calculates statistics.
    """
    obs = venv.reset()
    ep_returns, ep_steps = [], []
    ep_ret = np.zeros(venv.num_envs, dtype=np.float32)
    ep_len = np.zeros(venv.num_envs, dtype=np.int32)

    episodes_done = 0
    while episodes_done < n_episodes:
        # Policy can be SB3's .predict() or a callable expert
        if callable(policy):
            actions, _ = policy(obs, None, None)
        else:
            actions, _ = policy.predict(obs, deterministic=True)

        # Step environment
        next_obs, _, dones, infos = venv.step(actions)

        # Reward predicted from AIRL reward network
        r_im = reward_net.predict_processed(
            obs, actions, next_obs, dones, update_stats=False
        )
        ep_ret += r_im
        ep_len += 1

        # Handle completed episodes
        for i in range(venv.num_envs):
            if dones[i]:
                ep_returns.append(float(ep_ret[i]))
                ep_steps.append(int(ep_len[i]))
                ep_ret[i] = 0.0
                ep_len[i] = 0
                episodes_done += 1

        obs = next_obs

    metrics = {
        "imitation_return_mean": float(np.mean(ep_returns)),
        "imitation_return_std": float(np.std(ep_returns)),
        "steps_mean": float(np.mean(ep_steps)),
    }
    return metrics, {"ep_returns": ep_returns, "ep_steps": ep_steps}


def main():
    csv_path = "port_visits_with_sequences_collapsed.csv"
    env_ports = ["PORT HAWKESBURY", "PORT HASTINGS", "PORT X", "PORT Y", "PORT Z"]

    # Create 4 parallel environments from CSV data
    venv = DummyVecEnv(
        [lambda: make_env_from_csv(csv_path, env_ports) for _ in range(4)]
    )

    # Build mapping from ports in CSV
    port_to_index = build_port_to_index(csv_path, allowed_ports=env_ports)

    # Load expert policy (based on dataset rules/heuristics)
    expert_policy = ExpertPortPolicy(
        csv_path=csv_path,
        port_to_index=port_to_index,
        action_space=venv.action_space,
        observation_space=venv.observation_space,
    )

    # Generate expert demonstrations (trajectories) for AIRL training
    expert_rollouts = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=30),
        rng=np.random.default_rng(SEED),
    )

    # PPO learner (generator policy used in AIRL)
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

    # Reward network that AIRL trains to imitate expert reward function
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Set up AIRL trainer: adversarial setup between learner (gen) and reward model
    airl_trainer = AIRL(
        demonstrations=expert_rollouts,
        demo_batch_size=128,
        gen_replay_buffer_capacity=1024,
        n_disc_updates_per_round=16,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        log_dir="./airl_tensorboard/",
        init_tensorboard=True,
    )

    # Train learner with AIRL framework
    print("Training AIRL...")
    airl_trainer.train(total_timesteps=500_000)

    # Evaluate the trained learner
    print("Evaluating learner...")
    metrics, _ = evaluate_with_airl(learner, venv, reward_net, n_episodes=20)
    print("Learner AIRL evaluation metrics:", metrics)

    # Evaluate expert performance for comparison
    print("Evaluating expert...")
    expert_metrics, _ = evaluate_with_airl(
        expert_policy, venv, reward_net, n_episodes=20
    )
    print("Expert AIRL evaluation metrics:", expert_metrics)


if __name__ == "__main__":
    main()
