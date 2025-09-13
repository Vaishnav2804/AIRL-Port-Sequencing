# Port Routing with Adversarial Inverse Reinforcement Learning (AIRL)

This project demonstrates the use of **Adversarial Inverse Reinforcement Learning (AIRL)** for a simplified **port-to-port vessel routing problem**. The pipeline leverages `stable-baselines3` for reinforcement learning and `imitation` for AIRL training.

---

## Problem Setting

- **Environment (`PortRoutingEnv`)**: Custom `gymnasium` environment for navigating between ports with seasonal and vessel-type constraints.
  - **Observation (state):**
    `[current_port_id, vessel_type_id, season_id]`
  - **Action:**
    Discrete choice of next port to visit.
  - **Reward:**
    Always `0.0`. Rewards are learned via IRL from expert demonstrations.
  - **Episode Termination:**
    - Success: Destination reached.
    - Truncation: Max steps exceeded.

- **Goal:** Learn a reward function (via AIRL) that explains expert vessel routing behavior.

---

## Components

1. **Environment**
   - `PortRoutingEnv` implements ports, adjacency graph, vessel types, and seasonal regimes.
   - Invalid actions keep agent in place (optionally counted).

2. **Expert Demonstrations**
   - Generated from a PPO agent trained directly in the environment.

3. **AIRL Training**
   - Learner agent: PPO (`stable-baselines3`).
   - Reward network: `BasicShapedRewardNet` with input normalization.
   - Trainer: `imitation.algorithms.adversarial.airl.AIRL`.

4. **Evaluation**
   - Manual loop evaluating policy under learned AIRL reward.
   - Metrics: imitation reward, steps/episode, success rate, invalid actions/episode.
   - Compare learner vs expert on learned metrics.

---

## Project Structure

```
.
├── PortRoutingEnv.py    # Custom environment
├── main.py              # AIRL training & evaluation script
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---

## AIRL Pipeline Diagram

AIRL Pipeline Architecture for Port Routing Problem:

<img width="1024" height="327" alt="generated-image" src="https://github.com/user-attachments/assets/d492cb84-5d4f-4022-b370-285c98ad020e" />


---

## Installation & Usage

### 1. Setup Dependencies
```bash
pip install stable-baselines3 imitation gymnasium numpy
```

### 2. Train with AIRL
```bash
python main.py
```

### 3. Sample Evaluation Output
```
AIRL evaluation metrics: {
    'imitation_return_mean': -24.18,
    'imitation_return_std': 7.05,
    'success_rate': 0.0,
    'steps_mean': 20.0,
    'invalid_actions_mean': 18.95
}
Expert (imitation-reward) metrics: {
    'imitation_return_mean': -28.63,
    ...
}
```

---

## Key Metrics
- `imitation_return_mean`: Avg cumulative reward under the learned AIRL reward.
- `success_rate`: Fraction of episodes reaching destination.
- `steps_mean`: Avg steps/episode.
- `invalid_actions_mean`: Avg invalid actions/episode.

---

## References
- Finn et al. Generative Adversarial Imitation Learning (ICML 2016)
- Fu et al. Learning Robust Rewards with Adversarial Inverse Reinforcement Learning (ICLR 2018)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [imitation library](https://github.com/HumanCompatibleAI/imitation)

---

## Next Steps
- Add realistic maritime features (fuel cost, ETA penalties, weather constraints)
- Test reward generalization across seasons
- Benchmark against other IRL methods (e.g., GAIL, BC)
- Explore multi-objective RL/IRL for safety vs efficiency trade-offs
