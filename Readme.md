# 🛰️ UAV Path Planning using Deep Reinforcement Learning

This project implements intelligent UAV (Unmanned Aerial Vehicle) navigation in a grid-based environment using **Deep Reinforcement Learning (DRL)** techniques: **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**. The UAV learns to reach a goal while avoiding obstacles, and the system can optionally integrate with **CoppeliaSim** for 3D simulation via a custom **ZeroMQ API**.

---

## 📌 Features

- 🧠 **Deep Q-Network (DQN)** agent for static obstacle environments
- 🔁 **Proximal Policy Optimization (PPO)** agent for dynamic obstacle environments
- 📊 Real-time training visualization using `matplotlib`
- 🛰️ Optional 3D simulation using **CoppeliaSim** and **ZeroMQ Remote API**
  
---

## 🧠 Algorithms

### Deep Q-Network (DQN)
- Discrete action space: Up, Down, Left, Right
- Epsilon-greedy exploration
- Experience Replay Buffer
- Target Network updates every 10 episodes

### Proximal Policy Optimization (PPO)
- Actor-Critic architecture
- Clipped surrogate loss
- Moving obstacles
- Multi-epoch policy updates per episode

would love to collaborate if you have any idea to share or suggest! :)
