import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- ENVIRONMENT SETUP ---

GRID_SIZE = 10
OBSTACLE_COUNT = 15
EPISODES = 300

class UAVEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.agent_pos = [0, 0]
        self.goal_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
        self.place_obstacles()
        return self.get_state()

    def place_obstacles(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 0
        for _ in range(OBSTACLE_COUNT):
            x, y = np.random.randint(0, GRID_SIZE, size=2)
            if [x, y] != self.agent_pos and [x, y] != self.goal_pos:
                self.grid[x, y] = 1  # Obstacle

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        return state.flatten()

    def step(self, action):
        dxdy = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        new_x = self.agent_pos[0] + dxdy[action][0]
        new_y = self.agent_pos[1] + dxdy[action][1]

        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and self.grid[new_x, new_y] != 1:
            self.agent_pos = [new_x, new_y]

        done = self.agent_pos == self.goal_pos
        reward = 10 if done else -0.1
        return self.get_state(), reward, done

    def render_rgb(self):
        img = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8) * 255
        img[self.grid == 1] = [255, 0, 0]      # Red for obstacles
        img[self.goal_pos[0], self.goal_pos[1]] = [0, 255, 0]  # Green for goal
        img[self.agent_pos[0], self.agent_pos[1]] = [0, 0, 255]  # Blue for agent
        return img

# --- DQN SETUP ---

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID_SIZE * GRID_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 actions
        )

    def forward(self, x):
        return self.net(x)

env = UAVEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=10000)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(state).argmax().item()

def optimize_model():
    if len(memory) < 64:
        return
    batch = random.sample(memory, 64)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (0.99 * next_q * (~dones))

    loss = nn.MSELoss()(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- TRAINING LOOP & VISUALIZATION ---

rewards_list = []

fig, (ax_env, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))
env_img = ax_env.imshow(env.render_rgb())
reward_plot, = ax_plot.plot([], [], label='Reward per Episode')
ax_plot.set_xlim(0, EPISODES)
ax_plot.set_ylim(-10, 15)
ax_plot.set_title("Training Performance")
ax_plot.set_xlabel("Episode")
ax_plot.set_ylabel("Total Reward")
ax_plot.legend()

def update(frame):
    global policy_net, target_net
    state = env.reset()
    total_reward = 0
    done = False
    epsilon = max(0.1, 0.9 - frame / 200)

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        optimize_model()

    if frame % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    rewards_list.append(total_reward)
    env_img.set_data(env.render_rgb())

    reward_plot.set_data(range(len(rewards_list)), rewards_list)
    ax_plot.set_xlim(0, len(rewards_list))
    ax_plot.set_ylim(min(rewards_list) - 5, max(rewards_list) + 5)
    return env_img, reward_plot

ani = animation.FuncAnimation(fig, update, frames=EPISODES, repeat=False, interval=100)
plt.tight_layout()
plt.show()



