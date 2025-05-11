import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.distributions import Categorical

# --- Hyperparameters ---
GRID_SIZE = 10
OBSTACLE_COUNT = 15
EPISODES = 300
GAMMA = 0.99
EPS_CLIP = 0.2
LR = 2.5e-4
UPDATE_EPOCHS = 4
BATCH_SIZE = 64
MAX_STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment ---
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
        self.grid.fill(0)
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 0
        for _ in range(OBSTACLE_COUNT):
            while True:
                x, y = np.random.randint(0, GRID_SIZE, 2)
                if [x, y] != self.agent_pos and [x, y] != self.goal_pos:
                    self.grid[x, y] = 1
                    break

    def move_obstacles(self):
        new_grid = np.zeros_like(self.grid)
        new_grid[self.goal_pos[0], self.goal_pos[1]] = 0
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        obstacles = list(zip(*np.where(self.grid == 1)))
        for x, y in obstacles:
            np.random.shuffle(directions)
            moved = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
                    [nx, ny] != self.agent_pos and new_grid[nx, ny] != 1 and [nx, ny] != self.goal_pos):
                    new_grid[nx, ny] = 1
                    moved = True
                    break
            if not moved:
                new_grid[x, y] = 1
        self.grid = new_grid

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        return state.flatten()

    def step(self, action):
        dxdy = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        nx = self.agent_pos[0] + dxdy[action][0]
        ny = self.agent_pos[1] + dxdy[action][1]

        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[nx, ny] != 1:
            self.agent_pos = [nx, ny]

        done = self.agent_pos == self.goal_pos
        reward = 10 if done else -0.1

        self.move_obstacles()

        return self.get_state(), reward, done

    def render_rgb(self):
        img = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8) * 255
        img[self.grid == 1] = [255, 0, 0]      # Obstacles
        img[self.goal_pos[0], self.goal_pos[1]] = [0, 255, 0]  # Goal
        img[self.agent_pos[0], self.agent_pos[1]] = [0, 0, 255]  # Agent
        return img

# --- PPO Networks ---
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(GRID_SIZE * GRID_SIZE, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# --- PPO Agent ---
class PPOAgent:
    def __init__(self):
        self.policy = ActorCritic().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).to(device)
        probs, value = self.policy_old(state)
        dist = Categorical(probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        memory.values.append(value)
        return action.item()

    def update(self, memory):
        rewards = []
        discounted = 0
        for r, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done: discounted = 0
            discounted = r + GAMMA * discounted
            rewards.insert(0, discounted)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device).unsqueeze(1)

        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        old_values = torch.stack(memory.values).detach()

        advantages = rewards - old_values.detach()

        for _ in range(UPDATE_EPOCHS):
            probs, values = self.policy(old_states)
            dist = Categorical(probs)
            logprobs = dist.log_prob(old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear()

# --- Memory ---
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

# --- Main Training Loop & Visualization ---
env = UAVEnv()
agent = PPOAgent()
memory = Memory()
rewards_list = []

fig, (ax_env, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))
env_img = ax_env.imshow(env.render_rgb())
reward_plot, = ax_plot.plot([], [], label="Episode Reward")
ax_plot.set_xlim(0, EPISODES)
ax_plot.set_ylim(-10, 20)
ax_plot.set_title("PPO UAV Navigation")
ax_plot.set_xlabel("Episode")
ax_plot.set_ylabel("Reward")
ax_plot.legend()

def update(frame):
    state = env.reset()
    total_reward = 0
    memory.clear()

    for _ in range(MAX_STEPS):
        action = agent.select_action(state, memory)
        next_state, reward, done = env.step(action)

        memory.rewards.append(reward)
        memory.dones.append(done)
        state = next_state
        total_reward += reward

        if done:
            break

    agent.update(memory)

    rewards_list.append(total_reward)
    env_img.set_data(env.render_rgb())
    reward_plot.set_data(range(len(rewards_list)), rewards_list)
    ax_plot.set_xlim(0, len(rewards_list))
    ax_plot.set_ylim(min(rewards_list) - 5, max(rewards_list) + 5)
    return env_img, reward_plot

ani = animation.FuncAnimation(fig, update, frames=EPISODES, repeat=False, interval=100)
plt.tight_layout()
plt.show()
