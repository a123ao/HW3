import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from Gridworld import Gridworld

# ==========================================
# 1. Experience Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ==========================================
# 2. DQN Model
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Training Setup
# ==========================================
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
action_space = 4
state_space = 64 # 4 * 4 * 4 (num_pieces=4, size=4x4)

# Hyperparameters
batch_size = 32
gamma = 0.9
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 300
learning_rate = 1e-3
epochs = 1000
max_steps = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DQN(state_space, action_space).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
buffer = ReplayBuffer(1000)

def get_state(env):
    state = env.board.render_np().reshape(1, -1)
    # Add small noise to prevent zero gradients initially (optional, common trick in the book)
    state = state + np.random.rand(1, 64) / 100.0
    return state.astype(np.float32)

def compute_loss(batch):
    states, actions, rewards, next_states, dones = batch
    
    states = torch.FloatTensor(states).squeeze(1).to(device)
    next_states = torch.FloatTensor(next_states).squeeze(1).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    q_values = model(states)
    next_q_values = model(next_states)
    
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)
    
    loss = loss_fn(q_value, expected_q_value.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

losses = []
rewards_record = []

# ==========================================
# 4. Training Loop (Static Mode)
# ==========================================
print("Starting Training for Static Mode GridWorld...")
for epoch in range(epochs):
    env = Gridworld(size=4, mode='static')
    state = get_state(env)
    
    # Linear annealing of epsilon
    epsilon = max(epsilon_final, epsilon_start - epoch / epsilon_decay)
    
    status = 1
    total_reward = 0
    step = 0
    
    while status == 1 and step < max_steps:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_vals = model(torch.FloatTensor(state).to(device))
                action = torch.argmax(q_vals).item()
        
        # Take action
        env.makeMove(action_set[action])
        reward = env.reward()
        
        # In gridworld, if reward != 0, we hit Goal or Pit
        # To speed up training, let's also give a small penalty for each step
        step_reward = reward - 0.01 
        
        next_state = get_state(env)
        
        if reward != 0:
            status = 0
            done = 1
        else:
            done = 0
            
        buffer.push(state, action, step_reward, next_state, done)
        state = next_state
        total_reward += reward
        step += 1
        
        if len(buffer) > batch_size:
            loss = compute_loss(buffer.sample(batch_size))
            losses.append(loss)
            
    rewards_record.append(total_reward)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Epsilon {epsilon:.2f} | Last 100 Avg Reward: {np.mean(rewards_record[-100:]):.2f}")

# ==========================================
# 5. Plotting & Saving
# ==========================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss (DQN - Static)")
plt.xlabel("Update Steps")
plt.ylabel("MSE Loss")

plt.subplot(1, 2, 2)
window = 50
moving_avg = np.convolve(rewards_record, np.ones(window)/window, mode='valid')
plt.plot(moving_avg)
plt.axhline(y=1.0, color='r', linestyle='--')
plt.title(f"Rewards ({window}-epoch moving avg)")
plt.xlabel("Epochs")
plt.ylabel("Reward")

plt.tight_layout()
plt.savefig("hw3_1_static_results.png")
print("Training completed. Results saved to hw3_1_static_results.png")

torch.save(model.state_dict(), "hw3_1_dqn_static.pth")
