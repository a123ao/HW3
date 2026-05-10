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
# 2. DQN Architectures
# ==========================================
# Standard DQN (for Double DQN)
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

# Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# ==========================================
# 3. Training Function
# ==========================================
def train_agent(variant_name, use_double=False, use_dueling=False):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    action_space = 4
    state_space = 64

    # Hyperparameters
    batch_size = 32
    gamma = 0.9
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 800  # player mode is harder, slower decay
    learning_rate = 1e-3
    epochs = 2000
    max_steps = 50
    sync_target_freq = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    if use_dueling:
        model = DuelingDQN(state_space, action_space).to(device)
        target_model = DuelingDQN(state_space, action_space).to(device)
    else:
        model = DQN(state_space, action_space).to(device)
        target_model = DQN(state_space, action_space).to(device)
        
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(2000)

    def get_state(env):
        state = env.board.render_np().reshape(1, -1)
        state = state + np.random.rand(1, 64) / 100.0
        return state.astype(np.float32)

    losses = []
    rewards_record = []
    
    print(f"\n--- Starting Training for {variant_name} ---")
    
    global_step = 0
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='player') # Player mode: random start position
        state = get_state(env)
        
        epsilon = max(epsilon_final, epsilon_start - epoch / epsilon_decay)
        
        status = 1
        total_reward = 0
        step = 0
        
        while status == 1 and step < max_steps:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_vals = model(torch.FloatTensor(state).to(device))
                    action = torch.argmax(q_vals).item()
            
            env.makeMove(action_set[action])
            reward = env.reward()
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
            global_step += 1
            
            # Sync target network
            if global_step % sync_target_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
            # Train
            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = batch
                
                states = torch.FloatTensor(states).squeeze(1).to(device)
                next_states = torch.FloatTensor(next_states).squeeze(1).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                q_values = model(states)
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    if use_double:
                        # Double DQN logic
                        online_next_q_values = model(next_states)
                        next_actions = torch.argmax(online_next_q_values, dim=1)
                        target_next_q_values = target_model(next_states)
                        next_q_value = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else:
                        # Standard Target logic
                        target_next_q_values = target_model(next_states)
                        next_q_value = target_next_q_values.max(1)[0]
                        
                expected_q_value = rewards + gamma * next_q_value * (1 - dones)
                
                loss = loss_fn(q_value, expected_q_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
        rewards_record.append(total_reward)
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Epsilon {epsilon:.2f} | Last 100 Avg Reward: {np.mean(rewards_record[-100:]):.2f}")
            
    return losses, rewards_record

# Run the experiments
if __name__ == "__main__":
    _, rewards_double = train_agent("Double DQN", use_double=True, use_dueling=False)
    _, rewards_dueling = train_agent("Dueling DQN", use_double=False, use_dueling=True)
    
    # Plot comparisons
    plt.figure(figsize=(10, 6))
    
    window = 100
    ma_double = np.convolve(rewards_double, np.ones(window)/window, mode='valid')
    ma_dueling = np.convolve(rewards_dueling, np.ones(window)/window, mode='valid')
    
    plt.plot(ma_double, label="Double DQN", alpha=0.8)
    plt.plot(ma_dueling, label="Dueling DQN", alpha=0.8)
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.title(f"Comparison of DQN Variants in 'player' mode ({window}-epoch moving avg)")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("hw3_2_variants_results.png")
    print("\nComparison completed. Results saved to hw3_2_variants_results.png")
