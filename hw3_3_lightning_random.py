import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from Gridworld import Gridworld
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
import logging

# Set logging level to avoid too much text
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# ==========================================
# 1. Experience Replay Buffer & Dataset
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

class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=32):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.sample_size)
        for i in range(self.sample_size):
            yield states[i], actions[i], rewards[i], next_states[i], dones[i]

# ==========================================
# 2. PyTorch Lightning Module
# ==========================================
class LitDQN(pl.LightningModule):
    def __init__(self, state_space=64, action_space=4):
        super().__init__()
        self.save_hyperparameters()
        
        self.net = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        
        self.target_net = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        self.target_net.load_state_dict(self.net.state_dict())
        
        self.buffer = ReplayBuffer(10000)
        self.env = Gridworld(size=4, mode='random')
        self.state = self.get_state(self.env)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 10000 # Random mode is hard, explore longer
        self.batch_size = 64
        self.lr = 1e-3
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        
        self.episode_reward = 0
        self.rewards_record = []
        
        self.populate_buffer(self.batch_size * 2)

    def get_state(self, env):
        state = env.board.render_np().reshape(1, -1)
        state = state + np.random.rand(1, 64) / 100.0
        return state.astype(np.float32)

    def populate_buffer(self, steps):
        for _ in range(steps):
            self.play_step(force_random=True)

    def forward(self, x):
        return self.net(x)

    def play_step(self, force_random=False):
        if force_random or random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_vals = self(torch.FloatTensor(self.state).to(self.device))
                action = torch.argmax(q_vals).item()
                
        self.env.makeMove(self.action_set[action])
        reward = self.env.reward()
        step_reward = reward - 0.01 
        
        next_state = self.get_state(self.env)
        
        if reward != 0:
            done = 1
        else:
            done = 0
            
        self.buffer.push(self.state, action, step_reward, next_state, done)
        self.state = next_state
        self.episode_reward += reward
        
        if done:
            self.rewards_record.append(self.episode_reward)
            self.episode_reward = 0
            self.env = Gridworld(size=4, mode='random')
            self.state = self.get_state(self.env)
            
        self.epsilon = max(self.epsilon_min, self.epsilon - (1.0 - self.epsilon_min) / self.epsilon_decay)
        
    def training_step(self, batch, batch_idx):
        self.play_step()
        
        states, actions, rewards, next_states, dones = batch
        states = states.squeeze(1).float()
        next_states = next_states.squeeze(1).float()
        rewards = rewards.float()
        dones = dones.float()
        actions = actions.long()
        
        q_values = self(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_value = next_q_values.max(1)[0]
            
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        
        loss = nn.MSELoss()(q_value, expected_q_value)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('epsilon', self.epsilon, prog_bar=True)
        if len(self.rewards_record) > 0:
            self.log('avg_reward', np.mean(self.rewards_record[-100:]), prog_bar=True)
            
        return loss

    def on_train_epoch_end(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # 1. Learning Rate Scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size)

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == '__main__':
    model = LitDQN()
    
    # 2. PyTorch Lightning Trainer with Gradient Clipping
    trainer = pl.Trainer(
        max_epochs=200, 
        limit_train_batches=200, 
        gradient_clip_val=1.0, # Gradient Clipping integrated here
        enable_checkpointing=False,
        logger=False
    )
    
    print("\n--- Starting Training in 'random' mode with PyTorch Lightning ---")
    trainer.fit(model)
    
    plt.figure(figsize=(8, 5))
    window = 100
    if len(model.rewards_record) > window:
        ma = np.convolve(model.rewards_record, np.ones(window)/window, mode='valid')
        plt.plot(ma, label="Moving Average Reward")
    else:
        plt.plot(model.rewards_record, label="Reward")
        
    plt.title("PyTorch Lightning DQN in Random Mode")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hw3_3_lightning_results.png")
    print("Training completed. Results saved to hw3_3_lightning_results.png")
