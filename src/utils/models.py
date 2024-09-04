import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class ConnectFourNet(nn.Module):
    def __init__(self):
        super(ConnectFourNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 6 * 7, 256)
        self.policy_head = nn.Linear(256, 7)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 7)
        x = F.relu(self.fc(x))
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value


class PPO:
    def __init__(
        self, net, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01
    ):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.net(state)
        dist = Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_value = values[step + 1]

            delta = (
                rewards[step] + gamma * next_value * next_non_terminal - values[step]
            )
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)

        returns = np.array(advantages) + values
        advantages = np.array(advantages)
        return returns, advantages

    def update(self, states, actions, old_log_probs, rewards, dones, next_state):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        with torch.no_grad():
            _, next_value = self.net(torch.FloatTensor(next_state).unsqueeze(0))
            next_value = next_value.squeeze().item()

        # Compute GAE
        values = self.net(states)[1].detach().squeeze().numpy()
        returns, advantages = self.compute_gae(rewards, values, next_value, dones)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(10):  # Number of optimization epochs
            # Compute new policy and value
            new_policy, new_value = self.net(states)
            new_dist = Categorical(new_policy)
            new_log_probs = new_dist.log_prob(actions)

            # Compute ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

            # Compute losses
            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()
            value_loss = F.mse_loss(new_value.squeeze(), returns)
            entropy = new_dist.entropy().mean()

            # Compute total loss
            loss = (
                policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()
