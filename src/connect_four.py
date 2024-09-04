import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class ConnectFourEnv:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = None
        self.current_player = None
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.winner = None
        return self.get_state()

    def step(self, action):
        if self.winner is not None:
            return self.get_state(), 0, True, {}

        if not self.is_valid_action(action):
            return self.get_state(), -10, True, {"invalid_move": True}

        row = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        done = self.check_winner(row, action)
        reward = 0

        if done:
            if self.winner == self.current_player:
                reward = 1
            elif self.winner == 0:  # Draw
                reward = 0
            else:
                reward = -1

        self.current_player = (
            3 - self.current_player
        )  # Switch player (1 -> 2 or 2 -> 1)

        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.array(
            [
                (self.board == 0).astype(int),
                (self.board == 1).astype(int),
                (self.board == 2).astype(int),
            ]
        )

    def is_valid_action(self, action):
        return 0 <= action < self.cols and self.board[0][action] == 0

    def get_next_open_row(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                return r

    def check_winner(self, row, col):
        player = self.board[row][col]

        # Check horizontal
        for c in range(max(0, col - 3), min(col + 1, self.cols - 3)):
            if np.all(self.board[row, c : c + 4] == player):
                self.winner = player
                return True

        # Check vertical
        if row <= 2:
            if np.all(self.board[row : row + 4, col] == player):
                self.winner = player
                return True

        # Check diagonal (positive slope)
        for r, c in zip(range(row - 3, row + 1), range(col - 3, col + 1)):
            if 0 <= r and r + 3 < self.rows and 0 <= c and c + 3 < self.cols:
                if np.all(self.board[r : r + 4, c : c + 4].diagonal() == player):
                    self.winner = player
                    return True

        # Check diagonal (negative slope)
        for r, c in zip(range(row + 3, row - 1, -1), range(col - 3, col + 1)):
            if 0 <= r - 3 and r < self.rows and 0 <= c and c + 3 < self.cols:
                if np.all(
                    np.diagonal(np.fliplr(self.board[r - 3 : r + 1, c : c + 4]))
                    == player
                ):
                    self.winner = player
                    return True

        # Check for draw
        if np.all(self.board != 0):
            self.winner = 0  # 0 represents a draw
            return True

        return False

    def render(self):
        print(" 0 1 2 3 4 5 6")
        print("---------------")
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print(" ", end="|")
                elif cell == 1:
                    print("X", end="|")
                else:
                    print("O", end="|")
            print()
        print("---------------")


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


def train(num_episodes=10000, max_steps=100):
    env = ConnectFourEnv()
    net = ConnectFourNet()
    agent = PPO(net)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        states, actions, log_probs, rewards, dones = [], [], [], [], []

        for step in range(max_steps):
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Update the agent
        agent.update(states, actions, log_probs, rewards, dones, next_state)

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    return agent


trained_agent = train()

# Example of playing a game with the trained agent
env = ConnectFourEnv()
state = env.reset()
done = False
while not done:
    action, _ = trained_agent.get_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        if env.winner == 0:
            print("It's a draw!")
        else:
            print(f"Player {env.winner} wins!")
