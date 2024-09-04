import torch
import os
from utils.environment import ConnectFourEnv
from utils.models import ConnectFourNet, PPO

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def train(num_episodes=300, max_steps=100):
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
torch.save(
    trained_agent.net.state_dict(), f"{root_dir}/output/trained_agent_weights.pth"
)
print("Model weights saved as trained_agent_weights.pth")
