import torch
import os
from utils.environment import ConnectFourEnv
from utils.models import ConnectFourNet, PPO
from utils.config_parser import training_parser
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

###########################################################
# parse command line arguments
###########################################################

parser = training_parser(description="Train a simple RL model for connect-four")

args = vars(parser.parse_args())

output_suffix = args["output_suffix"]
epochs = args["epochs"]
if not args["input_file"]:
    input_file = None
else:
    input_file = args["input_file"]
starting_epoch = args["starting_epoch"]

max_steps = 42


def train(starting_epoch, num_epochs, input_file):
    env = ConnectFourEnv()
    net = ConnectFourNet()
    if input_file:
        net.load_state_dict(torch.load(input_file, weights_only=True))

    agent = PPO(net)

    for epoch in tqdm(range(starting_epoch, starting_epoch + num_epochs)):
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

        if epoch % 100 == 0:
            tqdm.write(f"Epoch {epoch}, Reward: {episode_reward}")

    return agent


trained_agent = train(starting_epoch, epochs, input_file)
output_filename = f"{root_dir}/output/trained_agent{output_suffix}.pth"
torch.save(
    trained_agent.net.state_dict(),
    output_filename,
)
print(f"Model weights saved as {output_filename}")
