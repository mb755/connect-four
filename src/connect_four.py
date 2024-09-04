import os
import torch
from utils.environment import ConnectFourEnv
from utils.models import ConnectFourNet, PPO
from utils.config_parser import evaluation_parser

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

###########################################################
# parse command line arguments
###########################################################

parser = evaluation_parser(description="Load a model and play a self-game")

args = vars(parser.parse_args())

input_file = args["input_file"]

# load saved model
net = ConnectFourNet()
net.load_state_dict(torch.load(input_file, weights_only=True))
trained_agent = PPO(net)
print(f"Loaded model from {input_file}")

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
