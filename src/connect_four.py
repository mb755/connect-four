import os
import torch
from utils.environment import ConnectFourEnv
from utils.models import ConnectFourNet, PPO

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# load saved model
net = ConnectFourNet()
net.load_state_dict(
    torch.load(f"{root_dir}/output/trained_agent_weights.pth", weights_only=True)
)
trained_agent = PPO(net)

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
