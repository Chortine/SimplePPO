import torch
import torch.nn as nn


# a wuziqi environment

# class Env():
#     def __init__(self):
#         self.
#
class PPONetwork(torch.nn.Module):
    def __init__(self):
        super.__init__()
        # ======== encoder ============= #
        self.input_dense = nn.Sequential(
            nn.Linear(in_features=8, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU()
        )
        # ======= decoder ============== #
        self.action_decoder = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=4),
        )

        # self.
    def sample_from_logits(self, logits):
        temperature = 1.0


def PPOModel():
    """
    Contains the network, loss, optimizer, gradient
    :return:
    """



def test_gym():
    import gym
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        # action = policy(observation)  # User-defined policy function
        action = 1
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == "__main__":
    test_gym()