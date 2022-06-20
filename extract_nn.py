import sys
import numpy as np
from stable_baselines3 import PPO


model = PPO.load(sys.argv[1])
mlp_extractor_weights = model.policy._modules["mlp_extractor"].policy_net.state_dict()

w0 = mlp_extractor_weights["0.weight"].numpy()
b0 = mlp_extractor_weights["0.bias"].numpy()
# there is no w1` and b1 for some reason so we skip straight to 2.weight and 2.bias
w1 = mlp_extractor_weights["2.weight"].numpy()
b1 = mlp_extractor_weights["2.bias"].numpy()
w_out = model.policy._modules["action_net"].weight.detach().numpy()

np.savez(sys.argv[1] + "_numpy_weights.npz", w0=w0, b0=b0, w1=w1, b1=b1, w_out=w_out)
