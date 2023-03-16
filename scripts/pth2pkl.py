import sys
import pickle as pkl
import torch

from sb3_contrib.trpo.utils import get_flat_params, set_flat_params


params = torch.load(sys.argv[1])
flat_params = torch.cat([param.data.view(-1) for name, param in params.items()])

with open(sys.argv[2], "rb") as f:
    template = pkl.load(f)

new_model = set_flat_params(template, flat_params)
with open(sys.argv[3], "wb") as f:
    template = pkl.dump(new_model, f)
