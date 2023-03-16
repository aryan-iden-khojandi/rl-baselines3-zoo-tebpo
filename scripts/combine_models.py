import copy
import pickle as pkl
from pathlib import Path
from re import LOCALE
import torch as th

import click

from sb3_contrib.trpo.utils import get_flat_params, set_flat_params


def load_model(fname):
    suffix = Path(fname).suffix
    if suffix == ".pkl":
        with open(fname, "rb") as file:
            return pkl.load(file)
    elif suffix == ".pth":
        return th.load(fname)


@click.command()
@click.argument("policy_a")
@click.argument("policy_b")
@click.argument("output")
@click.option("--lam", default=0.5, type=float)
def main(policy_a, policy_b, output, lam):
    A = load_model(policy_a)
    B = load_model(policy_b)

    mixed_policy_params = (1 - lam) * get_flat_params(A) + \
        (lam) * get_flat_params(B)

    new_policy = set_flat_params(A, mixed_policy_params)
    with open(output, "wb") as file:
        pkl.dump(new_policy, file)


if __name__ == '__main__':
    main()
