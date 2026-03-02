import random, numpy as np, torch


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_torch_generator(seed, device="cuda"):
    return torch.Generator(device=device).manual_seed(int(seed))
