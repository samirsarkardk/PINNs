# ==========================================================
# utils.py
# Utility Functions
# ==========================================================

import os
import random
import numpy as np
import torch


# ==========================================================
# Set Random Seed
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(model.state_dict(), save_path)

    print(f"\nModel saved to: {save_path}")


# ==========================================================
# Load Model
# ==========================================================

def load_model(model, load_path, device):

    model.load_state_dict(
        torch.load(load_path, map_location=device)
    )

    model.eval()

    print(f"\nModel loaded from: {load_path}")

    return model


# ==========================================================
# Count Trainable Parameters
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )