# ==========================================================
# data.py
# Generate Training Data for the Heat Equation
# ==========================================================

import torch
import numpy as np

from config import (
    DEVICE,
    N_INITIAL,
    N_BOUNDARY,
    N_COLLOCATION,
    X_MIN,
    X_MAX,
    T_MIN,
    T_MAX,
    SEED
)


# ==========================================================
# Set Random Seed
# ==========================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ==========================================================
# Initial Condition
# u(x,0) = sin(pi*x)
# ==========================================================

def generate_initial_points():

    x_i = torch.rand(N_INITIAL, 1) * (X_MAX - X_MIN) + X_MIN

    t_i = torch.zeros_like(x_i)

    u_i = torch.sin(torch.pi * x_i)

    return (
        x_i.to(DEVICE),
        t_i.to(DEVICE),
        u_i.to(DEVICE)
    )


# ==========================================================
# Boundary Condition
# u(0,t)=0
# u(1,t)=0
# ==========================================================

def generate_boundary_points():

    # Left Boundary

    x_left = torch.zeros(N_BOUNDARY // 2, 1)

    t_left = torch.rand(N_BOUNDARY // 2, 1) * (T_MAX - T_MIN) + T_MIN

    u_left = torch.zeros_like(x_left)


    # Right Boundary

    x_right = torch.ones(N_BOUNDARY // 2, 1)

    t_right = torch.rand(N_BOUNDARY // 2, 1) * (T_MAX - T_MIN) + T_MIN

    u_right = torch.zeros_like(x_right)


    # Combine

    x_b = torch.cat((x_left, x_right), dim=0)

    t_b = torch.cat((t_left, t_right), dim=0)

    u_b = torch.cat((u_left, u_right), dim=0)


    return (
        x_b.to(DEVICE),
        t_b.to(DEVICE),
        u_b.to(DEVICE)
    )


# ==========================================================
# Collocation Points
# ==========================================================

def generate_collocation_points():

    x_f = torch.rand(N_COLLOCATION, 1) * (X_MAX - X_MIN) + X_MIN

    t_f = torch.rand(N_COLLOCATION, 1) * (T_MAX - T_MIN) + T_MIN

    return (
        x_f.to(DEVICE),
        t_f.to(DEVICE)
    )