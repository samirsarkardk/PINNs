# ==========================================================
# predict.py
# Prediction for the Heat Equation
# ==========================================================

import torch
import numpy as np

from Aconfig import DEVICE


# ==========================================================
# Exact Solution
# ==========================================================

def exact_solution(x, t):
    """
    Exact solution of the Heat Equation

    u(x,t) = exp(-pi^2 t) sin(pi x)
    """

    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


# ==========================================================
# Prediction
# ==========================================================

def predict(model, t=0.25, num_points=5000):

    model.eval()

    x_test = torch.linspace(0, 1, num_points).view(-1, 1).to(DEVICE)

    t_test = torch.full_like(x_test, t)

    with torch.no_grad():

        u_pred = model(x_test, t_test)

    x = x_test.cpu().numpy().flatten()

    t_array = t * np.ones_like(x)

    u_exact = exact_solution(x, t_array)

    u_pred = u_pred.cpu().numpy().flatten()

    return x, u_exact, u_pred