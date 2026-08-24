# ==========================================================
# plots.py
# Plotting Functions
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# Plot Training Loss
# ==========================================================

def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(history["total"], label="Total Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.yscale("log")

    plt.grid(True)

    plt.legend()

    plt.show()


# ==========================================================
# Plot Exact vs Prediction
# ==========================================================

def plot_prediction(x, u_exact, u_pred):

    plt.figure(figsize=(8,5))

    plt.plot(
        x,
        u_exact,
        'k-',
        linewidth=2,
        label="Exact Solution"
    )

    plt.plot(
        x,
        u_pred,
        'r--',
        linewidth=2,
        label="PINN Prediction"
    )

    plt.xlabel("x")

    plt.ylabel("u(x,t)")

    plt.title("Exact vs PINN Prediction")

    plt.grid(True)

    plt.legend()

    plt.show()