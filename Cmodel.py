# ==========================================================
# model.py
# Neural Network Architecture for PINN
# ==========================================================

import torch
import torch.nn as nn

from Aconfig import (
    INPUT_DIM,
    OUTPUT_DIM,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS
)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN)
    """

    def __init__(self):
        super().__init__()

        layers = []

        # --------------------------------------------------
        # Input Layer
        # --------------------------------------------------
        layers.append(nn.Linear(INPUT_DIM, HIDDEN_DIM))
        layers.append(nn.Tanh())

        # --------------------------------------------------
        # Hidden Layers
        # --------------------------------------------------
        for _ in range(NUM_HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.Tanh())

        # --------------------------------------------------
        # Output Layer
        # --------------------------------------------------
        layers.append(nn.Linear(HIDDEN_DIM, OUTPUT_DIM))

        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : Spatial coordinate
        t : Time coordinate
        """

        inputs = torch.cat((x, t), dim=1)

        return self.network(inputs)