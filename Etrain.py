# ==========================================================
# train.py
# Training Loop for the PINN Heat Equation
# ==========================================================

import torch
import time

start_time = time.time()
from Aconfig import (
    LEARNING_RATE,
    EPOCHS
)

from Dloss import total_loss


def train(
    model,
    x_i,
    t_i,
    u_i,
    x_b,
    t_b,
    u_b,
    x_f,
    t_f
):

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()  # Set the model to training mode

    # --------------------------------------------------
    # History
    # --------------------------------------------------

    history = {

        "total": [],
        "ic": [],
        "bc": [],
        "pde": []

    }

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        (
            loss,
            loss_ic,
            loss_bc,
            loss_pde

        ) = total_loss(

            model,

            x_i,
            t_i,
            u_i,

            x_b,
            t_b,
            u_b,

            x_f,
            t_f

        )

        loss.backward()

        optimizer.step()

        # --------------------------------------------
        # Save losses
        # --------------------------------------------

        history["total"].append(loss.item())

        history["ic"].append(loss_ic.item())

        history["bc"].append(loss_bc.item())

        history["pde"].append(loss_pde.item())

        # --------------------------------------------
        # Print progress
        # --------------------------------------------

        if epoch % 1000 == 0:

            print(
                f"Epoch {epoch:6d} | "
                f"Total = {loss.item():.6e} | "
                f"IC = {loss_ic.item():.6e} | "
                f"BC = {loss_bc.item():.6e} | "
                f"PDE = {loss_pde.item():.6e}"
            )

    end_time = time.time()
    print(f"Training time: {(end_time - start_time)/60:.2f} Minutes")

    return history