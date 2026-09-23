# ==========================================================
# loss.py
# Loss Functions for the PINN Heat Equation
# ==========================================================

import torch
import torch.nn as nn

mse_loss = nn.MSELoss()


# ==========================================================
# Initial Condition Loss
# ==========================================================

def initial_condition_loss(model, x_i, t_i, u_i):

    u_pred = model(x_i, t_i)

    loss = mse_loss(u_pred, u_i)

    return loss


# ==========================================================
# Boundary Condition Loss
# ==========================================================

def boundary_condition_loss(model, x_b, t_b, u_b):

    u_pred = model(x_b, t_b)

    loss = mse_loss(u_pred, u_b)

    return loss


# ==========================================================
# PDE Residual Loss
# ==========================================================

def pde_residual_loss(model, x_f, t_f):

    x_f.requires_grad_(True)

    t_f.requires_grad_(True)


    u = model(x_f, t_f)


    # ----------------------------------------
    # First derivatives
    # ----------------------------------------

    u_t = torch.autograd.grad(
        u,
        t_f,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]


    u_x = torch.autograd.grad(
        u,
        x_f,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]


    # ----------------------------------------
    # Second derivative
    # ----------------------------------------

    u_xx = torch.autograd.grad(
        u_x,
        x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]


    # ----------------------------------------
    # Heat Equation Residual
    # u_t - u_xx = 0
    # ----------------------------------------

    residual = u_t - u_xx

    loss = mse_loss(
        residual,
        torch.zeros_like(residual)
    )

    return loss


# ==========================================================
# Total Loss
# ==========================================================

def total_loss(
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

    loss_ic = initial_condition_loss(
        model,
        x_i,
        t_i,
        u_i
    )

    loss_bc = boundary_condition_loss(
        model,
        x_b,
        t_b,
        u_b
    )

    loss_pde = pde_residual_loss(
        model,
        x_f,
        t_f
    )

    loss = loss_ic + loss_bc + loss_pde

    return (
        loss,
        loss_ic,
        loss_bc,
        loss_pde
    )