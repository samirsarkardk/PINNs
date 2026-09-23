# ==========================================================
# main.py
# Main Program
# ==========================================================

import torch
import numpy as np

from Aconfig import (
    DEVICE,
    MODEL_PATH,
    SEED
)

from Cmodel import PINN

from Bdata import (
    generate_initial_points,
    generate_boundary_points,
    generate_collocation_points
)

from Etrain import train

from Gpredict import (
    predict,
    exact_solution
)

from Hplots import (
    plot_loss,
    plot_prediction
)

from utils import (
    save_model,
    set_seed,
    count_parameters
)


# ==========================================================
# Main Function
# ==========================================================

def main():

    # ------------------------------------------------------
    # Random Seed
    # ------------------------------------------------------

    set_seed(SEED)

    # ------------------------------------------------------
    # Build Model
    # ------------------------------------------------------

    model = PINN().to(DEVICE)

    print(model)

    print(f"\nTrainable Parameters : {count_parameters(model)}")


    # ------------------------------------------------------
    # Generate Training Data
    # ------------------------------------------------------

    x_i, t_i, u_i = generate_initial_points()

    x_b, t_b, u_b = generate_boundary_points()

    x_f, t_f = generate_collocation_points()


    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    history = train(

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


    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    save_model(
        model,
        MODEL_PATH
    )



    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    x, u_exact, u_pred = predict(
    model,
    t=0.25,
    num_points=5000
)

# ------------------------------------------------------
# Relative L2 Error
# ------------------------------------------------------

    relative_l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

    print(f"\nRelative L2 Error : {relative_l2_error:.6e}")

    print(f"Relative L2 Error Percentage : {relative_l2_error * 100:.4f}%")

   

    # ------------------------------------------------------
    # Plots
    # ------------------------------------------------------

    plot_loss(history)

    plot_prediction(
    x,
    u_exact,
    u_pred
)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()