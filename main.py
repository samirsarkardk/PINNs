# ==========================================================
# main.py
# Main Program
# ==========================================================

import torch
import numpy as np

from config import (
    DEVICE,
    MODEL_PATH,
    SEED
)

from model import PINN

from data import (
    generate_initial_points,
    generate_boundary_points,
    generate_collocation_points
)

from train import train

from predict import (
    predict,
    exact_solution
)

from plots import (
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