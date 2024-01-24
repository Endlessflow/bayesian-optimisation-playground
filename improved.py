import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from skopt import Optimizer

# Define the objective function (distance from target wave)
def objective_function(wave):
    distances = [np.linalg.norm(target_wave - w) for w in wave]
    return distances

# Generate a random target sine wave
x_values = np.linspace(0, 2*np.pi, 100)
target_wave = np.sin(x_values + np.random.uniform(-0.5, 0.5))

# Define the visualization function
def visualize_wave(x, y, target_wave):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns, adjust the figure size as needed
    axs = axs.ravel()  # Flattens the array for easy iterating
    
    for i, wave in enumerate(y):
        axs[i].plot(x, wave, label=f'Wave {i+1}')
        axs[i].plot(x, target_wave, label='Target Wave', color='black', linewidth=2.5)
        axs[i].legend()
    
    # Add a title for the entire figure (optional)
    plt.suptitle('Suggested waves and target wave')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Adjust the top padding to accommodate the suptitle
    plt.show()



# Function to get user input and handle errors
def get_user_input(prompt, min_val=1, max_val=4):
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Invalid input. Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Set up the optimization loop
num_iterations = 10
bounds = [(0.0, 2*np.pi)] * 100  # Bounds for the input configuration (wave)
optimizer = Optimizer(bounds, base_estimator=GaussianProcessRegressor(alpha=0.1,
                                                                      kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)),
                      acq_optimizer="sampling")

# Get an initial guess from the user
#initial_phase_shift = get_user_input("Provide an initial guess for the phase shift (between 0 and 2π): ", 0, 2*np.pi)
#initial_guess = [initial_phase_shift] * 100  # Create initial guess with consistent phase shift across all points
#optimizer.tell([initial_guess], [0.0])  # Initialize optimizer with the initial guess


for _ in range(num_iterations):
    # Suggest new configurations using Bayesian optimization
    suggested_waves = optimizer.ask(n_points=4)

    # Visualize the suggested waves
    visualize_wave(x_values, [np.sin(x_values + wave) for wave in suggested_waves], target_wave)

    # Gather user preferences (rankings)
    rankings = [0] * 4
    for i in range(4):
        wave_ranked = int(input(f"Enter the wave number that is ranked {i+1} (1-4): ")) - 1
        rankings[wave_ranked] = i

    # Create a list of distances based on the user rankings
    distances = rankings


    # Create a list of distances based on the user rankings
    distances = [0.0] * 4
    for i, rank in enumerate(rankings):
        distances[rank - 1] = i

    # Provide feedback to the optimizer
    optimizer.tell(suggested_waves, distances)
