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
num_iterations = 0
max_iterations = 500  # Set a maximum to avoid infinite loops
bounds = [(0.0, 2*np.pi)] * 100  # Bounds for the input configuration (wave)
optimizer = Optimizer(bounds, base_estimator=GaussianProcessRegressor(alpha=0.1,
                                                                      kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)),
                      acq_optimizer="sampling")

improvement_threshold = 1e-3  # Threshold for minimum improvement
last_best_distance = np.inf  # Initialize best distance to infinity

while num_iterations < max_iterations:
    # Suggest new configurations using Bayesian optimization
    suggested_waves = optimizer.ask(n_points=4)

    # Generate the suggested waves
    suggested_waves_sin = [np.sin(x_values + wave) for wave in suggested_waves]

    # Visualize the suggested waves every 10 iterations
    if num_iterations % 10 == 0:
        visualize_wave(x_values, suggested_waves_sin, target_wave)

    # Evaluate the objective function for each suggested wave
    distances = objective_function(suggested_waves_sin)

    # Rank the suggested waves based on the distances
    rankings = list(np.argsort(distances))

    # Provide feedback to the optimizer
    optimizer.tell(suggested_waves, rankings)

    # Check for convergence: if the best distance hasn't improved by at least the threshold, break the loop
    best_distance = min(distances)
    if abs(last_best_distance - best_distance) < improvement_threshold:
        print(f"Convergence reached after {num_iterations} iterations.")
        break
    else:
        last_best_distance = best_distance

    num_iterations += 1

# Visualize final result
best_wave_index = np.argmin(distances)
best_wave = suggested_waves_sin[best_wave_index]
visualize_wave(x_values, [best_wave], target_wave)

print(f"Best wave found after {num_iterations} iterations.")


