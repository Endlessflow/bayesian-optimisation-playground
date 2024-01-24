import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from skopt import Optimizer
from sklearn.metrics.pairwise import cosine_similarity

# Generate a random target sine wave
n_points = 20  # reduce dimensionality
x_values = np.linspace(0, 2*np.pi, n_points)
target_wave = np.sin(x_values + np.random.uniform(-0.5, 0.5))

# Define the objective function (cosine similarity from target wave)
def objective_function(wave, target_wave=target_wave):
    wave = wave.reshape(1, -1)
    target_wave = target_wave.reshape(1, -1)
    similarity = cosine_similarity(target_wave, wave)
    return -similarity[0][0]  # maximize similarity

# Define the visualization function
def visualize_wave(x, y, target_wave):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.plot(x, y[i], label=f'Wave {i+1}')
        ax.plot(x, target_wave, label='Target Wave')
        ax.legend()
    plt.show()


bounds = [(0.0, 2*np.pi)] * n_points  # Bounds for the input configuration (wave)

# Use Matern kernel and set acquisition function as "EI"
optimizer = Optimizer(bounds, base_estimator=GaussianProcessRegressor(alpha=0.1,
                                                                      kernel=ConstantKernel(1.0) * Matern(length_scale=0.5)),
                                                                      acq_optimizer="sampling", acq_func="EI")

# Provide initial points
initial_points = np.random.uniform(0, 2*np.pi, size=(10, n_points))  # 10 initial points
initial_values = [objective_function(p) for p in initial_points]
optimizer.tell(initial_points.tolist(), initial_values)

# Set up the optimization loop
num_iterations = 1000
epsilon = 0.1  # Probability of choosing a random wave
tolerance = 1e-3  # Convergence tolerance

best_distance = np.inf

#...
for i in range(num_iterations):
    # Suggest new configurations using Bayesian optimization
    suggested_waves = optimizer.ask(n_points=4)

    if np.random.rand() < epsilon != 0:
        # Replace a random wave in the set of suggested waves with a completely random wave
        idx = np.random.randint(0, len(suggested_waves))
        suggested_waves[idx] = np.random.uniform(0, 2*np.pi, size=n_points)
        print(f'Random Wave Generated on iteration {i}')

    # Compute distances for each suggested wave and choose the best one
    distances = np.array([objective_function(np.sin(x_values + wave)) for wave in suggested_waves])
    chosen_wave = suggested_waves[np.argmin(distances)]
    chosen_value = objective_function(np.sin(x_values + chosen_wave))

    # Visualize the suggested waves every 10 iterations
    if i % 100 == 0:
        visualize_wave(x_values, [np.sin(x_values + wave) for wave in suggested_waves], target_wave)

    # Provide feedback to the optimizer
    optimizer.tell([chosen_wave.tolist()], [chosen_value])  # Pass chosen_value as a list

    # Print the best distance so far
    if np.min(distances) < best_distance:
        best_distance = np.min(distances)
        print(f'Iteration {i}, Best Distance So Far: {best_distance}')

    print(f"Iteration: {i+1}")


# Visualize final suggested waves
suggested_waves = optimizer.ask(n_points=4)
visualize_wave(x_values, [np.sin(x_values + wave)
                          for wave in suggested_waves], target_wave)
