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
    for i, wave in enumerate(y):
        plt.plot(x, wave, label=f'Wave {i+1}')
    plt.plot(x, target_wave, label='Target Wave')
    plt.legend()
    plt.show()


# Set up the optimization loop
num_iterations = 10
bounds = [(0.0, 2*np.pi)] * 100  # Bounds for the input configuration (wave)
optimizer = Optimizer(bounds, base_estimator=GaussianProcessRegressor(alpha=0.1,
                                                                      kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)),
                      acq_optimizer="sampling")

for _ in range(num_iterations):
    # Suggest new configurations using Bayesian optimization
    suggested_waves = optimizer.ask(n_points=4)

    # Visualize the suggested waves
    visualize_wave(x_values, [np.sin(x_values + wave)
                   for wave in suggested_waves], target_wave)

    # Gather user preferences (rankings)
    rankings = [
        int(input(f"Rank the suggested wave {i+1} (1-4): ")) for i in range(4)]

    # Create a list of distances based on the user rankings
    distances = [0.0] * 4
    for i, rank in enumerate(rankings):
        distances[rank - 1] = i

    # Provide feedback to the optimizer
    optimizer.tell(suggested_waves, distances)
