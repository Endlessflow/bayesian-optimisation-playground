import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from skopt import Optimizer
import matplotlib.pyplot as plt

# Define the objective function (our "unknown" function)
def objective_function(x):
    noise = np.random.normal(loc=0, scale=0.1)  # Some noise
    return np.sin(5 * x) + (x ** 2) - 0.2 * x + noise

# Create the bayesian optimizer
bounds = [(-2.0, 2.0)]  # We know our function is defined in the range -2 <= x <= 2
optimizer = Optimizer(bounds, base_estimator=GaussianProcessRegressor(alpha=0.1,
                                                                      kernel=ConstantKernel(1.0) * Matern(length_scale=0.5)),
                                                                      acq_optimizer="sampling", acq_func="EI")

# Initial random points
n_initial_points = 5
initial_points = np.random.uniform(-2, 2, (n_initial_points, 1))
initial_values = [objective_function(x[0]) for x in initial_points]

# Tell the optimizer about these points
for x, y in zip(initial_points, initial_values):
    optimizer.tell(x.tolist(), y)

# Now we let the optimizer propose points and we evaluate them
n_iterations = 300
for i in range(n_iterations):
    x = optimizer.ask()
    y = objective_function(x[0])
    optimizer.tell(x, y)

    print(f"Iteration {i+1}, x: {x[0]}, y: {y}")

# Let's visualize the optimization process
plt.figure(figsize=(10, 5))
plt.title('Bayesian Optimization Process')
plt.xlabel('x')
plt.ylabel('f(x)')

# True function
x_values = np.linspace(-2, 2, 400).reshape(-1, 1)
y_values = [np.sin(5 * x) + (x ** 2) - 0.2 * x for x in x_values]
plt.plot(x_values, y_values, 'r--', label='True function')

# Evaluated points
evaluated_points = optimizer.Xi
evaluated_values = optimizer.yi
plt.scatter(evaluated_points, evaluated_values, color='b', label='Evaluated points')

plt.legend()
plt.show()

# Let's see the final result
# Calculate true minimum
x_values = np.linspace(-2, 2, 1000)
y_values = [objective_function(x) for x in x_values]
true_min_y = min(y_values)
true_min_x = x_values[y_values.index(true_min_y)]

print(f"True minimum value: {true_min_y} at x = {true_min_x}")

# Calculate minimum found by optimizer
found_min_y = min(optimizer.yi)
found_min_x = optimizer.Xi[optimizer.yi.index(min(optimizer.yi))]

print(f"Found minimum value: {found_min_y} at x = {found_min_x}")

# Calculate error
error_y = abs(true_min_y - found_min_y)
error_x = abs(true_min_x - found_min_x)

print(f"Error in y: {error_y}")
print(f"Error in x: {error_x}")
