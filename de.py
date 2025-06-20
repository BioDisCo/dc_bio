import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


# Define a function with two parameters
def f(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y) + (
        x**2 + y**2
    )  # A function with local minima


# Wrapper function for differential evolution


def func_to_minimize(vars):
    x, y = vars
    return f(x, y)


# Store iteration history
history = []


def callback(xk, convergence):
    pass
    # history.append(tuple(xk))


def custom_strategy_fn(candidate: int, population: np.ndarray, rng=None):
    """
    candidate: the candidate nr that is being evolved
    population: (population_size, parameter_size) state of the complete population
    """
    # note in history
    if candidate == 0:
        history.append(population.transpose())
    # print(population)

    parameter_count = population.shape[-1]
    print("population size", population.shape[0], "candidate", candidate)
    mutation, recombination = 0.7, 0.9
    # mutation, recombination = 0.1, 1.0

    # evolve the candidate
    trial = np.copy(population[candidate])

    # choose a parameter dimension that will be always replaced
    fill_point = rng.choice(parameter_count)

    # random order of the population
    pool = np.arange(len(population))
    rng.shuffle(pool)

    # two unique random numbers that aren't the same, and
    # aren't equal to candidate.
    idxs = []
    while len(idxs) < 3 and len(pool) > 0:
        # pop from head
        idx = pool[0]
        pool = pool[1:]
        if idx != candidate:
            idxs.append(idx)
    r0, r1, r2 = idxs[:3]

    bprime = population[r0] + mutation * (population[r1] - population[r2])

    # for each parameter pick a uniform rnd number
    crossovers = rng.uniform(size=parameter_count)

    # check if this rnd value is < the recombination constant
    crossovers = crossovers < recombination

    # also one of them is the fill_point parameter that is always replaced
    # -> set it also to True
    crossovers[fill_point] = True

    # update the trial
    trial = np.where(crossovers, bprime, trial)
    return trial


# Perform Differential Evolution
bounds = [(-2, 2), (-2, 2)]  # Bounds for x and y
result = differential_evolution(
    func_to_minimize,
    bounds,
    callback=callback,
    strategy=custom_strategy_fn,
    init="random",
    popsize=5,
    seed=42,
    maxiter=10,
    polish=False,
)
min_x, min_y = result.x
min_value = result.fun

# Generate landscape
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Find the indices of the minimum value
min_index = np.unravel_index(np.argmin(Z), Z.shape)
global_min_x = X[min_index]
global_min_y = Y[min_index]
global_min_value = Z[min_index]
print(
    f"Minimum value of f(x, y) is {global_min_value} at coordinates ({global_min_x}, {global_min_y})"
)

# Plot function landscape and optimization trajectory
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
cbar = plt.colorbar()
cbar.set_label("f(x, y)", fontsize=16)
cbar.ax.tick_params(labelsize=14)

# Plot trajectories
history_x = [point[0] for point in history]
history_y = [point[1] for point in history]
plt.plot(
    history_x,
    history_y,
    color="red",
    marker="o",
    linestyle="dashed",
    label="Evolution trajectory",
)

plt.plot([global_min_x], [global_min_y], color="orange", marker="o")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
# plt.legend()
# plt.title('Differential Evolution on f(x, y) with Trajectories')
plt.savefig("de.pdf", bbox_inches="tight", transparent=True, pad_inches=0)
