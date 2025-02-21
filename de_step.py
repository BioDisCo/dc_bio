import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

show_particle_nr = 2
show_iter = 1

# Define a function with two parameters
def f(x, y):
    return np.sin(np.pi*x) * np.cos(np.pi*y) + (x**2 + y**2)  # A function with local minima

# Wrapper function for differential evolution

def func_to_minimize(vars):
    x, y = vars
    return f(x, y)

# Store iteration history
history = []
epoch = 0

def callback(xk, convergence):
    pass
    # history.append(tuple(xk))

def custom_strategy_fn(candidate: int, population: np.ndarray, rng=None):
    """
    candidate: the candidate nr that is being evolved
    population: (population_size, parameter_size) state of the complete population
    """
    global epoch, history

    # update epoch
    if candidate == 0:
        epoch += 1

    # note in history
    history.append(np.copy(population.transpose()))

    # do not propose anymore solutions after the candidate to show
    if candidate > show_particle_nr and epoch >= show_iter:
        return np.array([1000,1000])  # suboptimal one

    parameter_count = population.shape[-1]
    print("-- population size", population.shape[0], "candidate", candidate, "--")
    # mutation, recombination = 0.7, 0.9
    mutation, recombination = 1.0, 1.0
    
    # evolve the candidate
    trial = np.copy(population[candidate])
    print("trial", trial)

    # choose a parameter dimension that will be always replaced
    fill_point = rng.choice(parameter_count)
    print("fill_point = dimension that will be mutated", fill_point)

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
    print("r0, r1, r2", r0, r1, r2)

    if epoch == show_iter and candidate == show_particle_nr:
        plt.plot([population[r0][0].copy()], [population[r0][1].copy()], color='black', marker='o', markersize=20)
        # plt.plot([population[r1][0].copy()], [population[r1][1].copy()], color='lightblue', marker='o', markersize=20)
        # plt.plot([population[r2][0].copy()], [population[r2][1].copy()], color='blue', marker='o', markersize=20)
        plt.arrow(population[r2][0].copy(),
                  population[r2][1].copy(),
                  population[r1][0].copy() - population[r2][0].copy(),
                  population[r1][1].copy() - population[r2][1].copy(),
                  color='blue',
                  length_includes_head=True,
                  head_width=.2, head_length=0.2
                  )

    bprime = (population[r0] + mutation *
              (population[r1] - population[r2]))

    # for each parameter pick a uniform rnd number
    crossovers = rng.uniform(size=parameter_count)

    # check if this rnd value is < the recombination constant
    crossovers = crossovers < recombination

    # also one of them is the fill_point parameter that is always replaced
    # -> set it also to True
    crossovers[fill_point] = True
    
    # update trial
    trial = np.copy(np.where(crossovers, bprime, trial))

    # show the update
    if candidate == show_particle_nr and epoch == show_iter:
        plt.plot([trial[0].copy()], [trial[1].copy()], color='red', marker='x', markersize=24)

    return trial

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
print(f"Minimum value of f(x, y) is {global_min_value} at coordinates ({global_min_x}, {global_min_y})")

# Plot function landscape and optimization trajectory
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x, y)')

# Perform Differential Evolution
bounds = [(-2, 2), (-2, 2)]  # Bounds for x and y
result = differential_evolution(func_to_minimize,
                                bounds, callback=callback,
                                strategy=custom_strategy_fn,
                                init="random",
                                popsize=5,
                                seed=42,
                                maxiter=show_iter,
                                polish=False)
min_x, min_y = result.x
min_value = result.fun

# Plot trajectories
history_x = [point[0] for point in history]
history_y = [point[1] for point in history]

for pn in range(10):
    p_x = [ history_x[t][pn] for t in range(len(history)) ]
    p_y = [ history_y[t][pn] for t in range(len(history)) ]
    col = '#ffcccc' if pn != show_particle_nr else 'red'
    plt.plot(p_x, p_y, color=col, marker='o', linestyle='dashed', label='Evolution trajectory')

plt.plot([global_min_x], [global_min_y], color='orange', marker='o')

plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
# plt.title('Differential Evolution on f(x, y) with Trajectories')
plt.savefig("de_step.pdf",
            bbox_inches='tight', 
            transparent=True,
            pad_inches=0)