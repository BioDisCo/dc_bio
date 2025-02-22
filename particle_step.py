import numpy as np
import matplotlib.pyplot as plt

show_particle_nr = 6
show_iter = 1


# Define a function with two parameters
def f(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y) + (x**2 + y**2)


# Particle Swarm Optimization (PSO)
def particle_swarm_optimization(
    num_particles=10, max_iter=show_iter, w=1, c1=1.5, c2=1.5 / 2.5
):
    np.random.seed(42)  # For reproducibility
    particles = np.random.uniform(
        -2, 2, (num_particles, 2)
    )  # Initialize particle positions
    velocities = np.zeros_like(particles)  # Initialize velocities
    personal_best = particles.copy()
    personal_best_values = np.array([f(x, y) for x, y in particles])
    global_best = personal_best[np.argmin(personal_best_values)]
    global_best_value = np.min(personal_best_values)
    history = [particles.copy()]

    for iter in range(max_iter):
        for i in range(num_particles):
            if iter == max_iter - 1 and i == show_particle_nr:
                plt.plot(
                    [personal_best[i][0]],
                    [personal_best[i][1]],
                    color="lightblue",
                    marker="*",
                    markersize=24,
                )
                plt.plot(
                    [global_best[0]],
                    [global_best[1]],
                    color="blue",
                    marker="*",
                    markersize=24,
                )

            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - particles[i])
                + c2 * r2 * (global_best - particles[i])
            )
            particles[i] += velocities[i]

            value = f(particles[i, 0], particles[i, 1])
            if value < personal_best_values[i]:
                personal_best[i] = particles[i]
                personal_best_values[i] = value

            if value < global_best_value:
                global_best = particles[i]
                global_best_value = value

        history.append(particles.copy())

    return history


# Plot function landscape and swarm path
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(label="f(x, y)")

# Run Particle Swarm Optimization
history = particle_swarm_optimization()

# Find the indices of the minimum value
min_index = np.unravel_index(np.argmin(Z), Z.shape)
min_x = X[min_index]
min_y = Y[min_index]
min_value = Z[min_index]
print(f"Minimum value of f(x, y) is {min_value} at coordinates ({min_x}, {min_y})")

# Plot swarm path
for pn in range(len(history[0])):
    history_x = [history[i][pn][0] for i in range(len(history))]
    history_y = [history[i][pn][1] for i in range(len(history))]
    col = "#ffcccc" if pn != show_particle_nr else "red"
    plt.plot(history_x, history_y, color=col, marker="o", linestyle="dashed")
plt.plot([min_x], [min_y], color="orange", marker="o")

plt.xlabel("x")
plt.ylabel("y")
# plt.legend()
# plt.title('Particle Swarm Optimization on f(x, y)')
plt.savefig("particle_step.pdf", bbox_inches="tight", transparent=True, pad_inches=0)
# plt.show()
