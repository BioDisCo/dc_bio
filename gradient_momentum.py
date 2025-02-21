import numpy as np
import matplotlib.pyplot as plt

# Define a function with two parameters
def f(x, y):
    return np.sin(np.pi*x) * np.cos(np.pi*y) + (x**2 + y**2)

# Partial derivatives
def df_dx(x, y):
    return np.pi * np.cos(np.pi*x) * np.cos(np.pi*y) + 2*x

def df_dy(x, y):
    return -np.pi * np.sin(np.pi*x) * np.sin(np.pi*y) + 2*y

# Gradient descent with momentum implementation
def gradient_descent_momentum(learning_rate=0.05, max_iter=50, x_init=1.5, y_init=1.5, momentum=0.8):
    x, y = x_init, y_init  # Initial guess
    velocity_x, velocity_y = 0, 0  # Initialize velocities
    history = [(x, y)]  # Store descent path
    
    for _ in range(max_iter):
        grad_x = df_dx(x, y)  # Compute gradient
        grad_y = df_dy(x, y)
        
        # Update velocities with momentum
        velocity_x = momentum * velocity_x - learning_rate * grad_x
        velocity_y = momentum * velocity_y - learning_rate * grad_y
        
        # Update positions
        x += velocity_x
        y += velocity_y
        
        history.append((x, y))
        
        # Stop if gradient is small
        if np.sqrt(grad_x**2 + grad_y**2) < 1e-5:
            break
    
    return history

# Run gradient descent
history = gradient_descent_momentum()

# Plot function landscape and descent path
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Find the indices of the minimum value
min_index = np.unravel_index(np.argmin(Z), Z.shape)
min_x = X[min_index]
min_y = Y[min_index]
min_value = Z[min_index]
print(f"Minimum value of f(x, y) is {min_value} at coordinates ({min_x}, {min_y})")

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x, y)')

# Plot descent path
history_x = [point[0] for point in history]
history_y = [point[1] for point in history]
history_z = [f(x, y) for x, y in history]
plt.plot(history_x, history_y, color='red', marker='o', linestyle='dashed', label='Descent path')
plt.plot([min_x], [min_y], color='orange', marker='o')

plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on f(x, y)')
plt.savefig("gradient_momentum.pdf",
            bbox_inches='tight', 
            transparent=True,
            pad_inches=0)
# plt.show()
