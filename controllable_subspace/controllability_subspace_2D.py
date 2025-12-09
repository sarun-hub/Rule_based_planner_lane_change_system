import numpy as np
import matplotlib.pyplot as plt

# Parameter
h = 1

# Define two 2D vectors
v1 = np.array([1, -1/h])
v2 = np.array([0, 1])

# Kernel
ker = np.array([h,1])

# Create a grid of scalars to span the area
s = np.linspace(-10, 100, 1000)
t = np.linspace(-10, 100, 1000)
S, T = np.meshgrid(s, t)

# Compute the linear combinations of v1 and v2
X = S * v1[0] + T * v2[0]
Y = S * v1[1] + T * v2[1]

# Plot setup
fig, ax = plt.subplots()

# Fill the span area
ax.fill(X.flatten(), Y.flatten(), color='lightgray', alpha=0.5, label='Span(v1, v2)', edgecolor='gray')

# Fill the Kernel Vector
ker_span = np.linspace(-10,10,1000)
KER_X = ker_span * ker[0]
KER_Y = ker_span * ker[1]

# Draw the vectors
origin = [5, 0]
# ax.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='r', label='v1')
# ax.quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1, color='b', label='v2')
ax.quiver(*origin, *(KER_X[-1],KER_Y[-1]), angles='xy', scale_units='xy', scale=1, color='g', label='Kernel of G_c')

# Axes and grid
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(5, 50)
ax.set_ylim(-4, 4)
# ax.set_aspect('equal')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Relative velocity (m/s)')
ax.legend()
ax.set_title('Span of Two 2D Vectors')

plt.grid(True)
plt.show()
