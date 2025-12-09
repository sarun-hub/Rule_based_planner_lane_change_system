import numpy as np
import matplotlib.pyplot as plt

# Parameter
h = 1

# Define vector for spanning
v1 = np.array([0,1,0])
v2 = np.array([1,0,1/h])

# Kernel
ker = np.array([h,1,0])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# Plot the vectors2
origin = np.zeros(3)
ax.quiver(*origin, *v1, color='r',label = 'v1')
ax.quiver(*origin, *v2, color='b',label = 'v2')
ax.quiver(*origin, *ker, color = 'g')

# Plot the span (plane) of the two vectors
s = np.linspace(-3,3,10)
t = np.linspace(-3,3,10)
S,T = np.meshgrid(s,t)

X = S * v1[0] + T * v2[0]
Y = S * v1[1] + T * v2[1]
Z = S * v1[2] + T * v2[2]

ker_span = np.linspace(-2,2,20)
KER_X = ker_span * ker[0] 
KER_Y = ker_span * ker[1] 
KER_Z = ker_span * ker[2]

ax.plot(KER_X,KER_Y,KER_Z,color = 'g', label = 'Kernel of G_c')

ax.plot_surface(X,Y,Z, alpha=0.3, color = 'blue')

# Add coordinate axes centered at origin
axis_length = 2
ax.quiver(0, 0, 0, axis_length, 0, 0, color='black', linewidth=1)
ax.quiver(0, 0, 0, 0, axis_length, 0, color='black', linewidth=1)
ax.quiver(0, 0, 0, 0, 0, axis_length, color='black', linewidth=1)

# Add axis labels at ends
ax.text(axis_length, 0, 0, 'd', color='black', fontsize=12)
ax.text(0, axis_length, 0, 'vp', color='black', fontsize=12)
ax.text(0, 0, axis_length, 'vf', color='black', fontsize=12)

# Set plot limit
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])
ax.set_xlabel('d')
ax.set_ylabel('vp')
ax.set_zlabel('vf')
ax.legend()
ax.set_title('Img of Gc')

plt.show()