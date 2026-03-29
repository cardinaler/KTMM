import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = 2          
D = 2         
G = 1.0
m = np.ones(N) 

r0 = np.array([
    [-1.0, 0.0],
    [ 1.0, 0.0]
])

v0 = np.array([
    [0.0, -0.5],
    [0.0,  0.5]
])

# Временная сетка
t0 = 0.0
t_end = 20.0
dt = 0.04

t = np.arange(t0, t_end + dt, dt)
num_steps = len(t)


def acceleration(r):
    """
    r:  (N, D)
    return a: (N, D)
    """
    a = r - r.reshape(r.shape[0], 1, r.shape[1])
    d = np.linalg.norm(a, axis=2)
    a = a / d[:, :, np.newaxis]**3
    a[np.isnan(a)] = 0
    a = G * np.sum(a * m[np.newaxis, :, np.newaxis], axis=1)

    return a


r = np.zeros((num_steps, N, D))
v = np.zeros((num_steps, N, D))

r[0] = r0
v[0] = v0

a = acceleration(r[0])


# Метод Верле 

for n in range(num_steps - 1):
    r[n + 1] = r[n] + v[n] * dt + 0.5 * a * dt**2

    a_new = acceleration(r[n + 1])

    v[n + 1] = v[n] + 0.5 * (a + a_new) * dt

    a = a_new

# Визуализация траекторий

positions = r
fig, ax = plt.subplots(figsize=(6, 6))
ax.grid()

all_x = positions[:, :, 0]
all_y = positions[:, :, 1]
margin = 0.5
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

scat = ax.scatter([], [], s=100)

def update(frame):
    scat.set_offsets(positions[frame])

    return scat,

ani = FuncAnimation(fig, update, frames=num_steps, interval=30, blit=True)
plt.show()