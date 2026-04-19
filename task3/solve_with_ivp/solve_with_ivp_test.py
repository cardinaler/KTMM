import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from solve_with_ivp import solve_with_ivp

N = 100
rng = np.random.default_rng(42)
m = rng.uniform(0, 1.0, size=N)
r0 = rng.uniform(-10.0, 10.0, size=(N, 2))
v0 = rng.uniform(-1.0, 1.0, size=(N, 2))
# Визуализация траекторий


num_steps = 2000
t0 = 0.0
t_end = 100
D = 2

solver = solve_with_ivp(N, D)
positions = solver.solve(t0, t_end, num_steps, m, r0, v0)
fig, ax = plt.subplots(figsize=(10, 10))
ax.grid()

all_x = positions[:, :, 0]
all_y = positions[:, :, 1]
margin = 0.5
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

scat = ax.scatter(positions[0, :, 0], positions[0, :, 1])

N = m.shape[0]
for i in range(N):
    ax.scatter([], [])


def update(frame):
    scat.set_offsets(positions[frame])
    return scat,

ani = FuncAnimation(fig, update, frames=num_steps, interval=10, blit=True)
plt.show()