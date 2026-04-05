import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


D = 2
G = 6.67 * 1e-11

names = [
    "Sun", "Mercury", "Venus", "Earth", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
]

# -------------------- массы (кг) --------------------
m = np.array([
    1.989e30,   # Sun
    3.285e23,   # Mercury
    4.867e24,   # Venus
    5.972e24,   # Earth
    6.39e23,    # Mars
    1.898e27,   # Jupiter
    5.683e26,   # Saturn
    8.681e25,   # Uranus
    1.024e26,   # Neptune
    1.309e22    # Pluto
])

# -------------------- начальные координаты (м) --------------------
r0 = np.array([
    [0.0, 0.0],          # Sun
    [57.91e9, 0.0],      # Mercury
    [108.21e9, 0.0],     # Venus
    [149.60e9, 0.0],     # Earth
    [227.92e9, 0.0],     # Mars
    [778.57e9, 0.0],     # Jupiter
    [1433.53e9, 0.0],    # Saturn
    [2872.46e9, 0.0],    # Uranus
    [4495.06e9, 0.0],    # Neptune
    [5906.38e9, 0.0]     # Pluto
])

# -------------------- начальные скорости (м/с) --------------------
v0 = np.array([
    [0.0, 0.0],          # Sun 
    [0.0, 47870.0],      # Mercury
    [0.0, 35020.0],      # Venus
    [0.0, 29780.0],      # Earth
    [0.0, 24077.0],      # Mars
    [0.0, 13070.0],      # Jupiter
    [0.0, 9690.0],       # Saturn
    [0.0, 6810.0],       # Uranus
    [0.0, 5430.0],       # Neptune
    [0.0, 4740.0]        # Pluto
])

N = m.shape[0]
# Временная сетка
t0 = 0.0
t_end = 100 * 24 * 3600 * 100
dt = (t_end - t0) / 2000

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
fig, ax = plt.subplots(figsize=(10, 10))
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

ani = FuncAnimation(fig, update, frames=num_steps, interval=10, blit=True)
plt.show()