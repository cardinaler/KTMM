import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

D = 2
G = 6.67 * 1e-11

names = [
    "Sun", "Mercury", "Venus", "Earth", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
]

colors = [
    "yellow",      # Sun
    "gray",        # Mercury
    "orange",      # Venus
    "blue",        # Earth
    "red",         # Mars
    "brown",       # Jupiter
    "gold",        # Saturn
    "lightblue",   # Uranus
    "darkblue",    # Neptune
    "purple"       # Pluto
]

# массы (кг)
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

# начальные координаты (м) (расстояние от Солнца)
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

# начальные скорости (м/с) 
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
# Собираем в один вектор состояния
y_0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))

def system(t, y):
    # первые N - r, остальные N - v
    drdt = y[N*D:]
    r = y[:N*D]
    r = r.reshape(N, D)
    v = r - r.reshape(r.shape[0], 1, r.shape[1])
    d = np.linalg.norm(v, axis=2)
    v = v / d[:, :, np.newaxis]**3
    v[np.isnan(v)] = 0
    dvdt = G * np.sum(v * m[np.newaxis, :, np.newaxis], axis=1)
    dvdt = dvdt.reshape(-1)
    return np.concatenate((drdt, dvdt))

t_span = (0, 100 * 24 * 3600 * 100)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

sol = solve_ivp(system, t_span, y_0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Визуализация траекторий
positions = sol.y[:N * D].reshape(N, D, -1).transpose(2, 0, 1)


fig, ax = plt.subplots(figsize=(10, 10))

ax.grid()

all_x = positions[:, :, 0]
all_y = positions[:, :, 1]

margin = 0.5
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], c=colors)
for i in range(N):
    ax.scatter([], [], c=colors[i], label=names[i])


def update(frame):
    scat.set_offsets(positions[frame])
    return scat,

ani = FuncAnimation(fig, update, frames=len(sol.t), interval=10, blit=True)
plt.legend()
plt.show()