import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
N = 2
D = 2
G = 1
# начальные данные
y_0 = np.array([
    -1.0, 0.0, #r1
     1.0, 0.0, #r2
     0.0, -0.5,#v1
     0.0,  0.5 #v2
])

m = np.ones(N)

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
    
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol = solve_ivp(system, t_span, y_0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Визуализация траекторий
positions = sol.y[:N * D].reshape(N, D, -1).transpose(2, 0, 1)


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

ani = FuncAnimation(fig, update, frames=len(sol.t), interval=30, blit=True)
plt.show()