import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from verlet_par.verle_par import verle_par
import time

N_ = [100, 200, 300]
time_exec = list()
rng = np.random.default_rng(42)
for i in range(len(N_)):
    N = N_[i]
    m = rng.uniform(0, 1.0, size=N)
    r0 = rng.uniform(-10.0, 10.0, size=(N, 2))
    v0 = rng.uniform(-1.0, 1.0, size=(N, 2))

    num_steps = 2000
    t0 = 0.0
    t_end = 100 # 100 секунд
    N = m.shape[0]
    solver = verle_par()
    t1 = time.perf_counter()
    solver.solve(t0, t_end, num_steps, m, r0, v0)
    t2 = time.perf_counter()
    time_exec.append(t2 - t1)


plt.bar(N_, time_exec)
plt.show()