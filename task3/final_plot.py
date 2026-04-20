import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from verle.verle import verle
from verle_opencl.verle_opencl import verle_opencl
from verle_cython_openmp import verlet
from solve_with_ivp.solve_with_ivp import solve_with_ivp
import time


N_ = [100, 200, 300]

rng = np.random.default_rng(42)
solvers = [verle_opencl(), verle()]
time_solver_exec = list()

for solver in solvers:
    time_exec = list()
    for i in range(len(N_)):
        N = N_[i]
        m = rng.uniform(0, 1.0, size=N)
        r0 = rng.uniform(-10.0, 10.0, size=(N, 2))
        v0 = rng.uniform(-1.0, 1.0, size=(N, 2))

        num_steps = 2000
        t0 = 0.0
        t_end = 100 # 100 секунд
        N = m.shape[0]
        t1 = time.perf_counter()
        solver.solve(t0, t_end, num_steps, m, r0, v0)
        t2 = time.perf_counter()
        time_exec.append(t2 - t1)

    time_solver_exec.append(time_exec)


G = 6.67e-11
time_exec = list()

for i in range(len(N_)):
    N = N_[i]
    m = rng.uniform(0, 1.0, size=N)
    r0 = rng.uniform(-10.0, 10.0, size=(N, 2))
    v0 = rng.uniform(-1.0, 1.0, size=(N, 2))

    num_steps = 2000
    t0 = 0.0
    t_end = 100 # 100 секунд
    dt = (t_end - t0) / num_steps
    N = m.shape[0]
    t1 = time.perf_counter()
    verlet.simulate_verlet(r0, v0, m, G, num_steps, dt)
    t2 = time.perf_counter()
    time_exec.append(t2 - t1)

time_solver_exec.append(time_exec)


plt.plot(N_, time_solver_exec[0], label = "verlet_opencl")
plt.plot(N_, time_solver_exec[1], label = "verlet")
plt.plot(N_, time_solver_exec[2], label = "verlet_cython")

plt.xlabel("N")
plt.ylabel("c")
plt.legend()
plt.grid(True)
plt.show()