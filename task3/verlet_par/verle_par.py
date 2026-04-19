import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count

class verle_par:

    G = 6.67 * 1e-11

# Функция для одного тела
    def acceleration_for_body(self, args):
        i = args
        N = self.r.shape[0]
        D = 2
        ai = np.zeros(D)

        for j in range(N):
            if i != j:
                a = self.r[j] - self.r[i]
                d = np.linalg.norm(a)
                ai += self.G * self.m[j] * a / d**3

        return ai

# Параллельное вычисление ускорений
    def acceleration_parallel(self, r, m):
        self.r = r
        self.m = m
        N = r.shape[0]
        args = [i for i in range(N)]

        with Pool(processes=cpu_count()) as pool:
            a = pool.map(self.acceleration_for_body, args)

        return np.array(a)


    def solve(self, t0, t_end, num_steps, m, r0, v0):
        N = m.shape[0]
        D = 2
        dt = (t_end - t0) / num_steps
        t = np.arange(t0, t_end + dt, dt)

        r = np.zeros((num_steps, N, D))
        v = np.zeros((num_steps, N, D))

        r[0] = r0
        v[0] = v0

        a = self.acceleration_parallel(r[0], m)

        # Метод Верле
        for n in range(num_steps - 1):
            r[n + 1] = r[n] + v[n] * dt + 0.5 * a * dt**2

            a_new = self.acceleration_parallel(r[n + 1], m)

            v[n + 1] = v[n] + 0.5 * (a + a_new) * dt

            a = a_new
        
        return r
