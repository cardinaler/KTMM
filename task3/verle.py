import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class verle:
    G = 6.67 * 1e-11

    def __init__(self):

        pass
        # Временная сетка
        # num_steps = 2000
        # t0 = 0.0
        # t_end = 100 * 24 * 3600 * 100
        



    def acceleration(self, r, m):
        """
        r:  (N, D)
        return a: (N, D)
        """
        a = r - r.reshape(r.shape[0], 1, r.shape[1])
        d = np.linalg.norm(a, axis=2)
        a = a / d[:, :, np.newaxis]**3
        a[np.isnan(a)] = 0
        a = self.G * np.sum(a * m[np.newaxis, :, np.newaxis], axis=1)

        return a

    def solve(self, t0, t_end, num_steps, m, r0, v0):
        N = m.shape[0]
        D = 2
        dt = (t_end - t0) / num_steps
        t = np.arange(t0, t_end + dt, dt)

        r = np.zeros((num_steps, N, D))
        v = np.zeros((num_steps, N, D))

        r[0] = r0
        v[0] = v0

        a = self.acceleration(r[0], m)


        # Метод Верле 

        for n in range(num_steps - 1):
            r[n + 1] = r[n] + v[n] * dt + 0.5 * a * dt**2

            a_new = self.acceleration(r[n + 1], m)

            v[n + 1] = v[n] + 0.5 * (a + a_new) * dt

            a = a_new
        
        return r

