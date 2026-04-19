import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class solve_with_ivp:


    G = 6.67 * 1e-11

    def __init__(self, N, D):
        self.N = N
        self.D = D

    def solve(self, t0, t_end, num_steps, m, r0, v0):
        N = m.shape[0]
        D = 2

        def system(t, y):
            # первые N - r, остальные N - v
            D = 2
            N = y.shape[0] // (2 * D)
            drdt = y[N*D:]
            r = y[:N*D]
            r = r.reshape(N, D)
            v = r - r.reshape(r.shape[0], 1, r.shape[1])
            d = np.linalg.norm(v, axis=2)
            v = v / d[:, :, np.newaxis]**3
            v[np.isnan(v)] = 0
            dvdt = self.G * np.sum(v * m[np.newaxis, :, np.newaxis], axis=1)
            dvdt = dvdt.reshape(-1)
            return np.concatenate((drdt, dvdt))
        
        # Собираем в один вектор состояния
        y_0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))

        t_span = (t0, t_end)
        t_eval = np.linspace(t_span[0], t_span[1], num_steps)

        sol = solve_ivp(system, t_span, y_0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

        # Визуализация траекторий
        return sol.y[:N * D].reshape(N, D, -1).transpose(2, 0, 1)

        
