# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np

from libc.math cimport sqrt
from cython.parallel import prange

ctypedef np.float64_t DTYPE_t


cpdef np.ndarray simulate_verlet(
    np.ndarray[DTYPE_t, ndim=2] r0_np,
    np.ndarray[DTYPE_t, ndim=2] v0_np,
    np.ndarray[DTYPE_t, ndim=1] m_np,
    double G,
    int num_steps,
    double dt
):
    cdef int N = r0_np.shape[0]
    cdef int D = r0_np.shape[1]

    cdef int n, i, j
    cdef double dx, dy, dist2, dist, dist3, factor

    # Выходные массивы
    cdef np.ndarray[DTYPE_t, ndim=3] r_np = np.zeros((num_steps, N, D), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=3] v_np = np.zeros((num_steps, N, D), dtype=np.float64)

    # Ускорения на текущем и следующем шаге
    cdef np.ndarray[DTYPE_t, ndim=2] a_np = np.zeros((N, D), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] a_new_np = np.zeros((N, D), dtype=np.float64)

    # Memoryviews скорость + типа плюсов
    cdef double[:, :, :] r = r_np
    cdef double[:, :, :] v = v_np
    cdef double[:, :] a = a_np
    cdef double[:, :] a_new = a_new_np
    cdef double[:, :] r0 = r0_np
    cdef double[:, :] v0 = v0_np
    cdef double[:] m = m_np

    # Инициализация
    for i in range(N):
        for j in range(D):
            r[0, i, j] = r0[i, j]
            v[0, i, j] = v0[i, j]

    # Начальное ускорение a(r[0])
    for i in prange(N, nogil=True):
        a[i, 0] = 0.0
        a[i, 1] = 0.0

        for j in range(N):
            if i != j:
                dx = r[0, j, 0] - r[0, i, 0]
                dy = r[0, j, 1] - r[0, i, 1]

                dist2 = dx * dx + dy * dy
                dist = sqrt(dist2)
                dist3 = dist2 * dist # |r1 - r2|^3 

                factor = G * m[j] / dist3

                a[i, 0] += factor * dx # x составляющая
                a[i, 1] += factor * dy # y составляющая 

    # Основной цикл метода Верле
    for n in range(num_steps - 1):

        for i in prange(N, nogil=True):
            r[n + 1, i, 0] = r[n, i, 0] + v[n, i, 0] * dt + 0.5 * a[i, 0] * dt * dt
            r[n + 1, i, 1] = r[n, i, 1] + v[n, i, 1] * dt + 0.5 * a[i, 1] * dt * dt

        for i in prange(N, nogil=True):
            a_new[i, 0] = 0.0
            a_new[i, 1] = 0.0

            for j in range(N):
                if i != j:
                    dx = r[n + 1, j, 0] - r[n + 1, i, 0]
                    dy = r[n + 1, j, 1] - r[n + 1, i, 1]

                    dist2 = dx * dx + dy * dy
                    dist = sqrt(dist2)
                    dist3 = dist2 * dist

                    factor = G * m[j] / dist3

                    a_new[i, 0] += factor * dx
                    a_new[i, 1] += factor * dy

        for i in prange(N, nogil=True):
            v[n + 1, i, 0] = v[n, i, 0] + 0.5 * (a[i, 0] + a_new[i, 0]) * dt
            v[n + 1, i, 1] = v[n, i, 1] + 0.5 * (a[i, 1] + a_new[i, 1]) * dt

        for i in prange(N, nogil=True):
            a[i, 0] = a_new[i, 0]
            a[i, 1] = a_new[i, 1]

    return r_np