import numpy as np
import pyopencl as cl


class verle_opencl:
    G = 6.67e-11

    def __init__(self):
        # контекст
        platforms = cl.get_platforms()
        platform = platforms[0]  # rusticl

        devices = platform.get_devices()
        device = devices[0]      

        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        # OpenCL kernel
        self.program = cl.Program(self.ctx, """
            __kernel void compute_acc(
                const int N,
                const float G,
                __global const float *r,
                __global const float *m,
                __global float *a)
            {
                int i = get_global_id(0);

                float ax = 0.0f;
                float ay = 0.0f;

                float xi = r[2*i];
                float yi = r[2*i + 1];

                for (int j = 0; j < N; j++) {
                    if (i == j) continue;

                    float dx = r[2*j] - xi;
                    float dy = r[2*j + 1] - yi;

                    float dist2 = dx*dx + dy*dy + 1e-6f;
                    float inv_dist = 1.0f / sqrt(dist2);
                    float inv_dist3 = inv_dist * inv_dist * inv_dist;

                    ax += G * m[j] * dx * inv_dist3;
                    ay += G * m[j] * dy * inv_dist3;
                }

                a[2*i]     = ax;
                a[2*i + 1] = ay;
            }
            """).build()

    def acceleration(self, r, m):
        """
        r: (N, 2)
        m: (N,)
        """
        N = r.shape[0]

        # flatten
        r_flat = r.astype(np.float32).ravel()
        a_flat = np.zeros_like(r_flat)
        m = m.astype(np.float32)

        mf = cl.mem_flags

        r_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r_flat)
        m_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
        a_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, a_flat.nbytes)

        self.program.compute_acc(
            self.queue,
            (N,),
            None,
            np.int32(N),
            np.float32(self.G),
            r_buf,
            m_buf,
            a_buf
        )

        cl.enqueue_copy(self.queue, a_flat, a_buf)

        return a_flat.reshape(N, 2)

    def solve(self, t0, t_end, num_steps, m, r0, v0):
        N = m.shape[0]
        dt = (t_end - t0) / num_steps

        r = np.zeros((num_steps, N, 2))
        v = np.zeros((num_steps, N, 2))

        r[0] = r0
        v[0] = v0

        a = self.acceleration(r[0], m)

        for n in range(num_steps - 1):
            r[n + 1] = r[n] + v[n]*dt + 0.5*a*dt**2

            a_new = self.acceleration(r[n + 1], m)

            v[n + 1] = v[n] + 0.5*(a + a_new)*dt

            a = a_new

        return r