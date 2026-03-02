import elems, json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
class ode_sys_solver:
    c : np.array
    lmbda : np.array
    eps : np.array
    el_surf_cross : np.array
    el_surf : np.array
    c_0 = 5.67
    func_list : str
    T_0 : np.array
    t_span : tuple
    time_edge : np.float64

    def load_obj(self, obj_filename):
        d = elems.device(obj_filename)
        self.el_surf_cross = d.el_surf_cross
        self.el_surf = d.el_surf

    def load_json(self, json_filename):
        with open(json_filename, 'r') as f:
            config = json.load(f)
            self.c = np.array(config["constant"]["c"])
            self.lmbda = np.array(config["constant"]["lambda"])
            self.eps = np.array(config["constant"]["eps"])
            self.func_list = config["func"]
            
    def solve_ode(self):
        t_eval = np.linspace(self.t_span[0], self.t_span[1], 1000)
        self.sol = solve_ivp(self.system, self.t_span, self.T_0, t_eval=t_eval, method="BDF")

    def system(self, t, T):
        q1 = -np.sum(self.lmbda * self.el_surf_cross * np.subtract(T, T.reshape((len(T), 1))).T, axis=1)
        q2 = -self.eps * self.el_surf * self.c_0 * np.pow(T / 100, 4)
        q3 = eval(self.func_list)
        return (q1 + q2 + q3) / self.c

