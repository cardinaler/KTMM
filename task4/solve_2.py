from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation

def run_time_dependent_test(u_exact_code, f_code, param_a_val, R, T, num_steps, label):
    print(f"\nЗапуск теста: {label}")
    dt = T / num_steps
    
    # Создание сетки и пространства вычислений
    domain = Circle(Point(0, 0), R)
    mesh = generate_mesh(domain, 30)
    V = FunctionSpace(mesh, 'P', 1)

    # Определение выражений
    u_exact = Expression(u_exact_code, degree=4, R=R, t=0.0)
    f = Expression(f_code, degree=4, param_a=param_a_val, R=R, t=0.0)
    param_a = Constant(param_a_val)
    dt_const = Constant(dt)

    # Граничные условия
    def dirichlet_boundary(x, on_boundary):
        return on_boundary and x[0] < 1e-14
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class NeumannPart(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] >= -1e-14
    
    NeumannPart().mark(boundaries, 1)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    class BoundarySource(UserExpression):
        def __init__(self, ue, **kwargs):
            super().__init__(**kwargs)
            self.ue = ue
        def eval_cell(self, values, x, cell):
            r = np.sqrt(x[0]**2 + x[1]**2)
            n = x / r
            eps = 1e-6
            du_dx = (self.ue(x[0]+eps, x[1]) - self.ue(x[0]-eps, x[1]))/(2*eps)
            du_dy = (self.ue(x[0], x[1]+eps) - self.ue(x[0], x[1]-eps))/(2*eps)
            values[0] = du_dx*n[0] + du_dy*n[1]
        def value_shape(self): return ()

    g = BoundarySource(u_exact, degree=2)
    bc = DirichletBC(V, u_exact, dirichlet_boundary)

    # Начальное условие
    u_exact.t = 0.0
    u_n = interpolate(u_exact, V)

    # Вариационная постановка через неявную схему Эйлера
    u = TrialFunction(V)
    v = TestFunction(V)
    a_form = (u*v + dt_const*param_a*dot(grad(u), grad(v)))*dx
    
    u_num = Function(V)
    
    t_values = [0.0]
    l2_errors = [0.0]
    max_errors = [0.0]
    
    # Сохранение для дальнейшей визуализации
    n_vert = mesh.num_vertices()
    d_dim = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n_vert, d_dim))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
    
    z_num_frames = [u_n.compute_vertex_values(mesh)]
    z_ex_frames = [u_n.compute_vertex_values(mesh)]

    # Цикл по времени
    t = 0.0
    for step in range(num_steps):
        t += dt
        u_exact.t = t
        f.t = t

        # Solution
        L_form = (u_n*v + dt_const*f*v)*dx + dt_const*param_a*g*v*ds(1)
        solve(a_form == L_form, u_num, bc)
        
        error_l2 = errornorm(u_exact, u_num, 'L2')
        u_e_v = interpolate(u_exact, V)
        error_max = np.abs(u_e_v.vector()[:] - u_num.vector()[:]).max()
        
        t_values.append(t)
        l2_errors.append(error_l2)
        max_errors.append(error_max)
        
        u_n.assign(u_num)
        
        z_num_frames.append(u_num.compute_vertex_values(mesh))
        z_ex_frames.append(u_e_v.compute_vertex_values(mesh))

    print(f"Error L2:  {l2_errors[-1]:.2e}")
    print(f"Error Max: {max_errors[-1]:.2e}")

    # Вывод анимации и отрисовка
    plt.ion() 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Настройка масштабов цвета через выделение глобального максимума и минимума при всех t
    global_min = min(np.min(z_num_frames), np.min(z_ex_frames))
    global_max = max(np.max(z_num_frames), np.max(z_ex_frames))
    if np.isclose(global_min, global_max):
        global_min -= 0.1
        global_max += 0.1
        
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.15, label='Амплитуда поля u')
    
    def update_frame(idx):
        ax1.clear()
        ax2.clear()
        
        z_num = z_num_frames[idx]
        z_ex = z_ex_frames[idx]
        
        tc1 = ax1.tripcolor(triangulation, z_num, edgecolors='k', cmap='viridis', vmin=global_min, vmax=global_max)
        ax1.set_title(f"Численное решение (МКЭ)\n{label} | t = {t_values[idx]:.2f}")
        ax1.set_aspect('equal')
        
        tc2 = ax2.tripcolor(triangulation, z_ex, edgecolors='k', cmap='viridis', vmin=global_min, vmax=global_max)
        ax2.set_title(f"Аналитическое решение\n{label} | t = {t_values[idx]:.2f}")
        ax2.set_aspect('equal')
        
        return tc1, tc2
    # Анимация
    anim = FuncAnimation(fig, update_frame, frames=len(t_values), interval=250, blit=False, repeat=True)
    
    plt.ioff()
    plt.show()
# Параметры задачи
R_val = 1.0
param_a_coeff = 0.5  # Коэффициент теплопроводности
T_final = 20
steps = 20

# ТЕСТЫ

# 1. Полиномиальная по пространству и линейная по времени
# u = (x^2 + y^2)*(t + 1) 
# du/dt = x^2 + y^2, Δu = 4*(t + 1)
# f = du/dt - a*Δu = (x^2 + y^2) - 4*a*(t + 1)
run_time_dependent_test(
    "(x[0]*x[0] + x[1]*x[1]) * (t + 1.0)", 
    "(x[0]*x[0] + x[1]*x[1]) - 4.0 * param_a * (t + 1.0)", 
    param_a_coeff, R_val, T_final, steps, "Polynomial"
)

# 2. Тригонометрическая с затуханием по времени
# u = cos(6*x)*sin(6*x)*t
# du/dt = cos(6*x)*sin(6*x), Δu = -4*u
# f = du/dt - a*Δu = (36 *4 * a*t - 1.0) * cos(x[0] *6)*sin(x[0] * 6)
run_time_dependent_test(
    "cos(x[0]*6)*sin(x[0] * 6) * t", 
    "(36 *4 * param_a*t - 1.0) * cos(x[0] *6)*sin(x[0] * 6)", 
    param_a_coeff, R_val, T_final, steps, "Trigonometric"
)

# 3. Экспоненциальная по пространству, квадратичная по времени
# u = exp(x)*(t^2 + 1)
# du/dt = 2*t*exp(x), Δu = exp(x)*(t^2 + 1)
# f = du/dt - a*Δu = exp(x) * (2*t - a*(t^2 + 1))
run_time_dependent_test(
    "exp(x[0]) * (t*t + 1.0)", 
    "exp(x[0]) * (2.0*t - param_a * (t*t + 1.0))", 
    param_a_coeff, R_val, T_final, steps, "Exponential"
)