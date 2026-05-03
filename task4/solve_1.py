from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

def run_test(u_exact_code, f_code, alpha_val, R, label):
    print(f"{label}")
    
    # сетка и пространство mesh ибо другое не работает
    domain = Circle(Point(0, 0), R)
    mesh = generate_mesh(domain, 30) # второй параметр это плотность сетки
    V = FunctionSpace(mesh, 'P', 1)

    # само выражение
    u_exact = Expression(u_exact_code, degree=4, R=R)
    f = Expression(f_code, degree=4, alpha=alpha_val, R=R)
    alpha = Constant(alpha_val)

    # граничные условия
    # дирихле: x < 0
    def dirichlet_boundary(x, on_boundary):
        return on_boundary and x[0] < 1e-14
    
    # нейман: x >= 0 
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
            # n = x / |x|
            r = np.sqrt(x[0]**2 + x[1]**2)
            n = x / r
            # конечные разности 
            eps = 1e-6
            du_dx = (self.ue(x[0]+eps, x[1]) - self.ue(x[0]-eps, x[1]))/(2*eps)
            du_dy = (self.ue(x[0], x[1]+eps) - self.ue(x[0], x[1]-eps))/(2*eps)
            values[0] = du_dx*n[0] + du_dy*n[1]
        def value_shape(self): return ()

    g = BoundarySource(u_exact, degree=2)
    bc = DirichletBC(V, u_exact, dirichlet_boundary)

    # вариационная задача
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(u), grad(v)) + alpha*u*v)*dx
    L = f*v*dx + g*v*ds(1)

    # Solution
    u_num = Function(V)
    solve(a == L, u_num, bc)

    # L2
    error_l2 = errornorm(u_exact, u_num, 'L2')
    
    # Max-norm
    u_e_v = interpolate(u_exact, V)
    error_max = np.abs(u_e_v.vector()[:] - u_num.vector()[:]).max()

    print(f"L2 Error:  {error_l2:.2e}")
    print(f"Max Error: {error_max:.2e}")

    # визуализация
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

    # сетки графиков
    z_num = np.asarray([u_num(cell.midpoint()) for cell in cells(mesh)])
    z_ex = np.asarray([u_exact(cell.midpoint()) for cell in cells(mesh)])

    # численное решение левый график
    tc1 = ax1.tripcolor(triangulation, facecolors=z_num, edgecolors='k', cmap='viridis')
    fig.colorbar(tc1, ax=ax1)
    ax1.set_title(f"Численное решение (МКЭ)\n{label}")
    ax1.set_aspect('equal')

    # аналитическое решение правый график
    tc2 = ax2.tripcolor(triangulation, facecolors=z_ex, edgecolors='k', cmap='viridis')
    fig.colorbar(tc2, ax=ax2)
    ax2.set_title(f"Аналитическое решение\n{label}")
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

R = 1.0
alpha_v = 2.0

# 1.u = x^2 + y^2
# -Δu + αu = -4 + α(x^2 + y^2)
run_test("x[0]*x[0] + x[1]*x[1]", 
         "-4.0 + alpha*(x[0]*x[0] + x[1]*x[1])", 
         alpha_v, R, "Polynomial")

# 2. u = cos(x)*cos(y)
# -Δu + αu = 2*cos(x)*cos(y) + α*cos(x)*cos(y)
run_test("cos(x[0])*cos(x[1])", 
         "(2.0 + alpha)*cos(x[0])*cos(x[1])", 
         alpha_v, R, "Trigonometric")

# 3. u = exp(x)
# -Δu + αu = -exp(x) + α*exp(x)
run_test("exp(x[0])", 
         "(alpha - 1.0)*exp(x[0])", 
         alpha_v, R, "Exponential")