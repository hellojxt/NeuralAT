import bempp.api
import numpy as np

bempp.api.enable_console_logging("debug")
bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"

x0 = np.array([0.0, 0.0, 0.0])
CBIE_rerr_lst = []
HBIE_rerr_lst = []

for idx in range(100):
    wave_number = 100.6 + 0.01 * idx
    k = -wave_number

    @bempp.api.complex_callable
    def get_dirichlet(x, n, domain_index, result):
        r_vec = x - x0
        r = np.dot(r_vec, r_vec) ** 0.5
        result[0] = np.exp(1j * k * r) / (4 * np.pi * r)

    @bempp.api.complex_callable
    def get_neumann(x, n, domain_index, result):
        r_vec = x - x0
        r = np.dot(r_vec, r_vec) ** 0.5
        ikr = 1j * k * r
        result[0] = (
            -np.exp(ikr) / (4 * np.pi * r * r * r) * (1 - ikr) * ((x - x0) * n).sum()
        )

    wavelength = 2 * np.pi / abs(k)
    h = wavelength / 6
    # grid = bempp.api.shapes.sphere(0.1, h=h)
    grid = bempp.api.shapes.ellipsoid(0.15, 0.05, 0.05, h=h)

    space = bempp.api.function_space(grid, "P", 1)

    identity = bempp.api.operators.boundary.sparse.identity(space, space, space)
    slp = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, k)
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, k)
    hyp = bempp.api.operators.boundary.helmholtz.hypersingular(space, space, space, k)
    adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        space, space, space, k
    )

    LHS = 0.5 * identity - dlp
    RHS = -slp

    neumann_fun = bempp.api.GridFunction(space, fun=get_neumann)

    RHS = RHS * neumann_fun

    dirichlet_fun, info = bempp.api.linalg.gmres(LHS, RHS, tol=1e-5)
    dirichlet_fun_gt = bempp.api.GridFunction(space, fun=get_dirichlet)

    rerr = np.linalg.norm(
        dirichlet_fun.coefficients - dirichlet_fun_gt.coefficients
    ) / np.linalg.norm(dirichlet_fun_gt.coefficients)

    CBIE_rerr_lst.append(rerr)

    beta = 1j / k
    LHS = 0.5 * identity - dlp + beta * hyp
    RHS = (-slp - beta * (adlp + 0.5 * identity)) * neumann_fun

    dirichlet_fun, info = bempp.api.linalg.gmres(
        LHS, RHS, tol=1e-5, use_strong_form=True, maxiter=1000
    )
    dirichlet_fun_gt = bempp.api.GridFunction(space, fun=get_dirichlet)

    rerr = np.linalg.norm(
        dirichlet_fun.coefficients - dirichlet_fun_gt.coefficients
    ) / np.linalg.norm(dirichlet_fun_gt.coefficients)

    HBIE_rerr_lst.append(rerr)

    np.savetxt("CBIE_rerr_lst.txt", CBIE_rerr_lst)
    np.savetxt("HBIE_rerr_lst.txt", HBIE_rerr_lst)
