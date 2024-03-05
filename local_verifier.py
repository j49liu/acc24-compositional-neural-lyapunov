import numpy as np
from dreal import *
import sympy

def local_verifier(system, tol=1e-4, accuracy=1e-4):
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = tol
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    dreal_vars = [Variable(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]

    def sympy_to_dreal(expr, subs):
        if expr.is_Add:
            return sum(sympy_to_dreal(arg, subs) for arg in expr.args)
        elif expr.is_Mul:
            return sympy.prod(sympy_to_dreal(arg, subs) for arg in expr.args)
        elif expr.is_Pow:
            return sympy_to_dreal(expr.base, subs)**int(expr.exp)
        elif expr in subs:
            return subs[expr]
        else:
            return float(expr)


    # # Convert symbolic functions to dReal expressions
    # dreal_f = [expr.subs({sympy_var: dreal_var for sympy_var, dreal_var in zip(system.symbolic_vars, dreal_vars)})
    #            for expr in system.symbolic_f]

    dreal_f = [sympy_to_dreal(expr, dict(zip(system.symbolic_vars, dreal_vars))) for expr in system.symbolic_f]

    # print(dreal_f)

    def Check_local_stability(x, c, config):    

        Q_np = np.array(system.Q).astype(np.float64)
        r = np.min(np.linalg.eigvalsh(Q_np)) - tol

        bounds_conditions = []

        for i in range(len(xlim)):
            lower_bound = x[i] >= xlim[i][0]
            upper_bound = x[i] <= xlim[i][1]
            bounds_conditions.append(logical_and(lower_bound, upper_bound))

        all_bounds = logical_and(*bounds_conditions)

        quadratic_form_expr = sum(x[i] * sum(system.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))

        omega = quadratic_form_expr <= c

        g = [dreal_f[i] - sum(system.A[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]

        # print(g)

        Dg = [[None for _ in range(len(x))] for _ in range(len(g))]
        for i in range(len(g)):
            for j in range(len(x)):
                Dg[i][j] = g[i].Differentiate(x[j])

        # print(Dg)

        P_Dg = np.dot(system.P, Dg)

        # print(P_Dg)

        # Using Frobenius norm to estimate 2-norm

        frobenius_norm_square = sum(sum(m_ij**2 for m_ij in row) for row in P_Dg)
        h = (2**2 * frobenius_norm_square) <= r**2

        # print(h)

        stability = logical_imply(logical_and(omega, all_bounds), h)
        return CheckSatisfiability(logical_not(stability), config)

    # Bisection search for c
    c_low, c_high = 0, 100  
    c_best = None

    while c_high - c_low > accuracy:
        c_mid = (c_low + c_high) / 2
        result = Check_local_stability(dreal_vars, c_mid, config)

        if result:  
            c_high = c_mid
        else:  
            c_low = c_mid
            c_best = c_mid

    return c_best