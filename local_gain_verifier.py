import numpy as np
from dreal import *
import sympy

def self_local_gain_verifier(system, c, tol=1e-4, accuracy=1e-4):
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = tol
    config.number_of_jobs = 8

    xlim = system.domain

    x = [Variable(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]

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

    f = [sympy_to_dreal(expr, dict(zip(system.symbolic_vars, x))) for expr in system.symbolic_f]

    def Check_self_gain(r, config):

        x_bounds_conditions = []

        for i in range(len(xlim)):
            x_lower_bound = x[i] >= xlim[i][0]
            x_upper_bound = x[i] <= xlim[i][1]
            x_bounds_conditions.append(logical_and(x_lower_bound, x_upper_bound))

        x_all_bounds = logical_and(*x_bounds_conditions)

        xPx = sum(x[i] * sum(system.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))
        g = [f[i] - sum(system.A[i][j] * x[j] for j in range(len(x))) for i in range(len(x))]

        Dg = [[None for _ in range(len(x))] for _ in range(len(g))]
        for i in range(len(g)):
            for j in range(len(x)):
                Dg[i][j] = g[i].Differentiate(x[j])

        P_Dg = np.dot(system.P, Dg)

        frobenius_norm_square = sum(sum(m_ij**2 for m_ij in row) for row in P_Dg)
        h = frobenius_norm_square <= r**2

        # print(h)

        condition = logical_imply(logical_and(xPx <= c, x_all_bounds), h)
        return CheckSatisfiability(logical_not(condition), config)

    # Bisection search for r
    r_low, r_high = 0, 0.5 
    r_best = None

    while r_high - r_low > accuracy:
        r_mid = (r_low + r_high) / 2
        result = Check_self_gain(r_mid, config)

        if result:  
            r_low = r_mid
        else:  
            r_high = r_mid
            r_best = r_mid

    return r_best

    return 


def local_gain_verifier(system1, c1, system2, c2, mu12, tol=1e-4, accuracy=1e-4):
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = tol
    config.number_of_jobs = 8

    xlim = system1.domain
    ylim = system2.domain

    x = [Variable(f"x{i}") for i in range(1, len(system1.symbolic_vars) + 1)]
    y = [Variable(f"y{i}") for i in range(1, len(system2.symbolic_vars) + 1)]

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

    f1 = [sympy_to_dreal(expr, dict(zip(system1.symbolic_vars, x))) for expr in system1.symbolic_f]
    f2 = [sympy_to_dreal(expr, dict(zip(system2.symbolic_vars, y))) for expr in system2.symbolic_f]

    def Check_local_gain(r, config):

        x_bounds_conditions = []

        for i in range(len(xlim)):
            x_lower_bound = x[i] >= xlim[i][0]
            x_upper_bound = x[i] <= xlim[i][1]
            x_bounds_conditions.append(logical_and(x_lower_bound, x_upper_bound))

        x_all_bounds = logical_and(*x_bounds_conditions)

        y_bounds_conditions = []

        for i in range(len(ylim)):
            y_lower_bound = y[i] >= ylim[i][0]
            y_upper_bound = y[i] <= ylim[i][1]
            y_bounds_conditions.append(logical_and(y_lower_bound, y_upper_bound))

        y_all_bounds = logical_and(*y_bounds_conditions)

        xPx = sum(x[i] * sum(system1.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))
        yPy = sum(y[i] * sum(system2.P[i][j] * y[j] for j in range(len(y))) for i in range(len(y)))

        g = [Expression(0.0), mu12*x[0]*y[1]]

        Dg = [[None for _ in range(len(x)+len(y))] for _ in range(len(g))]

        for i in range(len(g)):
            for j in range(len(x)):
                Dg[i][j] = g[i].Differentiate(x[j])
            for j in range(len(y)):
                Dg[i][len(x)+j] = g[i].Differentiate(y[j])

        P_Dg = np.dot(system1.P, Dg)

        frobenius_norm_square = sum(sum(m_ij**2 for m_ij in row) for row in P_Dg)
        h = frobenius_norm_square <= r**2

        # print(h)

        x_final_bounds = logical_and(x_all_bounds,xPx<=c1)
        y_final_bounds = logical_and(y_all_bounds,yPy<=c2)

        condition = logical_imply(logical_and(x_final_bounds,y_final_bounds), h)
        return CheckSatisfiability(logical_not(condition), config)

    # Bisection search for r
    r_low, r_high = 0, 100  
    r_best = None

    while r_high - r_low > accuracy:
        r_mid = (r_low + r_high) / 2
        result = Check_local_gain(r_mid, config)

        if result:  
            r_low = r_mid
        else:  
            r_high = r_mid
            r_best = r_mid

    return r_best

    return