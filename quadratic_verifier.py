import numpy as np
from dreal import *
import sympy

def quadratic_verifier(system, c1, tol=1e-4, accuracy=1e-4):
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = tol
    config.number_of_jobs = 32
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

    dreal_f = [sympy_to_dreal(expr, dict(zip(system.symbolic_vars, dreal_vars))) for expr in system.symbolic_f]

    def Check_reachability(x, c1, c2, config):    

        bounds_conditions = []

        for i in range(len(xlim)):
            lower_bound = x[i] >= xlim[i][0]
            upper_bound = x[i] <= xlim[i][1]
            bounds_conditions.append(logical_and(lower_bound, upper_bound))

        all_bounds = logical_and(*bounds_conditions)

        V = sum(x[i] * sum(system.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))

        lie_derivative_of_V = Expression(0)

        for i in range(len(x)):
            lie_derivative_of_V += dreal_f[i] * V.Differentiate(x[i])

        vars_in_bound = logical_and(c1 <= V, V <= c2)
        x_bound = logical_and(all_bounds, vars_in_bound)

        x_boundary = logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = logical_or(x[i] == xlim[i][1], x_boundary)

        set_inclusion = logical_imply(x_bound, logical_not(x_boundary))
        reach = logical_imply(x_bound, lie_derivative_of_V <= -tol)
        condition = logical_and(reach, set_inclusion)
        
        return CheckSatisfiability(logical_not(condition), config)

    # Bisection search for c
    c_low, c_high = 0, 100  
    c_best = None

    while c_high - c_low > accuracy:
        c_mid = (c_low + c_high) / 2
        result = Check_reachability(dreal_vars, c1, c_mid, config)

        if result:  
            c_high = c_mid
        else:  
            c_low = c_mid
            c_best = c_mid

    return c_best
