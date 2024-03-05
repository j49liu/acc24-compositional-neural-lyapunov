import numpy as np
from dreal import *
import sympy
from utils import *

config = config_dReal()

def self_quadratic_gain_verifier(system, c2_V, total_gain):
    epsilon = -1e-4
    xlim = system.domain
    x = [Variable(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    f = [sympy_to_dreal(expr, dict(zip(system.symbolic_vars, x))) for expr in system.symbolic_f]
    V = sum(x[i] * sum(system.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))
    lie_derivative_of_V = Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])
    def Check_self_gain(c):
        x_bound = get_bound(x, xlim, V, c1_V=c, c2_V=c2_V)
        condition = logical_imply(x_bound, lie_derivative_of_V <= -total_gain + epsilon)
        return CheckSatisfiability(logical_not(condition), config)
    return bisection_lub(Check_self_gain, 0, c2_V, accuracy=1e-4)

def quadratic_gain_verifier(system1, c1, system2, c2):
    xlim = system1.domain
    ylim = system2.domain
    x = [Variable(f"x{i}") for i in range(1, len(system1.symbolic_vars) + 1)]
    y = [Variable(f"y{i}") for i in range(1, len(system2.symbolic_vars) + 1)]
    V1 = sum(x[i] * sum(system1.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))
    V2 = sum(y[i] * sum(system2.P[i][j] * y[j] for j in range(len(y))) for i in range(len(y)))
    x_bound = get_bound(x, xlim, V1, c2_V=c1)
    y_bound = get_bound(y, ylim, V2, c2_V=c2)
    def Check_ub1(r):
        condition = logical_imply(x_bound, (V1.Differentiate(x[1]) * x[0])**2 <= r)
        return CheckSatisfiability(logical_not(condition), config)
    def Check_ub2(r):
        condition = logical_imply(y_bound, y[1]**2 <= r)
        return CheckSatisfiability(logical_not(condition), config)
    r1 = bisection_lub(Check_ub1, 0, 100, accuracy=1e-4)
    r2 = bisection_lub(Check_ub2, 0, 100, accuracy=1e-4)
    # print("r1: ", r1)
    # print("r2: ", r2)
    return np.sqrt(r1 * r2)
