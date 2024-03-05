import numpy as np
from dreal import *
import sympy
from utils import *

config = config_dReal()

def compute_V(model, x):
    # evaluate the expression of a neural network function
    layers = len(model.layers)
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]
    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()
    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [tanh(z[j]) for j in range(len(weights[i]))]    
    return (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]

def compute_lower_bound_V(system, model, c2_V):
    # compute the greatest lower bound of a neural network function 
    x = [Variable(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    V = compute_V(model, x)
    xlim = system.domain
    x_bound = get_bound(x, xlim, V, c2_V=c2_V)
    def Check_bound(lb):
        condition = logical_imply(x_bound, V >= lb)
        return CheckSatisfiability(logical_not(condition), config)    
    return bisection_glb(Check_bound, -0.1, 1)

def compute_V_inclusion(system, model, c2_P, c2_V):
    # check if the level set V<=c2_V is contaiend in the level set of x^TPx<=c2_P
    xlim = system.domain
    x = [Variable(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    V = compute_V(model, x)
    quad_V = sum(x[i] * sum(system.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))
    target = quad_V <= c2_P
    def Check_inclusion(c):
        x_bound = get_bound(x, xlim, V, c2_V=c)
        condition = logical_imply(x_bound, target)
        return CheckSatisfiability(logical_not(condition), config)
    c1_V_best = bisection_glb(Check_inclusion, 0, c2_V)
    print(f"Verified V<={c1_V_best} is contained in x^TPx<={c2_P}.")
    return c1_V_best

def self_neural_gain_verifier(system, model, c2_V, total_gain):
    epsilon = -1e-4
    xlim = system.domain
    x = [Variable(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    f = [sympy_to_dreal(expr, dict(zip(system.symbolic_vars, x))) for expr in system.symbolic_f]
    V = compute_V(model, x)
    lie_derivative_of_V = Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])
    def Check_self_gain(c):
        x_bound = get_bound(x, xlim, V, c1_V=c, c2_V=c2_V)
        condition = logical_imply(x_bound, lie_derivative_of_V <= -total_gain + epsilon)
        return CheckSatisfiability(logical_not(condition), config)
    return bisection_lub(Check_self_gain, 0, c2_V, accuracy=1e-4)

def neural_gain_verifier(system1, model1, c1, system2, model2, c2):
    xlim = system1.domain
    ylim = system2.domain
    x = [Variable(f"x{i}") for i in range(1, len(system1.symbolic_vars) + 1)]
    y = [Variable(f"y{i}") for i in range(1, len(system2.symbolic_vars) + 1)]
    V1 = compute_V(model1, x)
    V2 = compute_V(model2, y)
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
