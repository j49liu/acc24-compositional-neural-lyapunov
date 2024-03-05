import numpy as np
from dreal import *
import sympy
import time 
from utils import *

def neural_verifier(system, model, c2, tol=1e-4, accuracy=1e-2):
    config = config_dReal()
    print("Configuration for dReal: ")
    print(config_dReal())
    print("\n")
    xlim = system.domain
    epsilon = 1e-4

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

    # Construct V from neural network model
    layers = len(model.layers) 
    # Extract weights for each layer
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    # Extract weights and biases for the final layer separately
    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    # Calculate h for each layer
    h = dreal_vars
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [tanh(z[j]) for j in range(len(weights[i]))]
        # h = [sin(z[j]) for j in range(len(weights[i]))]

    
    V_learn = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    
    # print(V_learn)


    def Check_inclusion(x, c1_V, c2, config):

        bounds_conditions = []
        for i in range(len(xlim)):
            lower_bound = x[i] >= xlim[i][0]
            upper_bound = x[i] <= xlim[i][1]
            bounds_conditions.append(logical_and(lower_bound, upper_bound))

        all_bounds = logical_and(*bounds_conditions)

        vars_in_bound = V_learn <= c1_V
        x_bound = logical_and(all_bounds, vars_in_bound)

        quad_V = sum(x[i] * sum(system.P[i][j] * x[j] for j in range(len(x))) for i in range(len(x)))
        target = quad_V <= c2

        condition = logical_imply(x_bound, target)
        return CheckSatisfiability(logical_not(condition), config)

    def Check_reachability(x, c1, c2, config):    

        lie_derivative_of_V = Expression(0)

        for i in range(len(x)):
            lie_derivative_of_V += dreal_f[i] * V_learn.Differentiate(x[i])

        bounds_conditions = []
        for i in range(len(xlim)):
            lower_bound = x[i] >= xlim[i][0]
            upper_bound = x[i] <= xlim[i][1]
            bounds_conditions.append(logical_and(lower_bound, upper_bound))

        all_bounds = logical_and(*bounds_conditions)

        vars_in_bound = logical_and(c1 <= V_learn, V_learn <= c2)
        x_bound = logical_and(all_bounds, vars_in_bound)

        x_boundary = logical_or(x[0] == xlim[0][0], x[0] == xlim[0][1])
        for i in range(1, len(x)):
            x_boundary = logical_or(x[i] == xlim[i][0], x_boundary)
            x_boundary = logical_or(x[i] == xlim[i][1], x_boundary)

        set_inclusion = logical_imply(x_bound, logical_not(x_boundary))
        reach = logical_imply(x_bound, lie_derivative_of_V <= -epsilon)
        condition = logical_and(reach, set_inclusion)
        
        return CheckSatisfiability(logical_not(condition), config)

    start_time = time.time()
    # Bisection search for c1_V
    c_low, c_high = 0, 1  
    c1_best = None

    while c_high - c_low > accuracy:
        c_mid = (c_low + c_high) / 2
        result = Check_inclusion(dreal_vars, c_mid, c2, config)

        if result:  
            c_high = c_mid
        else:  
            c_low = c_mid
            c1_best = c_mid

    print(f"Verified V<={c1_best} is contained in x^TPx<={c2}.")

    # Bisection search for c2_V
    c_low, c_high = c1_best, 1  
    c2_best = None

    while c_high - c_low > accuracy:
        c_mid = (c_low + c_high) / 2
        result = Check_reachability(dreal_vars, c1_best, c_mid, config)

        if result:  
            c_high = c_mid
        else:  
            c_low = c_mid
            c2_best = c_mid

    print(f"Verified V<={c2_best} will reach V<={c1_best} and hence x^TPx<={c2}.")
    end_time = time.time()
    print(f"Time taken for verifying Lyapunov function of {system.name}: {end_time - start_time} seconds.\n")
    
    return c1_best, c2_best
