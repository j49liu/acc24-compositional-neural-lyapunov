import sympy
from dreal import *
import os
import re

def bisection_glb(Check_bound, low, high, accuracy=1e-4):
    best = None

    while high - low > accuracy:
        mid = (low + high) / 2
        result = Check_bound(mid)
        if result:
            high = mid
        elif result is None:
            low = mid
            best = mid
            # print(mid)
    return best

def bisection_lub(Check_bound, low, high, accuracy=1e-4):
    best = None
    while high - low > accuracy:
        mid = (low + high) / 2
        result = Check_bound(mid)
        if result:
            low = mid
        elif result is None:
            high = mid
            best = mid
            # print(mid)
    return best

def bisection_lub_sat(Check_bound, low, high, accuracy=1e-4):
    best = None
    while high - low > accuracy:
        mid = (low + high) / 2
        result = Check_bound(mid)
        if result:
            low = mid
            best = mid
        elif result is None:
            high = mid
            # print(mid)
    return best

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

def get_bound(x, xlim, V, c1_V=0.0, c2_V=1.0): 
    bounds_conditions = []
    for i in range(len(xlim)):
        lower_bound = x[i] >= xlim[i][0]
        upper_bound = x[i] <= xlim[i][1]
        bounds_conditions.append(logical_and(lower_bound, upper_bound))
    all_bounds = logical_and(*bounds_conditions)
    c1_V = Expression(c1_V)
    c2_V = Expression(c2_V)
    vars_in_bound = logical_and(c1_V <= V, V <= c2_V)
    x_bound = logical_and(all_bounds, vars_in_bound)
    return x_bound

def config_dReal():
    config = Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = 1e-4
    config.number_of_jobs = 32
    return config


def read_sos():
    x1, x2 = sympy.symbols('x1 x2')
    sos_expressions = []
    # Loop through all the files
    for i in range(1, 11):
        file_name = f'sos/output_vdp_network_subsystem_{i}.txt'
        
        # Open the file
        with open(file_name, 'r') as file:
            content = file.read()
            
            # Find the expression for "Final V"
            match = re.search(r'Final V:\s*([\s\S]+?)\n\n', content)
            if match:
                expression_str = match.group(1)
                sos_expressions.append(sympy.sympify(expression_str))
    return sos_expressions

def write_sos(expressions, output_path):
    with open(output_path, 'w') as file:
        file.write("expressions = [\n")
        for expr in expressions:
            file.write(f"    \"{expr}\",\n")
        file.write("]\n")

def lambdify_sos_V(expressions):
    functions = []
    for expr in expressions:
        # symbolic_vars = sorted(list(expr.free_symbols), key=str)
        x1, x2 = sympy.symbols('x1 x2')
        symbolic_vars = [x1, x2]
        func = sympy.lambdify(symbolic_vars, expr, "numpy")
        functions.append(func)
    return functions
