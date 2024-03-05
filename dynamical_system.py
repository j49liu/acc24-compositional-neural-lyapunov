import numpy as np
import sympy
from scipy.linalg import solve_continuous_lyapunov

class DynamicalSystem:
    def __init__(self, f, domain, name):
        self.f = f  # This can be a lambda function or any other callable
        self.domain = domain
        self.name = name
        self.symbolic_vars = sympy.symbols('x1:{}'.format(len(f)+1))  # Create symbolic variables for SymPy
        self.symbolic_f = [func(*self.symbolic_vars) for func in f]  # List of symbolic expressions
        self.A = self.compute_linearization()  # Compute the linearization matrix A
        self.Q = np.eye(len(self.f))  # Default to the identity matrix
        self.P = self.compute_quadratic_lyapunov_function() # Compute a quadratic Lyapunov function

    def compute_quadratic_lyapunov_function(self):
        """Compute the P matrix using the continuous Lyapunov equation."""
        # Solve for P
        self.P = solve_continuous_lyapunov(self.A.T, -self.Q)
        return self.P

    def compute_linearization(self):
        """Compute the Jacobian matrix for the dynamical system using symbolic differentiation and evaluate it at the origin."""
        jacobian = sympy.Matrix(self.symbolic_f).jacobian(self.symbolic_vars)
        # Substitute the variables with zeros to evaluate at the origin
        origin = {var: 0 for var in self.symbolic_vars}
        jacobian_at_origin = jacobian.subs(origin)
        # return jacobian_at_origin
                # Convert the symbolic Jacobian to a NumPy array
        # func = sympy.lambdify(self.symbolic_vars, jacobian_at_origin, "numpy")
        # jacobian_np = func(*[0 for _ in self.symbolic_vars])
        jacobian_np = np.array(jacobian_at_origin).astype(np.float64)
        return jacobian_np
