#7. WAP TO IMPLEMENT NEWTON METHOD:
"""
def f(x):
    return x**2 - 4*x + 4
def f_prime(x):
    return 2*x - 4
def f_double_prime(x):
    return 2
x0 = 3
tolerance = 1e-6
max_iterations = 100
for i in range(max_iterations):
    x1 = x0 - f_prime(x0) / f_double_prime(x0)
    if abs(x1 - x0) < tolerance:
        break
    x0 = x1
print()
print(f"Minimum Value: {f(x0)}")
print(f"Location: {x0}")



#OUTPUT:
Minimum Value: 0.0
Location: 2.0
"""



#8. WAP TO IMPLEMENT GRADIENT DESCENT METHOD:
"""
import sympy as sp
def gradient_descent(initial_x, learning_rate, num_iterations):
    x = sp.symbols('x')
    f = x**2 + 5*x + 6
    df = sp.diff(f, x)
    f_prime = sp.lambdify(x, df, 'numpy')
    for _ in range(num_iterations):
        gradient = f_prime(initial_x)
        initial_x = initial_x - learning_rate * gradient
    return initial_x, f.subs(x, initial_x)
initial_x = 0
learning_rate = 0.1
num_iterations = 1000
min_x, min_value = gradient_descent(initial_x, learning_rate, num_iterations)
print(f"Minimum value is {min_value} at x = {min_x}")




#OUTPUT:
Minimum value is -0.250000000000000 at x = -2.499999999999999
"""




#9. WAP TO IMPLEMENT STEEPEST DESCENT METHOD:
"""
import numpy as np
def objective_function(x):
    return x**2 + 4*x + 4
def gradient(x):
    return 2*x + 4
def steepest_descent(initial_guess, learning_rate, tolerance):
    x = initial_guess
    while True:
        grad_value = gradient(x)
        if np.linalg.norm(grad_value) < tolerance:
            break # Convergence criteria met
        x = x - learning_rate * grad_value
    return x, objective_function(x)
# Initial guess, learning rate, and tolerance
initial_guess = np.array([0.0])
learning_rate = 0.1
tolerance = 1e-6
result = steepest_descent(initial_guess, learning_rate, tolerance)
print("Optimal solution: x =", result[0])



#OUTPUT:
Optimal solution: x = [-1.99999959]
"""



#10. WAP TO IMPLEMENT TAYLOR POLYNOMIAL:
"""
import math
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
def taylor_coefficient(func, point, degree, derivative_order):
    return func(point) / factorial(derivative_order)
def taylor_polynomial(func, degree, point, var='x'):
    x = point
    taylor_poly = sum(taylor_coefficient(func, point, i, i) * (x - point)**i for i in range(degree + 1))
    return taylor_poly
def sin_function(x):
    return math.cos(x)
degree_of_polynomial = 3
center_point = 0
taylor_poly = taylor_polynomial(sin_function, degree_of_polynomial, center_point)
print("Taylor Polynomial:", taylor_poly)



#OUTPUT:
Taylor Polynomial: 1.0
"""

















