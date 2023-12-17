#1. WAP TO FIND OPTIMAL SOLUTION USING LINE SEARCH METHOD:
"""
import numpy as np
from scipy.optimize import minimize_scalar

def objective_function(x):
    return (x-0.5)**2

def line_search_optimization():
    x0=1.0
    res=minimize_scalar(objective_function, method="Brent")
    print("Status :%s" % res["message"])
    print("Total Evaluations: %d" % res["nfev"])
    print("Optimal solution: x=%.3f" % res["x"])
    print("Objective Function value: f(x)=%.3f" % res["fun"])

line_search_optimization()
"""
#OUTPUT:
"""
Status :
Optimization terminated successfully;
The returned value satisfies the termination criteria
(using xtol = 1.48e-08 )
Total Evaluations: 8
Optimal solution: x=0.500
Objective Function value: f(x)=0.000
"""

#2. WAP TO SOLVE A LPP GRAPHICALLY:
"""
import matplotlib.pyplot as plt
import numpy as np
#Define constraint equations:
x = np.linspace (0, 10, 1000)
y1 = (6 - 2 * x) /3
y2 = (9 - 3 * x) /2
#Plot constraint lines:
plt.plot(x, y1, label="2x + 3y <= 6")
plt.plot(x, y2, label="3x + 2y <=9")
#Shade feasible region:
y3 = np.maximum(y1, y2)
plt.fill_between(x, y3, 0, where=(x >= 0) & (y1 >= 0) & (y2 >= 0), alpha=0.2)
#Define obj. func.:
z = lambda x, y: 5 * x + 4 * y
#Evaluate obj. func. at corners:
corners = [(0,0), (0,3), (1.8,1.2), (3,0)]
values = [z(*corner) for corner in corners]
#Plot corner points and optimal solution:
plt.scatter(*zip(*corners))
plt.annotate(f'Optimal solution: {max(values)}', corners[values.index(max(values))])
# Add labels and legend:
plt.xlabel('x')
plt.ylabel('y')
plt.title(" Graphical Solution of LPP: ")
plt.legend()
# Show plot
plt.show()
"""



#3. WAP TO COMPUTE THE GRADIENT AND HESSIAN OF FUNC. 100(X2-X1^2)^2 + (1-X1)^2:
"""
import numpy as np
def gradient (x) :
    return np. array(
        [400*x[0]*(x[0]**2 - x[1])- 2*(1 -x[0]), 200 * (x[1] - x[0]**2)]
    )
def hessian (x):
    return np.array(
        [[1200 * x[0] ** 2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]]
        )
# Test the functions
x = np. array ( [1, 1])
print ("Gradient at x = {}: {}".format(x, gradient (x)))
print ("Hessian at x = {}: {}".format (x, hessian(x)))
"""
#OUTPUT:
"""
Gradient at x = [1 1]: [0 0]
Hessian at x = [1 1]: [[ 802 -400]
 [-400  200]]"""




#4. WAP TO FIND GLOBAL OPTIMAL SOLUTION OF FN -10COS(PIx-2.2) ALGEBRAICALLY:
"""
import numpy as np
from scipy.optimize import minimize_scalar
def f(x):
    return -10*np.cos(np.pi*x-2.2)+(x+1.5)*x
res=minimize_scalar(f,bounds=(-10,10),method="bounded")
print(f"Global optimal solution of f(x) is {res.x:.4f} with a value of {res.fun:.4f}.")
"""
#OUTPUT:
"""
Global optimal solution of f(x) is 0.6714 with a value of -8.5010.
"""


#5. WAP TO FIND GLOBAL OPTIMAL SOLUTION OF FN -10COS(PIx-2.2) GRAPHICALLY:
"""
import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return -10 * np.cos(np.pi*x - 2.2) + (x + 1.5) * x
#Define the range of X values:
x = np.linspace(-5, 5, 1000)
#Plot the Function:
plt.plot(x, f(x))
#Find the global minimum:
global_min = np.min(f(x))
global_min_x = x[np.argmin(f(x))]
#Plot the global minimum:
plt.plot(global_min_x, global_min,"ro")
#Add labels and title:
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title ("Global optimal solution of f(x)")
# Show the plot
plt.show()
"""



#6. WAP TO SOLVE CONSTRAINT OPTIMIZATION PROBLEM:
"""
from scipy.optimize import minimize
def objective (x) :
    return -2 * x[0] - 2 * x[1] - 3 *x[2]
def constraint1(x) :
    return 50 - 2 * x[0] - 7 * x[1] - 3 * x[2]
def constraint2(x):
    return 45 - 3 * x[0] + 5 * x[1] - 7 * x[2]
def constraint3(x):
    return 37 - 5 * x[0] - 2 * x[1] + 6 * x[2]
x0 = [8, 0, 0]
b = (0.0, None)
bnds = (b, b, b)
con1 = {"type":"ineq", "fun": constraint1}
con2 = {"type":"ineq", "fun": constraint2}
con3 = {"type":"ineq", "fun": constraint3}
cons = [con1, con2, con3]

sol = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons)
print(sol)
"""


#OUTPUT:
"""
message: Optimization terminated successfully
 success: True
  status: 0
     fun: -37.39919354838712
       x: [ 1.074e+01  2.520e+00  3.625e+00]
     nit: 3
     jac: [-2.000e+00 -2.000e+00 -3.000e+00]
    nfev: 12
    njev: 3

"""




