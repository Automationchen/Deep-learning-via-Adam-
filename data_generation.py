# Import modules
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from casadi import *

# Parameters
n_training_points = 2000 # Choose here the number of training points
function_type = 'rastrigin' # either 'rastrigin' or 'goldstein_price'
save_name = 'data_set_'

# Define variables and functions
x = SX.sym("x")
y = SX.sym("y")
if function_type == 'rastrigin':
    A = 10.0
    expr = 2*A + (x**2 - A * cos(2*pi*x)) + (y**2 - A * cos(2*pi*y))
    fun = Function('fun',[x,y],[expr])
    save_name += 'rastrigin.npy'

elif function_type == 'goldstein_price':
    expr = (1+(x+y+1)**2 * (19-14*x+3*x**2-14*y+6*x*y+3*y**2)) * (30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    fun = Function('fun',[x,y],[expr])
    save_name += 'goldstein_price.npy'

# Define domain
if function_type == 'rastrigin':
    lb = [-5,-5]
    ub = [5,5]
elif function_type == 'goldstein_price':
    lb = [-2,-2]
    ub = [2,2]

# Generate random data and save it
inp_rand = np.random.uniform(lb,ub,size=(n_training_points,2))
out_rand = fun(inp_rand[:,0],inp_rand[:,1])
np.save(save_name,np.hstack([inp_rand,out_rand]))

# Generate grid data for plotting
x_ls = np.linspace(lb,ub,100)
X, Y = np.meshgrid(x_ls[:,0],x_ls[:,1])

# Evaluate grid
Z = np.array(fun(X,Y))

# Plot function
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.savefig(function_type+'.png')
plt.show()
