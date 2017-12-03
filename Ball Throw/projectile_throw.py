import math
import numpy as np


y = 40
x=200
v=10
g=9.81

a = math.sqrt(( (v**4) - ( (2*g*v**2) - (g**2*x**2) ) ))
#print(a)
b = (2*v**2) - (2*g*y) - (2*a)
#print(b)
val =  math.sqrt( -b )/g
#val = (v**2 - math.sqrt(v**4-g*(g*x*x + 2*y*v*v))/(g*x))
#theta = np.arctan( (v**2 - math.sqrt(v**4-g*(g*x*x + 2*y*v*v))/(g*x)) )


print(val)
a = v**2/(g*x)
b = ( v**2*(v**2 - 2*g*y) )/(g**2*x**2)
theta = a - math.sqrt( b - 1 )
theta_degree = math.degrees( math.atan( theta ) )
print(theta_degree) 