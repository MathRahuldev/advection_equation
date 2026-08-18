"""
Solving u_t + f(u)_x = 0 for f(u) = u
Numerical schemes in finite difference form
"""

import numpy as np
import matplotlib.pyplot as plt

a = -0.5
b = 0.5
N = 100
dx = (b-a) / N
x = np.linspace(a, b, N, endpoint=False)

final_t = 0.2
CFL = 0.9

y0 = np.exp(-100*x**2) # np.sin(2 * np.pi * x)

def fD(y):
    return (np.roll(y,-1)-y)/ dx
def bD(y):
    return (y - np.roll(y,1))/ dx
def cD(y):
    return (np.roll(y,-1) - np.roll(y,1))/ (2 * dx)

t = 0
dt = CFL * dx
y = y0.copy()
while (t < final_t):
    y -= dt * bD(y)
    t += dt
    dt = min(dt, final_t-t)
    print("time ", t) 

t = 0
dt = CFL * dx
y1 = y0.copy()
while (t < final_t):
    y1 -= dt * fD(y1)
    t += dt
    dt = min(dt, final_t-t)
    print("time ", t)

t = 0
dt = CFL * dx
y2 = y0.copy()
while (t < final_t):
    y2 -= dt * cD(y2)
    t += dt
    dt = min(dt, final_t-t)
    print("time ", t)

plt.plot(x, y0, "--", label = "initial data")
plt.plot(x, np.exp(-100*(x-t)**2), "-", label = "exact solution")
plt.plot(x, y, "o", label = "Backward at t = 0.2")
plt.plot(x, y2, "*", label = "Centred at t = 0.2")
plt.plot(x, y1, "-", label = "Forward at t = 0.2")
plt.title("advection equation with unit velocity and CFL = 0.9")
plt.legend()
plt.show()
