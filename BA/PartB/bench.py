import numpy as np


# (-5.12, 5.12)
def sphere(x):
    return np.sum(x**2)

# (-5, 10)
def rosenbrock(x):
    xi = x[..., :-1]
    xnext = x[..., 1:]
    return np.sum(100 * (xnext - xi**2)**2 + (xi - 1)**2)

# (-5.12, 5.12)
def rastrigin(x):
    dim = x.shape[-1]
    return 10*dim + np.sum(x**2 - 10*np.cos(2 * np.pi * x))

# (-32.768, 32.768)
def ackley(x):
    dim = x.shape[-1]
    t1 = -0.2 * np.sqrt(np.sum(x**2, axis=-1) / dim)
    t2 = np.sum(np.cos((2*np.pi)*x), axis=-1) / dim
    return -20 * np.exp(t1) - np.exp(t2) + 20 + np.e
