import numpy as np

def rastrigin(x: np.ndarray, out=None, A = 10):
    if len(x.shape) ==2:
        axis=1
    else:
        axis = 0

    # if out is None:
    #     y = np.zeros((x.shape[0],1),dtype=x.dtype)
    # else:
    #     y = out

    n = np.size(x,axis)

    y = A*n + np.sum(x**2-A*np.cos(2*np.pi*x))
    # for i in range(y.shape[0]):
    #     y[i] = A*n + np.sum(x[i]**2 - A*np.cos(2*np.pi*x[i]))
    
    return y

def schwefel(x: np.ndarray, out=None):
    if len(x.shape) == 2:
        axis = 1
    else:
        axis = 0

    n = np.size(x,axis)

    y = 418.9829*n - np.sum(x*np.sin(np.sqrt(np.abs(x))))

    return y

def rosenbrock(x: np.ndarray, out=None):
    if len(x.shape) == 2:
        axis = 1
    else:
        axis = 0

    n = np.size(x,axis)

    y = 0
    for i in range(n-1):
        y += 100*np.power((x[i+1] - np.power(x[i],2)),2) + np.power((x[i]-1),2)

    return y