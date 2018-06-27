import numpy as np 


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entoropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y)) / batch_size

def numerical_diff(f , x):
    h = 1e-4
    return (f(x+h) -  f(x-h)) / (2*h)


def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val +h
        fxh1 = f(x)

def gradient_descent(f,init_x , lr =0.01 , step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x

        x[idx]= tmp_val -h
        fxh2 = f(x)

        grad[idx] =  (fxh1 - fxh2) /(2*h)
        x[idx] = tmp_val

    return grad

