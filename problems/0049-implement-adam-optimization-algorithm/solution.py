import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	# Your code here
    v_t = 0
    w_t = 0;
    for i in range(num_iterations):
        d_w = grad(x0)
        v_t = beta2 * v_t + (1-beta2)* (d_w**2)
        w_t = beta1 * w_t + (1-beta1)*d_w
        x0 = x0 - (learning_rate*(w_t/(1-(beta1**(i+1)))))/np.sqrt((v_t/(1-(beta2**(i+1)))) + epsilon)
    return x0
