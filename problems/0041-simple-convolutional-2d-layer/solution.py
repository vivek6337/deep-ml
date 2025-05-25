import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Your code here
    padded_matrix = np.pad(input_matrix , ((padding,padding) , (padding,padding)) , mode = 'constant' , constant_values = 0)
    p_h = input_height + 2*padding
    p_w = input_width + 2*padding
    o_c =(( p_w - (kernel_width -1)-1)/stride) + 1
    o_r = ((p_h - (kernel_height -1)-1)/stride) + 1
    output_matrix = list()
    for i in range(int(o_r)):
        l = []
        for j in range(int(o_c)):
            m_p = padded_matrix[i*stride : i*stride+kernel_height , j*stride : j*stride+kernel_width ]
            v = np.sum(m_p*kernel)
            l.append(v)
        output_matrix.append(l)

    return output_matrix
