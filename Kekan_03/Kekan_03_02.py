import numpy as np
import imageio
import time
def read_one_image_and_convert_to_vector(file_name):
    f='.\data\mnist_images\\'+ file_name
    img = imageio.imread(f).astype(np.float32) # read image and convert to float
    return img.reshape(-1,1) # reshape to column vector and return it

def filtered_learning(weight_old, gamma, alpha, t, p):
    weight_new=(1-gamma)*np.matrix(weight_old) + alpha * (np.matrix(t) * np.matrix(p).transpose())
    return np.matrix(weight_new)

def delta_rule(weight_old, alpha, t, a, p):
    error=np.subtract(t,a)
    tmp=np.matrix(error).transpose().dot(np.matrix(p))
    return weight_old+alpha * tmp

def unsupervised(weight_old, alpha, a, p):
    weight_new=weight_old + alpha * (np.array(a)*np.array(p).transpose())
    return weight_new
