# Kekan Nikhilkumar
# 1001-563-734
# 2018-09-09
# Assignment-01-02
import numpy as np
# This module calculates the activation function
def calculate_activation_function(weight,bias,input_array,type='Sigmoid'):
	net_value = weight * input_array + bias
	if type == 'Sigmoid':
		activation = 1.0 / (1 + np.exp(-net_value))
	elif type == "Linear":
		activation = net_value
	elif type == "Hyperbolic Tangent":
		activation = np.tanh(net_value)
	elif type == "Positive Linear":
		activation = np.maximum(0,net_value)
	return activation