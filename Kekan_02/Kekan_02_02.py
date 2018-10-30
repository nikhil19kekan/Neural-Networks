# Kekan Nikhilkumar
# 1001-563-734
# 2018-09-09
# Assignment-01-02
import numpy as np
# This module calculates the activation function
def calculate_activation_function(w1,w2,bias,input_x):
	output_y=(-bias-(input_x*w1))/w2
	return output_y
