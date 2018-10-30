import numpy as np
import time
def read_csv_as_matrix(file_name):
    # Each row of data in the file becomes a row in the matrix
    # So the resulting matrix has dimension [num_samples x sample_dimension]
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
    return data

#to normalize input vector passed to get values between -1 and 1
def normalize_normal(vector):
    mini=min(vector)
    maxi=max(vector)
    #print('minimum:',mini,'maximum:',maxi)
    for i in range(len(vector)):
        vector[i]=((vector[i]-mini)/(maxi-mini))*2-1
    return list(vector)

#get R
def calculate_R(input_vector):
    rows=len(input_vector)
    cols=len(input_vector[0])
    R=np.zeros(shape=(cols-1,cols-1))
    for i in range(rows):
        input=np.array([input_vector[i][:cols-1]])
        temp=input.transpose().dot(input)
        R += temp
    R=R/rows
    return R

#calculate h
def calculate_h(vector):
    rows = len(vector)
    cols = len(vector[0])
    h=np.zeros(shape=(1,cols-1))
    for i in range(len(vector)):
        input_sample=list(vector[i])
        target=input_sample[-1]
        input_sample=input_sample[:len(input_sample)-2]
        input_sample.append(1)
        input_sample = np.array(input_sample).dot(target)
        h[0] += input_sample
    print(h,rows)
    h=h/rows
    return h

#get list of all samples that can be fed directly to neuron
def get_samples(stride,delay,vector):
    all_samples=list()
    counter=0
    while((counter+delay+2)<len(vector)):
        sample1 = list()
        sample2 = list()
        inside_counter=0
        while(inside_counter < (delay+1)):
            sample1.append(vector[counter + inside_counter][0])
            inside_counter+=1
        inside_counter=0
        while (inside_counter < (delay + 1)):
            sample2.append(vector[counter + inside_counter][1])
            inside_counter += 1
        sample=sample1+sample2
        sample.append(1)
        sample.append(vector[counter + inside_counter][0])
        all_samples.append(sample)
        counter+=stride
    return all_samples


