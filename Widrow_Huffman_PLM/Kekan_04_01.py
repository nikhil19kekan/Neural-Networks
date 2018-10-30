# Kekan, Nikhilkumar
# 1001-563-734
# 2018-10-08
# Assignment-04-01
import time
from tkinter import *
import tkinter as tk
import numpy as np
import Kekan_04_02
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MainWindow:

    def __init__(self,master):
        # variables of work
        self.weights = list()
        self.samples = list()
        self.R = list()
        self.h = list()
        self.input_file="./stock_data1.txt"
        # variables just to hold values
        self.delay = 10
        self.learning_rate = 0.1
        self.training_sample_size = 80
        self.stride = 1
        self.number_of_iterations = 10

        self.graph_frame = Frame(master, borderwidth=2, relief="sunken")
        self.graph_frame.grid(row=0, column=0, columnspan=2)
        self.graph_frame.rowconfigure(0, weight=1)
        self.graph_frame.columnconfigure(0, weight=1)

        self.figure = plt.figure(figsize=(6, 3))
        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.8])
        self.axes = self.figure.gca()
        self.axes.set_xlabel("Iterations")
        self.axes.set_ylabel("Mean Square Error")
        self.axes.set_ylim(0,2)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.plot_figure = self.canvas.get_tk_widget()
        self.plot_figure.grid(row=0, column=0)

        self.graph_frame1 = Frame(master, borderwidth=2, relief="sunken")
        self.graph_frame1.grid(row=0, columnspan=2, column=3, sticky="nesw")
        self.graph_frame1.rowconfigure(0, weight=1)
        self.graph_frame1.columnconfigure(1, weight=1)

        self.figure1 = plt.figure(figsize=(6, 3))
        self.axes1 = self.figure1.add_axes([0.15, 0.15, 0.6, 0.8])
        self.axes1 = self.figure1.gca()
        self.axes1.set_xlabel("Iterations")
        self.axes1.set_ylabel("Absolute maximum error")
        self.axes1.set_ylim(0, 2)

        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.graph_frame1)
        self.plot_figure1 = self.canvas1.get_tk_widget()
        self.plot_figure1.grid(row=0, column=3, sticky="nesw")

        self.controls_frame = Frame(master, borderwidth=2, relief="sunken")
        self.controls_frame.grid(row=2, columnspan=3, column=1, sticky="nesw")
        self.controls_frame.rowconfigure(0, weight=1)
        self.controls_frame.columnconfigure(0, weight=1)

        self.delay_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL, from_=0,
                                     to_=100, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
                                     highlightcolor="#00FFFF", label="delay",
                                     command=lambda event: self.get_delay_slider_value())
        self.delay_slider.set(self.delay)
        self.delay_slider.bind("<ButtonRelease-1>", lambda event: self.get_delay_slider_value())
        self.delay_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.learning_rate_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0.001,
                                             to_=1, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
                                             highlightcolor="#00FFFF", label="learning rate",
                                             command=lambda event: self.get_learning_rate_slider_value())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.get_learning_rate_slider_value())
        self.learning_rate_slider.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.training_sample_size_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                                    from_=0,
                                                    to_=100, resolution=10, bg="#DDDDDD", activebackground="#FF0000",
                                                    highlightcolor="#00FFFF", label="training sample size",
                                                    command=lambda event: self.get_training_sample_size_slider_value())
        self.training_sample_size_slider.set(self.training_sample_size)
        self.training_sample_size_slider.bind("<ButtonRelease-1>",
                                              lambda event: self.get_training_sample_size_slider_value())
        self.training_sample_size_slider.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.stride_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=1,
                                      to_=100, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
                                      highlightcolor="#00FFFF", label="stride",
                                      command=lambda event: self.get_stride_slider_value())
        self.stride_slider.set(self.stride)
        self.stride_slider.bind("<ButtonRelease-1>", lambda event: self.get_stride_slider_value())
        self.stride_slider.grid(row=1, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        self.number_of_iterations_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                                    from_=1,
                                                    to_=100, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
                                                    highlightcolor="#00FFFF", label="number of iterations",
                                                    command=lambda event: self.get_number_of_iterations_slider_value())
        self.number_of_iterations_slider.set(self.number_of_iterations)
        self.number_of_iterations_slider.bind("<ButtonRelease-1>",
                                              lambda event: self.get_number_of_iterations_slider_value())
        self.number_of_iterations_slider.grid(row=1, column=4, sticky=tk.N + tk.E + tk.S + tk.W)

        self.set_weights_to_zero_button = tk.Button(self.controls_frame, text="set weights to zero", justify="center")
        self.set_weights_to_zero_button.bind("<Button-1>", lambda event: self.set_weights_to_zero_button_callback())
        self.set_weights_to_zero_button.grid(row=2, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.adjust_weights_lms_button = tk.Button(self.controls_frame, text="adjust weights lms", justify="center")
        self.adjust_weights_lms_button.bind("<Button-1>", lambda event: self.adjust_weights_lms_button_callback())
        self.adjust_weights_lms_button.grid(row=2, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.adjust_weights_direct_button = tk.Button(self.controls_frame, text="adjust weights direct",
                                                      justify="center")
        self.adjust_weights_direct_button.bind("<Button-1>", lambda event: self.adjust_weights_direct_button_callback())
        self.adjust_weights_direct_button.grid(row=2, column=3, sticky=tk.N + tk.E + tk.S + tk.W)

        #-----------------------------------------
        self.file_selection_label = tk.Label(self.controls_frame, text="Input File:",
                                              justify="center")
        self.file_selection_label.grid(row=3, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.file_selection_variable = tk.StringVar()
        self.file_slection_dropdown = tk.OptionMenu(self.controls_frame, self.file_selection_variable,
                                                      "./stock_data1.txt", "./stock_data2.txt",
                                                      command=lambda
                                                          event: self.get_file_selection_callback())
        self.file_selection_variable.set("./stock_data1.txt")
        self.file_slection_dropdown.grid(row=3, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        #-----------------------------------------



    def setup_inputs(self):
        self.set_weights_to_zero_button_callback()
        temp = Kekan_04_02.read_csv_as_matrix(self.input_file)
        price_change_list = [temp[i][0] for i in range(len(temp))]
        volume_change_list = [temp[i][1] for i in range(len(temp))]
        print(price_change_list)
        print(volume_change_list)
        price_change_list = Kekan_04_02.normalize_normal(price_change_list)
        volume_change_list = Kekan_04_02.normalize_normal(volume_change_list)
        temp1 = list()
        for i in range(len(price_change_list)):
            temp1.append([price_change_list[i], volume_change_list[i]])
        self.samples = Kekan_04_02.get_samples(self.stride, self.delay, temp1)

    def get_mean_square_error_and_absolute_error(self):
        square_error=0.0
        absolutes_list=list()
        for input_sample in self.samples[ int(self.training_sample_size/100*len(self.samples)) : ]:
            actual=np.dot(self.weights,input_sample[:len(input_sample)-1])
            target=input_sample[-1]
            absolutes_list.append(abs(target-actual))
            square_error += (target-actual)*(target-actual)
        mean_square_error=square_error/len(self.samples[ int(self.training_sample_size/100*len(self.samples)) : ])
        return mean_square_error,max(absolutes_list)

    #adjusting weights
    def adjust_weights_and_plot_lms(self):
        iterations_list=list()
        mean_sqr_list=list()
        absolute_maximum_list=list()
        for j in range(self.number_of_iterations):
            length_of_samples=int(self.training_sample_size/100*len(self.samples))
            for i in range(length_of_samples):
                input_sample = np.array(self.samples[i])
                actual=np.array(self.weights).dot(input_sample[:len(input_sample)-1])
                target=input_sample[-1]
                error=target-actual
                self.weights = self.weights + 2 * self.learning_rate * error * input_sample[:len(input_sample)-1]
            mean_square_error,absolute_maximum_error=self.get_mean_square_error_and_absolute_error()
            print('mean square',mean_square_error)
            print('absolute maximum error',absolute_maximum_error)
            mean_sqr_list.append(mean_square_error)
            absolute_maximum_list.append(absolute_maximum_error)
            iterations_list.append(j)
        self.axes.plot(iterations_list,mean_sqr_list,color='red')
        self.axes1.plot(iterations_list,absolute_maximum_list,color='red')
        self.canvas.draw()
        self.canvas1.draw()

    def plot_direct(self):
        iterations_list = list()
        mean_sqr_list = list()
        absolute_maximum_list = list()
        for i in range(self.number_of_iterations):
            mean_square_error, absolute_maximum_error = self.get_mean_square_error_and_absolute_error()
            print('mean square', mean_square_error)
            print('absolute maximum error', absolute_maximum_error)
            mean_sqr_list.append(mean_square_error)
            absolute_maximum_list.append(absolute_maximum_error)
            iterations_list.append(i)
        print('now plotiing DIRECT')
        self.axes.plot(iterations_list, mean_sqr_list,color='blue')
        self.axes1.plot(iterations_list, absolute_maximum_list,color='blue')
        self.canvas.draw()
        self.canvas1.draw()

    #get slider values
    def get_delay_slider_value(self):
        self.delay=self.delay_slider.get()
        self.set_weights_to_zero_button_callback()

    def get_learning_rate_slider_value(self):
        self.learning_rate=self.learning_rate_slider.get()
        self.setup_inputs()

    def get_training_sample_size_slider_value(self):
        self.training_sample_size=self.training_sample_size_slider.get()

    def get_stride_slider_value(self):
        self.stride=self.stride_slider.get()
        self.setup_inputs()

    def get_number_of_iterations_slider_value(self):
        self.number_of_iterations=self.number_of_iterations_slider.get()

    #callbacks
    def set_weights_to_zero_button_callback(self):
        self.weights=np.zeros(shape=(1,int((self.delay+1)*2+1)))
        self.axes.cla()
        self.axes1.cla()
    def adjust_weights_lms_button_callback(self):
        self.adjust_weights_and_plot_lms()
    def adjust_weights_direct_button_callback(self):
        self.R=Kekan_04_02.calculate_R(self.samples[:int(self.training_sample_size / 100 * len(self.samples))])
        self.h=Kekan_04_02.calculate_h(self.samples[:int(self.training_sample_size / 100 * len(self.samples))])
        R_inverse=np.linalg.inv(self.R)
        self.weights=R_inverse.dot(self.h.transpose())
        self.weights=np.transpose(self.weights)
        self.plot_direct()
    def get_file_selection_callback(self):
        self.input_file=self.file_selection_variable.get()
        self.set_weights_to_zero_button_callback()
        self.setup_inputs()
root = Tk()
main_window = MainWindow(root)
root.mainloop()