# Kekan, Nikhilkumar
# 1001-563-734
# 2018-09-24
# Assignment-02-01
import sys
import os
import Kekan_03.Kekan_03_02
import time

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib
import random
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)


class MainWindow(tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    Nikhilkumar Kekan 2018_06_03
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.rowconfigure(0, weight=1, minsize=400)
        ## handles inside graph columnsize
        self.columnconfigure(0, weight=1, minsize=600)
        self.columnconfigure(1, weight=1, minsize=600)

        self.master_frame.rowconfigure(3, weight=1, minsize=10, uniform='xx')
        self.master_frame.columnconfigure(0, weight=20, minsize=300, uniform='xx')
        self.master_frame.columnconfigure(1, weight=20, minsize=300, uniform='xx')
        # create all the widgets

        self.left_frame = tk.Frame(self.master_frame)
        self.left_frame.columnconfigure(0, weight=10, minsize=300)
        self.left_frame.columnconfigure(1, weight=10, minsize=300)
        self.left_frame.grid(row=2, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_activation_functions = LeftFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)


class LeftFrame:
    """
    This class creates and controls the widgets and figures in the left frame which
    are used to display the activation functions.
    Kekan Nikhilkumar 2018_09_24
    """

    def __init__(self, root, master, debug_print_flag=False):

        self.input_array = []
        self.filepath = "./data/mnist_images/";
        files = os.listdir(self.filepath)  # get all files in specified directory

        for i in range(1000):
            self.vector = np.divide(Kekan_03.Kekan_03_02.read_one_image_and_convert_to_vector(self.filepath + files[i]), 127.5) - 1
            self.vector = np.insert(self.vector, 784, 1, axis=0)  # add bias to the array
            self.input_array.append(self.vector)
        self.input_array = np.array(self.input_array)


        self.xmin = 0
        self.xmax = 100
        self.ymin = 0.0
        self.ymax = 2.0
        self.learning_rate = 0.1
        self.transfer_function_type = "Symmetrical Hard Limit"
        self.learning_method_type = "Delta Rule"
        self.weight_matrix = self.create_random_weights()
        self.target_values = self.get_target_values()
        self.random_numbers = np.random.permutation(self.input_array.shape[0])
        self.training_data = self.input_array[self.random_numbers[:800]]
        self.test_data = self.input_array[self.random_numbers[800:]]
        self.target_training_data = self.target_values[self.random_numbers[:800]]
        self.target_test_data = self.target_values[self.random_numbers[800:]]
        self.xplot = 1

        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, sticky=tk.N + tk.E + tk.S + tk.W)

        master.columnconfigure(0, weight=10)
        master.columnconfigure(1, weight=10)
        master.columnconfigure(2, weight=10)

        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.columnconfigure(0, weight=10)
        self.plot_frame.columnconfigure(1, weight=10)
        self.figure = plt.figure("")
        self.axes = self.figure.add_axes([0.10, 0.15, 0.8, 0.8])
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the control widgets such as sliders and selection boxes
        #########################################################################
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL, from_=0.001,
                              to_=1, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
                              highlightcolor="#00FFFF", label="Alpha",
                              command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.learn = tk.Button(self.controls_frame, text="Learn", justify="center")
        self.learn.bind("<Button-1>", lambda event: self.train_callback())
        self.learn.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.randomize_weights = tk.Button(self.controls_frame, text="Randomize weights", justify="center")
        self.randomize_weights.bind("<Button-1>", lambda event: self.randomize_weights_callback())
        self.randomize_weights.grid(row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.display_confusion_matrix = tk.Button(self.controls_frame, text="Display Confusion Matrix", justify="center")
        self.display_confusion_matrix.bind("<Button-1>", lambda event: self.display_confusion_matrix())
        self.display_confusion_matrix.grid(row=1, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################
        self.learning_method_label = tk.Label(self.controls_frame, text="Learning Method:",
                                              justify="center")
        self.learning_method_label.grid(row=1, column=4, sticky=tk.N + tk.E + tk.S + tk.W)

        self.learning_method_variable = tk.StringVar()
        self.learning_method_dropdown = tk.OptionMenu(self.controls_frame, self.learning_method_variable,
                                                          "Delta Rule", "Filtered Learning", "Unsupervised Learning",
                                                      command=lambda
                                                              event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set("Delta Rule")
        self.learning_method_dropdown.grid(row=1, column=5, sticky=tk.N + tk.E + tk.S + tk.W)

        self.transfer_function_label = tk.Label(self.controls_frame, text="Transfer Function:",
                                                justify="center")
        self.transfer_function_label.grid(row=1, column=6, sticky=tk.N + tk.E + tk.S + tk.W)

        self.transfer_function_variable = tk.StringVar()
        self.transfer_function_dropdown = tk.OptionMenu(self.controls_frame, self.transfer_function_variable,
                                                      "Symmetric Hard Limit", "Linear", "Hyperbolic Tangent",
                                                      command=lambda
                                                          event: self.transfer_function_dropdown_callback())
        self.transfer_function_variable.set("Symmetric Hard Limit")
        self.transfer_function_dropdown.grid(row=1, column=7, sticky=tk.N + tk.E + tk.S + tk.W)
        self.create_image_matrix()


    def get_target_values(self):
        target_val = []
        target = [0 for i in range(97)]
        target += [1 for i in range(116)]
        target += [2 for i in range(99)]
        target += [3 for i in range(93)]
        target += [4 for i in range(105)]
        target += [5 for i in range(92)]
        target += [6 for i in range(94)]
        target += [7 for i in range(117)]
        target += [8 for i in range(87)]
        target += [9 for i in range(100)]

        for i, val in enumerate(target):
            temp_array = np.zeros(shape=(10,1))
            temp_array[val] = 1
            target_val.append(temp_array)

        target_val = np.array(target_val)
        return target_val

    #This method will fetch the activation function selected from dropdown
    def get_transfer_function(self):
        self.transfer_function_type = self.transfer_function_variable.get()
        self.axes.cla()
        self.xplot = 1

    # This method will fetch the learning method selected from dropdown
    def get_learning_method(self):
        self.learning_method_type = self.learning_method_variable.get()
        self.axes.cla()
        self.xplot = 1

    #This method will fetch the learning rate (alpha) from the slider
    def get_learning_rate(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.axes.cla()
        self.xplot = 1

    def create_image_matrix(self):
        #self.samples=np.random.random((785,800))
        #self.t=np.random.random((10,800))
        for f in os.listdir("./data/mnist_images"):
            c=int(f[0])
            current_t=np.zeros(shape=(10,1))
            current_t[c][0]=1
            self.t.append(current_t)

            image_out=np.array(Kekan_03.Kekan_03_02.read_one_image_and_convert_to_vector(str(f)))
            final_imag=np.subtract(np.divide(np.array(image_out),127.5),1.0)
            current_p=np.array(final_imag)
            self.samples.append(current_p)

    def train_callback(self):
        random_indices=np.random.permutation(range(1000))
        self.axes.cla()
        for epochs in range(100):
            for i in random_indices[:800]:
                t_temp=np.array( [ float(val) for val in self.t[i] ] )
                p=np.array([float(k) for k in self.samples[i]])
                self.net=self.weight_matrix.dot(p)
                if (self.learning == 'Filtered Learning'):
                    self.weight_matrix=Kekan_03.Kekan_03_02.filtered_learning(self.weight_matrix, self.alpha, self.alpha, t_temp, p)
                elif (self.learning == 'Delta Rule' and self.transfer_function == 'Symmetric Hard Limit'):
                    self.weight_matrix=Kekan_03.Kekan_03_02.delta_rule(self.weight_matrix,self.alpha,t_temp, self.symmetric_hard_limit(self.net), p)
                elif (self.learning == 'Delta Rule' and self.transfer_function == 'Linear'):
                    self.weight_matrix = Kekan_03.Kekan_03_02.delta_rule(self.weight_matrix, self.alpha, t_temp, self.normalize(self.net), p)
                elif (self.learning == 'Delta Rule' and self.transfer_function == 'Hyperbolic Tangent'):
                    self.weight_matrix = Kekan_03.Kekan_03_02.delta_rule(self.weight_matrix, self.alpha, t_temp, np.tanh(self.net), p)
                elif (self.learning == 'Unsupervised Learning' and self.transfer_function == 'Symmetric Hard Limit'):
                    self.weight_matrix = Kekan_03.Kekan_03_02.unsupervised(self.weight_matrix, self.alpha, self.symmetric_hard_limit(self.net), p)
                elif (self.learning == 'Unsupervised Learning' and self.transfer_function == 'Linear'):
                    self.weight_matrix = Kekan_03.Kekan_03_02.unsupervised(self.weight_matrix, self.alpha, self.net, p)
                elif (self.learning == 'Unsupervised Learning' and self.transfer_function == 'Hyperbolic Tangent'):
                    self.weight_matrix = Kekan_03.Kekan_03_02.unsupervised(self.weight_matrix, self.alpha, np.tanh(self.net), p)
            #applying test data now to adjusted weights to calculate the error
            self.error=0
            for k in random_indices[800:]:
                curr_sample = np.array([float(num) for num in self.samples[k]]).transpose()
                a = self.weight_matrix.dot(curr_sample)
                if(self.transfer_function=='Symmetric Hard Limit'):
                    self.net=self.symmetric_hard_limit(a)
                elif(self.transfer_function=='Linear'):
                    self.net=a
                elif(self.transfer_function=='Hyperbolic Tangent'):
                    self.net=np.tanh(a)
                self.net=self.normalize(self.net)
                if(np.array_equal(a,self.t[k]) ==  False):
                    self.error=self.error+1
            print('images correctly predicted',200-self.error)
            self.error=self.error/200
            self.display_activation_function(epochs,self.error)

    def symmetric_hard_limit(self,net):
        net[net < 0] = -1
        net[net >= 0] = 1
        return net

    def normalize(self,mat):
        max_val_index = np.argmax(mat, axis=0)
        mat=np.zeros(shape=np.shape(mat))
        mat[max_val_index]=1
        return mat

    def randomize_weights_callback(self):
        #self.weight_matrix = [[float("%.4f"%random.uniform(-0.001, 0.001)) for j in range(785)] for i in range(10)]
        self.weight_matrix = np.array([[float(random.uniform(-0.001, 0.001)) for j in range(785)] for i in range(10)])
        self.display_activation_function()

    def display_confusion_matrix(self):
        self.display_activation_function()

    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())
        #self.display_activation_function()

    def learning_method_dropdown_callback(self):
        self.learning = self.learning_method_variable.get()
        #self.display_activation_function()

    def transfer_function_dropdown_callback(self):
        self.transfer_function = self.transfer_function_variable.get()
        #self.display_activation_function()

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Assignment_02 --  Kekan')
main_window.minsize(800, 600)
main_window.mainloop()