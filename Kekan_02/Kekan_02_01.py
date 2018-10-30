# Kekan, Nikhilkumar
# 1001-563-734
# 2018-09-24
# Assignment-02-01
import Kekan_02_02
import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib

matplotlib.use ('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

class MainWindow (tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    Nikhilkumar Kekan 2018_06_03
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__ (self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame (self)
        self.master_frame.grid (row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.rowconfigure (0, weight=1, minsize=400)
        ## handles inside graph columnsize
        self.columnconfigure (0, weight=1, minsize=600)
        self.columnconfigure (1, weight=1, minsize=600)

        self.master_frame.rowconfigure (3, weight=1, minsize=10, uniform='xx')
        self.master_frame.columnconfigure (0, weight=20, minsize=300, uniform='xx')
        self.master_frame.columnconfigure (1, weight=20, minsize=300, uniform='xx')
        # create all the widgets

        self.left_frame = tk.Frame (self.master_frame)
        self.left_frame.columnconfigure (0, weight=10, minsize=300)
        self.left_frame.columnconfigure (1, weight=10, minsize=300)
        self.left_frame.grid (row=2, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_activation_functions = LeftFrame (self, self.left_frame, debug_print_flag=self.debug_print_flag)

class LeftFrame:
    """
    This class creates and controls the widgets and figures in the left frame which
    are used to display the activation functions.
    Kekan Nikhilkumar 2018_09_24
    """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        ###############
        self.xx = None
        self.yy = None
        self.w1 = 1
        self.w2 = 1
        self.bias = 0.0

        self.random = [[np.random.uniform (-10, 10), np.random.uniform (-10, 10)] for i in range (4)]
        self.random[0].append (-1)
        self.random[1].append (-1)
        self.random[2].append (1)
        self.random[3].append (1)
        self.plot_frame = tk.Frame (self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid (row=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight = 1
        self.bias = 0.0
        self.activation_type = "Symmetric Hard Limit"
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.columnconfigure (0, weight=10)
        master.columnconfigure (1, weight=10)
        master.columnconfigure (2, weight=10)

        self.plot_frame = tk.Frame (self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid (row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.columnconfigure (0, weight=10)
        self.plot_frame.columnconfigure (1, weight=10)
        self.figure = plt.figure ("")
        self.axes = self.figure.add_axes ([0.10, 0.15, 0.8, 0.8])
        self.axes.set_xlabel ('Input')
        self.axes.set_ylabel ('Output')
        self.axes.set_title ("")
        plt.xlim (self.xmin, self.xmax)
        plt.ylim (self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg (self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget ()
        self.plot_widget.grid (row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        #Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame (self.master)
        self.controls_frame.grid (row=1, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the control widgets such as sliders and selection boxes
        #########################################################################
        self.w1_slider = tk.Scale (self.controls_frame, variable=tk.DoubleVar (), orient=tk.HORIZONTAL, from_=-10.0,
                                   to_=10.0, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
                                   highlightcolor="#00FFFF", label="w1",
                                   command=lambda event: self.w1_slider_callback ())
        self.w1_slider.set (self.w1)
        self.w1_slider.bind ("<ButtonRelease-1>", lambda event: self.w1_slider_callback ())
        self.w1_slider.grid (row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.w2_slider = tk.Scale (self.controls_frame, variable=tk.DoubleVar (), orient=tk.HORIZONTAL, from_=-10.0,
                                   to_=10.0, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
                                   highlightcolor="#00FFFF", label="w2",
                                   command=lambda event: self.w2_slider_callback ())
        self.w2_slider.set (self.w2)
        self.w2_slider.bind ("<ButtonRelease-1>", lambda event: self.w2_slider_callback ())
        self.w2_slider.grid (row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.bias_slider = tk.Scale (self.controls_frame, variable=tk.DoubleVar (), orient=tk.HORIZONTAL, from_=-10.0,
                                     to_=10.0, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
                                     highlightcolor="#00FFFF", label="Bias",
                                     command=lambda event: self.bias_slider_callback ())
        self.bias_slider.set (self.bias)
        self.bias_slider.bind ("<ButtonRelease-1>", lambda event: self.bias_slider_callback ())
        self.bias_slider.grid (row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.button_train = tk.Button (self.controls_frame, text="Train", justify="center")
        self.button_train.bind ("<Button-1>", lambda event: self.train_callback ())
        self.button_train.grid (row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.button_random_data = tk.Button (self.controls_frame, text="Create random data", justify="center")
        self.button_random_data.bind ("<Button-1>", lambda event: self.random_callback ())
        self.button_random_data.grid (row=1, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################
        self.label_for_activation_function = tk.Label (self.controls_frame, text="Activation Function Type:",
                                                       justify="center")
        self.label_for_activation_function.grid (row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar ()
        self.activation_function_dropdown = tk.OptionMenu (self.controls_frame, self.activation_function_variable,
                                                           "Symmetric Hard Limit", "Linear", "Hyperbolic Tangent",
                                                           command=lambda
                                                               event: self.activation_function_dropdown_callback ())
        self.activation_function_variable.set ("Symmetric Hard Limit")
        self.activation_function_dropdown.grid (row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
    #this function is to display things on UI
    def display_activation_function(self):
        try:
            input_values = np.linspace (-10, 10, 100)
            activation = Kekan_02_02.calculate_activation_function (self.w1, self.w2, self.bias, input_values)
            self.axes.set_xlabel ('Input')
            self.axes.set_ylabel ('Output')

            self.xx = np.linspace (-10, 10, 100)
            self.yy = np.linspace (-10, 10, 100)
            self.xx, self.yy = np.meshgrid (self.xx, self.yy)
            zz = self.w1 * self.xx + self.w2 * self.yy + self.bias
            if(self.activation_type == 'Symmetric Hard Limit'):
                zz[zz<0]=-1
                zz[zz>0]=+1
            elif(self.activation_type == 'Hyperbolic Tangent'):
                zz=np.tanh(zz)
            else:
                zz=zz
        except:
            print('exception occurred in values calculation')
        self.axes.cla()
        self.axes.pcolormesh(self.xx, self.yy, zz, cmap=(matplotlib.colors.ListedColormap(['r', 'g'])))
        for i in range (4):
            self.axes.scatter (self.random[i][0], self.random[i][1],
                                   color="blue" if self.random[i][2] == 1 else "yellow")
        self.axes.plot (input_values, activation,color="black")
        self.axes.xaxis.set_visible (True)
        plt.xlim (self.xmin, self.xmax)
        plt.ylim (self.ymin, self.ymax)
        plt.title (self.activation_type)
        self.canvas.draw()
    #this is to train a single perceptron
    def train_callback(self):
        output = 0
        try:
            for j in range (100):
                print ('epoch:', j)
                for i in range (4):
                    try:
                        n = self.w1 * self.random[i][0] + self.w2 * self.random[i][1] + self.bias
                        if (self.activation_type == 'Symmetric Hard Limit'):
                            if (n < 0):
                                output = -1
                            elif (n >= 0):
                                output = 1
                        elif (self.activation_type == 'Hyperbolic Tangent'):
                            output = np.tanh (n)
                        elif (self.activation_type == 'Linear'):
                            output = n
                        error = self.random[i][2] - output
                        self.w1 += (error * self.random[i][0])
                        self.w2 += (error * self.random[i][1])
                        self.bias += error
                    except:
                        raise Exception('BreakIt')
                self.display_activation_function ()
        except 'BreakIt':
            pass
        if(self.w1>=-10 and self.w1<=10):
            self.w1_slider.set(self.w1)
        if (self.w2 >= -10 and self.w2 <= 10):
            self.w2_slider.set(self.w2)
        if (self.bias >= -10 and self.bias <= 10):
            self.bias_slider.set(self.bias)
            
    #button to generate 4 random numbers
    def random_callback(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
        for i in range (4):
            self.random[i][0] = np.random.uniform (-10, 10)
            self.random[i][1] = np.random.uniform (-10, 10)
        self.display_activation_function ()
#slider to adjust weight 1
    def w1_slider_callback(self):
        self.w1 = np.float (self.w1_slider.get ())
        self.display_activation_function ()
#slider to adjust weight 2
    def w2_slider_callback(self):
        self.w2 = np.float (self.w2_slider.get ())
        self.display_activation_function ()
#slider to adjust bias
    def bias_slider_callback(self):
        self.bias = np.float (self.bias_slider.get ())
        self.display_activation_function ()

    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get ()
        self.display_activation_function ()


def close_window_callback(root):
    if tk.messagebox.askokcancel ("Quit", "Do you really wish to quit?"):
        root.destroy ()


main_window = MainWindow (debug_print_flag=False)
main_window.wm_state ('zoomed')
main_window.title ('Assignment_02 --  Kekan')
main_window.minsize (800, 600)
main_window.mainloop ()
