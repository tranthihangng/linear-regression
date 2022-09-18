import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fun(x):
    return x**2
def fun_gra(x):
    return 2*x
learning_rates = [0.01,0.02,1.3]
for learning_rate in learning_rates:
    x_list = []
    y_list = []
    x = 20
    x_list.append(x)
    y_list.append(fun(x))
    for i in range (100):
        x = x - learning_rate*fun_gra(x)
        x_list.append(x)
        y_list.append(fun(x))
    r = max(x_list)
    x = np.linspace(-r,r,100)
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(1,2,1)
    ax1.plot(x,fun(x))
    ax1.scatter(x_list,y_list,c = 'r')

    ax2 = plt.subplot(1,2,2)
    ax2.plot(y_list, c= 'g')
    plt.show()