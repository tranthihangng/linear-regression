import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

data_square = pd.read_csv('data_square.csv').values
n = data_square.shape[0]
x = data_square[:,0].reshape(-1,1)
x = preprocessing.scale(x)
y = data_square[:,1].reshape(-1,1)
y = preprocessing.scale(y)
x_nw = np.hstack((np.ones((n,1)),x,np.square(x)))
plt.figure(figsize=(12,6))
ax1 = plt.subplot(2,2,1)
ax1.scatter(x,y)
plt.xlabel('m2')
plt.ylabel('price')

w = np.random.rand(3,1)
numberOfIteration = 100
cost = np.zeros((numberOfIteration,1))
idl = [i for i in range(numberOfIteration)]
learning_rate = 0.008
for i in range(numberOfIteration):
    y_hat = np.dot(x_nw, w)
    cost[i] = 0.5/n * np.squeeze(np.dot((y_hat-y).T,(y_hat-y)))
    dw = np.dot(x_nw.T,(y_hat-y))
    w = w - dw* learning_rate

ax2 = plt.subplot(2,2,2)
ax2.plot(cost)
plt.xlabel('iteration')
plt.ylabel('loss fun')


w_0 = float(w[0])
w_1 = float(w[1])
w_2 = float(w[2])
y_pred = w_2*(x**2)+w_1*x+w_0
ax3 = plt.subplot(2,2,3)
ax3.scatter(x,y)
ax3.plot(x, y_pred, c = 'r')
plt.xlabel('m2')
plt.ylabel('gia')



plt.show()


