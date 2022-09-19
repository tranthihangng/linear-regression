import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data_linear.csv')
#print(data)
x = data['Diện tích']
y = data['Giá']
plt.figure(figsize=(12,6))
ax1 = plt.subplot(2,2,1)
ax1.scatter(x,y,c = 'b')
plt.xlabel('square')
plt.ylabel('price')

x = x.to_numpy().reshape(-1,1)
y = y.to_numpy().reshape(-1,1)
n = x.shape[0]

x_nw = np.hstack((np.ones((n,1)),x))
w = np.array([0., 1.]).reshape(-1,1)
numberOfIteration = 100
cost = np.zeros((numberOfIteration,1))
idl = [i for i in range(numberOfIteration)]
learning_rate = 0.00002
for i in range(numberOfIteration):
    y_hat = np.dot(x_nw,w)
    cost[i] = 0.5/n * np.dot((y_hat-y).T,(y_hat-y))
    dw = 1/n * np.dot(x_nw.T,y_hat-y)
    w = w -dw *learning_rate
    if i%10==1:
        print('step {}, cost{} '.format(i,cost[i]))
ax2 = plt.subplot(2,2,2)
ax2.scatter(idl[1:], cost[1:])
plt.ylabel('loss fun')
plt.xlabel('iteration')

ax3 = plt.subplot(2,2,3)
ax3.scatter(x,y)
x_ = np.array([[1,30],[1,100]])
y_ = np.dot(x_,w)
plt.xlabel('m2')
plt.ylabel('price')
ax3.plot(x_[:,1],y_, c = 'r')
plt.show()






