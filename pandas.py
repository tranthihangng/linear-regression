import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data_linear.csv')
print(data.shape)
print(data.head(10))
cor_2 = data.corr()
print(cor_2)

# write csv file

df = pd.DataFrame(cor_2)
df.to_csv('data_linear1.csv')

# scatter plot
x = data['Diện tích']
y = data['Giá']
plt.xlabel('S')
plt.ylabel('P')
plt.scatter(x,y,c = 'g')
plt.show()