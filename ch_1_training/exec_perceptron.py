#exec_perceptron.py

### view Iris results
import pandas as pd 

# load the Iris dataset into a DataFrame object and print the last 5 lines via tail() to check to make sure the data was loaded correctly
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#print(df.tail()) 

df.iloc[0:100, 4].values

import matplotlib.pyplot as plt 
import numpy as np 

# extract the first 100 class labels that corespond to the first 50 instances of both Iris types
y = df.iloc[0:100, 4].values;

#convert the two classes into integer values (1, -1)
y = np.where(y == 'Iris-setosa', -1, 1)

#extract the 1st and 3rd attributes (sepal length & petal length) into the feature matrix X
X = df.iloc[0:100, [0, 2]].values

# plot the variables
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.ylabel('sepal length')
plt.xlabel('petal length')

# Plot the output, making sure not to block subsequent code
plt.legend(loc='upper left')
plt.show(block=False);


# ----------

ppn = Perceptron(eta=0.1, n_iter=10)