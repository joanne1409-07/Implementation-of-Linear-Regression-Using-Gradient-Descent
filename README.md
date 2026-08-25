# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Pandas, and StandardScaler, then define the Linear Regression function.
2. Read the dataset from 50_Startups.csv and separate the input features X and target variable y.
3. Standardize the input features and target values using StandardScaler.
4. Initialize the parameters (theta) and update them iteratively using Gradient Descent for the given number of iterations.
5. Predict the output for new input data by scaling it and calculating the dot product with the learned theta values.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: RAGUL.B
RegisterNumber: 212225240112
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
*/
```

## Output:
<img width="1012" height="721" alt="image" src="https://github.com/user-attachments/assets/69ba5122-ca35-44c7-8459-ea911626461e" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
