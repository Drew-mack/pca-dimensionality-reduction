import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# a quick walkthrough of feature normalization

## make 10 samples with 2 features each
X = np.zeros((2, 10))
phi = np.random.rand(10)
noise = np.random.rand(10)
X[0, :] = 3 * phi
X[1, :] = phi + 0.3 * noise
D = X.shape[0] # number of features
N = X.shape[1] # number of samples



## plot the original data
fig, ax = plt.subplots()
ax.scatter(X[0, :], X[1, :], color='black')
plt.title('Original data')
plt.show()
plt.close(fig)



## normalize each sample by its l2 norm
print("################################### Now in L2 normalization", "\n")
norm_X = np.sum(X ** 2, 0) ** 0.5 # one norm per column
norm_X = norm_X.reshape(1, -1)
print("L2 norm of each data instance: ", norm_X)
print("\n")

normalized_X = X / norm_X # divide each column by its own norm
# print("L2 normalized data instances: ", normalized_X)
#print("\n")
## plot the normalized data
fig, ax = plt.subplots()
ax.scatter(normalized_X[0, :], normalized_X[1, :], color='black')
plt.title('L2-normalized data')
plt.show()
plt.close(fig)

norm_normalized_X = np.sum(normalized_X ** 2, 0) ** 0.5
norm_normalized_X = norm_normalized_X.reshape(1, -1)
print("L2 norm of each data instance after L2 normalization: ", norm_normalized_X)
print("\n")



## compute the mean
print("################################### Now in the mean", "\n")
mu = np.mean(X, 1).reshape(-1, 1) # one mean per feature
print("mu: ", mu)
print("\n")

fig, ax = plt.subplots()
ax.scatter(X[0, :], X[1, :], color='black')
ax.scatter(mu[0], mu[1], color='red') # mark the mean in red
plt.title('Original data and its mean')
plt.show()
plt.close(fig)

bar_X = X - mu # subtract the mean from each sample
print("mu of bar_X: ", np.mean(bar_X, 1).reshape(-1, 1))
print("mu bar_X is almost 0, with some numerical error")
print("\n")



## compute the covariance matrix
print("################################### Now in the covariance matrix", "\n")

Sigma = np.cov(X, bias=True)
print("Covariance matrix is: ", Sigma)
print("\n")



## standardize each feature with z-scores
print("################################### Now in Z-score", "\n")
std_X = np.std(X, axis = 1, keepdims = True)
print("Standard deviation of X: ", std_X)
print("\n")

Z = (X - mu) / std_X
std_Z = np.std(Z, axis = 1, keepdims = True)
mu_Z = np.mean(Z, 1).reshape(-1, 1)
print("mu of Z: ", mu_Z)
print("mu of Z is almost 0, with some numerical error")
print("\n")
print("Standard deviation of Z: ", std_Z)
print("Standard deviation of Z is 1")
print("\n")

fig, ax = plt.subplots()
ax.scatter(Z[0, :], Z[1, :], color='black')
plt.title('Z-score data')
plt.show()
plt.close(fig)


## whiten the data
print("################################### Now in whitening", "\n")
Sigma_inv = np.linalg.inv(Sigma)
Lambda, Q = np.linalg.eigh(Sigma_inv)
Lambda = np.diag(Lambda ** 0.5)
Sigma_inv_05 = np.matmul(Lambda, Q.transpose())
Z = np.matmul(Sigma_inv_05, X - mu)
print("mu of Z: ", np.mean(Z, 1).reshape(-1, 1))
print("mu of Z is almost 0, with some numerical error")
print("\n")
print("Standard deviation of Z: ", np.std(Z, axis = 1, keepdims = True))
print("Standard deviation of Z is 1")
print("\n")
Sigma_Z = np.cov(Z, bias=True)
print("Covariance matrix of Z: ", Sigma_Z)
print("Covariance matrix of Z is an identity matrix")
print("\n")

fig, ax = plt.subplots()
ax.scatter(Z[0, :], Z[1, :], color='black')
plt.title('Whitened data')
plt.show()
plt.close(fig)


