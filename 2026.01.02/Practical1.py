import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.linear_model import LinearRegression

#Generate random numbers
# x = np.random.randn(5)
# print(x)

# Get mean of the random numbers
# x = np.random.randn(5000)
# print(np.mean(x))

#Create a matrix
# x = np.random.randn(5, 2)
# print(x)

# When using seed command every time can be get the same value
# np.random.seed(10)
# x = np.random.randn(4)
# print(x)

m =2
N =200
c = 1
mu = 1
std =1


# x = np.random.randn(N)
# y = m*x +c
# print(y)

def visualize(X,y,y_hat):
    plt.scatter(X[:,0], y,'b.')
    plt.scatter(X[:,0], y_hat,'r.')
    plt.title("Data")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


#x = np.random.randn(N)

#can be move the y vlues using std amd m
# x = mu + std * np.random.randn(N) 

# adding thr noice
# noice = np.random.randn(N) * 1
# x = mu + std * np.random.randn(N) 
# y = m*x +c + noice
# print(y)
# visualize(x,y)

#call the method
def getX(x):
    b= np.ones([N,1])
    X = np.append(x,b,axis = 1)
    return X

noice = np.random.randn(N) * 1
x = mu + std * np.random.randn(N) 
x = x.reshape([N,1])
y = m*x +c + noice

# print(x.shape)
X = getX(x)
# print(X.shape)

#calculate the thita
def calTheta(X,y):
    #calculate Taranspose
    TranceX = X.T

    # inverse ( TranceX * X)
    InveseX = np.linalg.inv(np.dot(TranceX,X))

    # TranceX * Y
    z= np.dot(InveseX,y)

    # calculate thita 

    thita  = np.dot(InveseX,z)
    return thita

#calculat prediction 
def calPedictions(theta,X):
    return np.dot(X,theta)

def calError(y, y_hat):
    err = (y - y_hat)**2
    err = err.mean()
    print(err.shape)
    return err

theta  = calTheta(X,y)
print(theta)
reg = LinearRegression(). fit(x,y)
print(reg.coef_)
print(reg.intercept_)

y_hat  = calPedictions(theta,X)
err = calError(y, y_hat)
#visualize(X, y, y_hat)
print(X.shape)
print(y.shape)






#Create Array
# a = np.array([[1,2,3],[1,1,1]])
# print(a.shape)  #shape of the array


# b = np.array([[4,3,4]])
# #a.reshape(1,1)   should be reshape beacaue the arrays are in two dimentions
# print(b.shape)


# #Append the  two arrays
# array = np.append(a,b, axis =0) 
# print(array)

#Get the element of array
# a = np.array([[23,42,2],[4,234,3]])
# print(a)

# print(a[0])
# print(a[:,1])
# print(a[1:,1:])

